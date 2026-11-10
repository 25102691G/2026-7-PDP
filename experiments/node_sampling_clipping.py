"""Email-only C=3 per-subgraph clipping, zero noise. No privacy guarantee.

Existing utility/source modules are reused unchanged. Raw gradient/loss
diagnostics and weights stay in local restricted internal directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import node_sampling_utility as u
import torch
from experiments.privacy import PrivateSGD

CONFIG = dict(u.CONFIG, clip_gradients=True, clip_norm=3.0,
              gaussian_std_on_sum=0.0, gaussian_std_on_mean=0.0,
              batch_policy="reuse_exact_saved_unclipped_batches",
              clipping_tolerance=1e-5)


def joint_norm(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


class ObservedPrivateSGD(PrivateSGD):
    """Instrument original clipping/noise-zero optimizer without changing it."""

    def __init__(self, params, *, l2_norm_clip, minibatch_size, lr):
        super().__init__(params, l2_norm_clip=l2_norm_clip, noise_multiplier=0.0,
                         minibatch_size=minibatch_size, microbatch_size=1,
                         lr=lr, momentum=0.0)
        self.current_samples = []
        self.records = []

    def parameters(self):
        return [p for group in self.param_groups for p in group["params"] if p.requires_grad]

    def accumulated(self):
        return [g for group in self.param_groups for g in group["accum_grads"] if g is not None]

    def zero_accum_grad(self):
        super().zero_accum_grad()
        self.current_samples = []

    def microbatch_step(self):
        gradients = [p.grad for p in self.parameters() if p.grad is not None]
        raw_norm = joint_norm(gradients)
        if not math.isfinite(raw_norm):
            raise ValueError("non-finite per-subgraph joint gradient")
        before = [g.clone() for g in self.accumulated()]
        legacy_norm = super().microbatch_step()
        if not math.isclose(raw_norm, legacy_norm, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("joint gradient observation disagrees with original optimizer")
        coefficient = min(self.l2_norm_clip / (legacy_norm + 1e-6), 1.0)
        post_norm = joint_norm([g.detach() * coefficient for g in gradients])
        # Also observe the actual float32 contribution added to the accumulator.
        contribution_norm = joint_norm([new.double()-old.double() for new, old in zip(self.accumulated(), before)])
        bound_ok = max(post_norm, contribution_norm) <= self.l2_norm_clip + CONFIG["clipping_tolerance"]
        if not bound_ok:
            raise ValueError("per-subgraph clipping exceeded its joint L2 bound")
        self.current_samples.append(dict(pre_norm=raw_norm, post_norm=post_norm,
                                         accumulator_contribution_norm=contribution_norm,
                                         coefficient=coefficient, bound_ok=bound_ok))
        return legacy_norm

    def step(self, closure=None):
        if len(self.current_samples) != self.minibatch_size:
            raise ValueError("clipped batch must contain exactly B subgraphs")
        before = [p.detach().clone() for p in self.parameters()]
        sum_norm = joint_norm(self.accumulated())
        result = super().step(closure)
        mean_norm = joint_norm([p.grad for p in self.parameters() if p.grad is not None])
        update_ok = True
        index = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    expected = before[index].add(parameter.grad, alpha=-group["lr"])
                    update_ok &= torch.allclose(parameter.detach(), expected, rtol=1e-6, atol=1e-7)
                    index += 1
        bound_ok = (all(s["bound_ok"] for s in self.current_samples)
                    and mean_norm <= self.l2_norm_clip + CONFIG["clipping_tolerance"]
                    and sum_norm <= self.minibatch_size*self.l2_norm_clip + CONFIG["clipping_tolerance"])
        zero_noise = self.last_noise_norm == 0.0 and self.noise_multiplier == 0.0
        if not update_ok or not bound_ok or not zero_noise or not math.isclose(mean_norm, sum_norm/self.minibatch_size, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("SGD update, averaging, clipping or zero-noise verification failed")
        self.records.append(dict(update=len(self.records)+1, samples=list(self.current_samples),
                                 clip_fraction=self.last_clip_fraction,
                                 mean_pre_norm=statistics.mean(s["pre_norm"] for s in self.current_samples),
                                 max_post_norm=max(s["post_norm"] for s in self.current_samples),
                                 gradient_sum_norm=sum_norm, mean_gradient_norm=mean_norm,
                                 joint_bound_ok=bound_ok, update_verified=bool(update_ok), noise_norm=self.last_noise_norm))
        return result


def audit_initialization(model, graph, dg, denominator):
    rt = u.runtime()
    model.eval()
    with torch.no_grad():
        logits = rt["evaluate"].ranking_scores(model, dg, dg.ndata["feat"]).reshape(-1)
        h1 = torch.relu(model.grat.grat1(dg, dg.ndata["feat"]))
        h2 = torch.relu(model.grat.grat2(dg, h1))
        selected = rt["evaluate"].select_topk_nodes(logits, 50).tolist()
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite initial logits")
    covered = set(selected)
    for node in selected:
        covered.update(graph.successors(node))
    return dict(initial_state_sha256=u.state_hash(model), unique_logits=int(torch.unique(logits).numel()),
                all_logits_equal=bool((logits == logits[0]).all()), logit_min=float(logits.min()),
                logit_max=float(logits.max()), logit_std=float(logits.std(unbiased=False)),
                max_tie_count=int((logits == logits.max()).sum()), test_nodes=graph.vcount(),
                selected_zero_outdegree_count=sum(graph.outdegree(node)==0 for node in selected),
                boundary_tie_count=int((logits == logits[selected[-1]]).sum()),
                selected_local_ids=selected, coverage_ratio_percent=100*len(covered)/denominator,
                hidden1_zero_fraction=float((h1 == 0).float().mean()),
                hidden2_zero_fraction=float((h2 == 0).float().mean()))


def read_baseline(previous):
    manifest = json.loads((previous/"manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["records"] != 9:
        raise ValueError("complete nine-cell unclipped baseline is required")
    for key, expected in u.CONFIG.items():
        if manifest["configuration"].get(key) != expected:
            raise ValueError(f"unclipped baseline configuration mismatch: {key}")
    for path, digest in {**manifest["input_sha256"], **manifest["source_sha256"]}.items():
        if u.file_hash(path) != digest:
            raise ValueError(f"unclipped baseline provenance mismatch: {path}")
    return {(int(row["seed"]), row["arm"]): row for row in u.read_csv(previous/"results.csv")}


def train_case(case, arm, seed, previous, baseline, internal, test_graph, test_dgl, denominator, test_hash):
    rt = u.runtime()
    containers = u.prepare_container(case)
    old_diagnostics = json.loads((previous/"internal"/f"seed{seed}_{arm}"/"diagnostics.json").read_text())
    batches = old_diagnostics["batch_indices"]
    if len(batches) != 100 or any(len(batch) != len(set(batch)) or len(batch) != 16 or any(i < 0 or i >= len(containers) for i in batch) for batch in batches):
        raise ValueError("saved paired batch sequence is invalid")
    model = u.new_model(seed)
    initial = audit_initialization(model, test_graph, test_dgl, denominator)
    if initial["initial_state_sha256"] != baseline["initial_state_sha256"]:
        raise ValueError("clipped and unclipped initial weights differ")
    shapes = {name:list(value.shape) for name,value in model.state_dict().items()}
    if shapes != json.loads(baseline["parameter_shapes"]):
        raise ValueError("clipped and unclipped parameter shapes differ")
    loss_fn = rt["Loss"](0.5)
    optimizer = ObservedPrivateSGD(model.parameters(), l2_norm_clip=3.0, minibatch_size=16, lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=1.0)
    def container_loss():
        model.eval()
        with torch.no_grad():
            return statistics.mean(float(loss_fn(model(dg,dg.ndata["feat"]).squeeze(1),adj,50)) for dg,_,adj in containers)
    before_loss = container_loss()
    losses = []
    started = time.perf_counter()
    for step,indices in enumerate(batches):
        batch = [containers[i] for i in indices]
        values = rt["train"].train_single(model,optimizer,scheduler,loss_fn,
            [b[2] for b in batch],[b[1] for b in batch],[b[0] for b in batch],
            [None]*16,[None]*16,[None]*16,50,step,clip_gradients=True)
        if not all(math.isfinite(v) for v in values) or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("non-finite clipped training")
        losses.append(dict(update=step+1,mean_batch_loss=statistics.mean(values)))
    elapsed = time.perf_counter()-started
    after_loss = container_loss()
    if not all(math.isfinite(x) for x in (before_loss,after_loss)):
        raise ValueError("non-finite container loss")
    evaluation = u.evaluate(model,test_graph,test_dgl,denominator,seed,arm)
    case_dir = internal/f"seed{seed}_{arm}"
    case_dir.mkdir(mode=0o700)
    torch.save(model.state_dict(),case_dir/"model.pt")
    u.write_csv(case_dir/"losses.csv",losses)
    u.write_json(case_dir/"gradients.json",optimizer.records)
    u.write_json(case_dir/"diagnostics.json",dict(initial_container_loss=before_loss,final_container_loss=after_loss,
                 batch_indices=batches,initialization=initial,final_state_sha256=u.state_hash(model),configuration=CONFIG))
    for path in case_dir.iterdir():
        path.chmod(0o600)
    records = optimizer.records
    gradient_summary = dict(seed=seed,arm=arm,clip_fraction=statistics.mean(r["clip_fraction"] for r in records),
        max_post_norm=max(r["max_post_norm"] for r in records),
        mean_gradient_norm=statistics.mean(r["mean_gradient_norm"] for r in records),
        mean_pre_norm=statistics.mean(r["mean_pre_norm"] for r in records),
        container_loss_decreased=after_loss<before_loss)
    source = case["source"]
    row = dict(seed=seed,arm=arm,updates=len(records),model_seed=seed,batch_seed=int(baseline["batch_seed"]),
        initial_state_sha256=initial["initial_state_sha256"],parameter_shapes=shapes,
        parameter_count=sum(p.numel() for p in model.parameters()),test_graph_sha256=test_hash,
        saved_subgraphs_verified=True,**evaluation,
        initial_public_coverage_ratio_percent=initial["coverage_ratio_percent"],
        unclipped_coverage_ratio_percent=float(baseline["coverage_ratio_percent"]),
        delta_vs_unclipped_pp=evaluation["coverage_ratio_percent"]-float(baseline["coverage_ratio_percent"]),
        subgraph_container_size=len(containers),scs_subgraphs=int(source["scs_successes"]),
        bes_subgraphs=int(source["bes_successes"]),training_seconds=elapsed,
        node_retention=float(source["node_retention"]),edge_retention=float(source["edge_retention"]),
        largest_wcc_fraction=float(source["largest_wcc_fraction"]),mean_degree=float(source["mean_degree"]),
        clipping_microbatches=sum(len(r["samples"]) for r in records),
        clipping_bounds_verified=all(r["joint_bound_ok"] for r in records),
        updates_verified=all(r["update_verified"] for r in records),
        zero_noise_verified=all(r["noise_norm"]==0 for r in records),
        saved_batches_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest())
    return row,gradient_summary,dict(seed=seed,**initial)


def paired_rows(rows, baseline):
    pairs = u.paired_differences(rows)
    for pair in pairs:
        seed,arm = pair["seed"],pair["arm"]
        old_sampling_delta = float(baseline[(seed,arm)]["coverage_ratio_percent"])-float(baseline[(seed,"original")]["coverage_ratio_percent"])
        pair["unclipped_sampling_delta_pp"] = old_sampling_delta
        pair["sampling_clipping_interaction_pp"] = pair["delta_vs_original_pp"]-old_sampling_delta
    return pairs


def validate_clipping_smoke(rows):
    u.validate_smoke(rows)
    for row in rows:
        if row["clipping_microbatches"] != 1600 or not all(row[key] for key in
            ("clipping_bounds_verified","updates_verified","zero_noise_verified")):
            raise ValueError("joint clipping smoke checks failed")


def write_report(output,rows,baseline,initializations,pairs):
    summaries = u.summarize(rows)
    for s in summaries:
        changes=[r["delta_vs_unclipped_pp"] for r in rows if r["arm"]==s["arm"]]
        s["delta_vs_unclipped_pp_mean"]=statistics.mean(changes)
        s["delta_vs_unclipped_pp_std"]=statistics.stdev(changes) if len(changes)>1 else None
    u.write_csv(output/"summary.csv",summaries)
    u.write_csv(output/"paired_differences.csv",pairs)
    u.write_csv(output/"initialization.csv",initializations)
    lines=["# Email ε₀=2：C=3逐子图联合裁剪、零噪声", "",
           "本地受限的经验诊断，不具有DP/PDP保证。未运行带噪声训练。", "",
           "原PrivateSGD.microbatch_step与train_single(clip_gradients=True)直接复用；对所有可训练参数的联合梯度裁剪，C=3，B=16，T=100，λ=.5，lr=.005，momentum=0，实际高斯标准差=0。模型、32维全一特征、SCS/BES和IC p=1单步top-50评估保持不变。", "",
           "每组直接复用上一轮节点/子图及100个batch的顺序；同种子模型初始化逐项相同。seed=0三组通过裁剪/更新/评估检查后才执行seed=1、2。", "",
           "## 逐种子效用与裁剪效应", "",
           "| seed | 方法 | Spread | Coverage % | 无裁剪Coverage % | 裁剪−无裁剪 Δpp | 训练秒 |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['influence_spread']} | {r['coverage_ratio_percent']:.3f} | {r['unclipped_coverage_ratio_percent']:.3f} | {r['delta_vs_unclipped_pp']:+.3f} | {r['training_seconds']:.3f} |")
    lines += ["", "## 均值与样本标准差", "", "| 方法 | Spread | Coverage % | 裁剪−无裁剪 Δpp |", "|---|---:|---:|---:|"]
    def fmt(s,k):
        sd=s[k+"_std"]
        return f"{s[k+'_mean']:.3f}"+(f" ± {sd:.3f}" if sd is not None else "")
    for s in summaries:
        lines.append(f"| {s['arm']} | {fmt(s,'influence_spread')} | {fmt(s,'coverage_ratio_percent')} | {fmt(s,'delta_vs_unclipped_pp')} |")
    lines += ["", "## 采样与裁剪的配对交互", "",
              "交互Δ=(采样−Original)有裁剪 − (采样−Original)无裁剪；单位为百分点，不是显著性检验。", "",
              "| seed | 方法 | 裁剪后采样Δpp | 裁剪前采样Δpp | 交互Δpp |", "|---:|---|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['arm']} | {p['delta_vs_original_pp']:+.3f} | {p['unclipped_sampling_delta_pp']:+.3f} | {p['sampling_clipping_interaction_pp']:+.3f} |")
    lines += ["", "## 初始化并列检查", "",
              "| seed | unique logits | 最高分并列数 | 所选零出度节点数 | h1零比例 | h2零比例 | 初始Coverage % |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in initializations:
        lines.append(f"| {r['seed']} | {r['unique_logits']} | {r['max_tie_count']} | {r['selected_zero_outdegree_count']} | {r['hidden1_zero_fraction']:.4f} | {r['hidden2_zero_fraction']:.4f} | {r['coverage_ratio_percent']:.3f} |")
    lines += ["", "最高分并列组若超过50个，原有stable排序按节点ID打破并列；相同Coverage不意味着相同随机权重或全部logits相同。初始化哈希及所选ID见initialization.csv。零出度节点在反向消息图中没有入边，最后一层的零聚合再经同一个bias产生相同logit；是否成为最高分取决于初始化。", "",
              "## 原始诊断与限制", "",
              "internal/gradients_summary.csv及每组gradients.json记录1600个子图的裁剪前/后范数、实际累加贡献、每步裁剪比例、平均梯度范数和更新验证。仅内部保存，不作为公开DP/PDP输出。", "",
              "每个梯度项的联合范数及实际浮点累加贡献检查≤3+1e-5；每步平均梯度范数同样检查≤3+1e-5，噪声范数严格为0。此为实现检查，不是隐私证明。", "",
              "仅三个配对种子，不能证明效用等价或稳定因果交互。CPU带诊断耗时不与上一轮裸训练耗时作效率优劣比较。下一阶段只提交noise_design.md，等待决定，不自动训练。", ""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")


def run_experiment(structure_root,previous,prepared_root,output):
    structure_root,previous,prepared_root,output=map(Path,(structure_root,previous,prepared_root,output))
    internal=u.create_output(output)
    snapshot=u.protected_snapshot(structure_root)
    snapshot.update({str(p.resolve()):u.file_hash(p) for p in previous.rglob("*") if p.is_file()})
    for p in (ROOT/"experiments/node_sampling_utility.py",ROOT/"tests/test_node_sampling_utility.py"):
        snapshot[str(p)]=u.file_hash(p)
    u.write_json(internal/"protected_before.json",snapshot)
    baseline=read_baseline(previous)
    rt=u.runtime()
    import numpy as np
    import igraph
    environment=dict(python=sys.version,torch=torch.__version__,dgl=rt["dgl"].__version__,
                     numpy=np.__version__,igraph=igraph.__version__,threads=torch.get_num_threads())
    old_environment=json.loads((previous/"manifest.json").read_text())["environment"]
    if any(environment[key]!=old_environment[key] for key in environment):
        raise ValueError("paired experiment runtime differs from the unclipped baseline")
    _,test_graph,directed=rt["loader"].load_prepared_graph("email","test",prepared_root)
    test_dgl=rt["graph_util"].get_rev_dgl(test_graph,"ones",32,directed,False)
    test_paths=[prepared_root/"email"/name for name in ("email_test_edges.txt","test_nodes.csv","metadata.json")]
    test_hash=hashlib.sha256("".join(u.file_hash(p) for p in test_paths).encode()).hexdigest()
    test_ids={int(r["original_id"]) for r in u.read_csv(prepared_root/"email/test_nodes.csv")}
    denominator=rt["evaluate"].calculate_celf_coverages(test_graph,1,[50])[50]
    if not math.isfinite(denominator) or denominator<=0 or test_graph.vcount()<50:
        raise ValueError("invalid common test graph or CELF denominator")
    manifest=dict(configuration=CONFIG,started_at=datetime.now(timezone.utc).isoformat(),status="running",
                  baseline_run=str(previous.resolve()),structure_run=str(structure_root.resolve()),
                  privacy_claim=False,noise_training_performed=False,celf_spread=denominator,
                  environment=environment,baseline_manifest_sha256=u.file_hash(previous/"manifest.json"),
                  source_sha256={str(p.relative_to(ROOT)):u.file_hash(p) for p in (
                      Path(__file__),ROOT/"tests/test_node_sampling_clipping.py")},input_sha256={})
    u.write_json(output/"manifest.json",manifest)
    rows,gradients,initializations=[],[],[]
    for seed in (0,1,2):
        for arm in u.ARMS:
            case=u.load_saved_case(structure_root,prepared_root,arm,seed)
            if test_ids.intersection(case["original_ids"]):
                raise ValueError("training/test node overlap")
            if test_hash != baseline[(seed,arm)]["test_graph_sha256"]:
                raise ValueError("test graph differs from the unclipped run")
            manifest["input_sha256"].update(case["input_sha256"])
            row,gradient,initial=train_case(case,arm,seed,previous,baseline[(seed,arm)],internal,
                                            test_graph,test_dgl,denominator,test_hash)
            rows.append(row);gradients.append(gradient)
            if arm=="original":initializations.append(initial)
            u.write_csv(output/"results.csv",rows)
            print(f"Email clip3 zero-noise seed={seed} arm={arm}: updates={row['updates']} spread={row['influence_spread']} coverage={row['coverage_ratio_percent']:.3f}%",flush=True)
        if seed==0:
            validate_clipping_smoke(rows)
            u.verify_snapshot(snapshot)
            u.write_json(output/"smoke_passed.json",dict(passed=True,joint_clipping=True,noise_std=0,
                         microbatches_per_model=1600,exact_sgd_updates_verified=True))
            manifest["status"]="smoke_passed"
            u.write_json(output/"manifest.json",manifest)
            print("SMOKE PASSED: joint clipping, zero noise, exact updates; permitting seeds 1,2.",flush=True)
    u.verify_snapshot(snapshot)
    pairs=paired_rows(rows,baseline)
    write_report(output,rows,baseline,initializations,pairs)
    u.write_csv(internal/"gradients_summary.csv",gradients)
    (internal/"gradients_summary.csv").chmod(0o600)
    for p in output.iterdir():
        if p.is_file():p.chmod(0o600)
    manifest.update(status="completed",completed_at=datetime.now(timezone.utc).isoformat(),records=len(rows),
                    existing_files_unchanged=True,smoke_passed=True)
    u.write_json(output/"manifest.json",manifest)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root",type=Path,default=ROOT/"outputs/structure_sampling/personalized-node-20261030-v2")
    parser.add_argument("--previous",type=Path,default=ROOT/"outputs/utility_sampling/email-eps2-nonoise-20261030-v1")
    parser.add_argument("--prepared-root",type=Path,default=ROOT/"dataset/prepared")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run_experiment(args.structure_root,args.previous,args.prepared_root,args.output)


if __name__=="__main__":
    main()
