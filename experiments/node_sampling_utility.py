"""Email-only, epsilon0=2 no-noise utility diagnosis using saved structure cells.

No node/budget/subgraph resampling, privacy accounting or DP/PDP claim. Run
seed=0 smoke gate before seeds 1,2. All artifacts remain local and restricted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(Path(tempfile.gettempdir()) / "privim-utility-dgl"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "privim-utility-mpl"))

from experiments.node_sampling_structure import file_hash, induce_nodes, load_prepared_train, stream_seed, write_csv

ARMS = ("original", "personalized", "uniform_matched")
CONFIG = dict(dataset="email", epsilon0=2, seeds=[0, 1, 2], model="GRAT3",
              grat_variant="legacy_released", feature_protocol="ones", input_dim=32,
              hidden_dims=[32, 32], batch_size=16, iterations=100, loss_lambda=0.5,
              learning_rate=0.005, momentum=0.0, lr_decay=1.0, clip_gradients=False,
              dp_noise=False, privacy_claim=False, ic_probability=1, diffusion_steps=1,
              seed_size=50, device="cpu", torch_threads=1)


@contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


@lru_cache(maxsize=1)
def runtime():
    # Legacy evaluate_models initializes a log directory at import time. Keep
    # those logs out of the repository and ordinary experiment outputs.
    scratch = Path(tempfile.mkdtemp(prefix="privim-utility-runtime-"))
    scratch.chmod(0o700)
    with working_directory(scratch):
        torch = importlib.import_module("torch")
        dgl = importlib.import_module("dgl")
        train = importlib.import_module("train")
        evaluate = importlib.import_module("evaluate_models")
        graph_util = importlib.import_module("graph_util")
        load_graph = importlib.import_module("load_graph")
        loader = importlib.import_module("experiments.igraph_loader")
        from model.GRAT import GRAT3
        from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
    torch.set_num_threads(1)
    return dict(torch=torch, dgl=dgl, train=train, evaluate=evaluate,
                graph_util=graph_util, load_graph=load_graph, loader=loader,
                GRAT3=GRAT3, Loss=KSetMaxCoverAdjLoss, scratch=scratch)


def create_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.chmod(0o700)
    internal = path / "internal"
    internal.mkdir(mode=0o700)
    return internal


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    Path(path).chmod(0o600)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_saved_case(structure_root, prepared_root, arm, seed):
    """Read saved decisions and node sets. Never call a sampling function."""
    if arm not in ARMS or seed not in (0, 1, 2):
        raise ValueError("only Email seeds 0,1,2 and the three declared arms are allowed")
    structure_root = Path(structure_root)
    manifest_path = structure_root / "manifest.json"
    records_path = structure_root / "records.csv"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("source structure experiment must be complete")
    for relative, digest in manifest["source_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f"structure-source provenance mismatch: {relative}")
    graph, _, inputs = load_prepared_train(prepared_root, "email")
    for path, digest in inputs.items():
        if manifest["inputs"].get(path) != digest:
            raise ValueError(f"prepared input differs from saved structure experiment: {path}")
    records = read_csv(records_path)
    matches = [row for row in records if row["dataset"] == "email" and row["arm"] == arm
               and int(row["seed"]) == seed and (arm == "original" or float(row["epsilon0"]) == 2)]
    if len(matches) != 1:
        raise ValueError("source cell is missing or duplicated")
    source = matches[0]
    source_paths = [manifest_path, records_path, structure_root / "budgets/email.csv"]
    selected = list(range(graph.vcount()))
    if arm != "original":
        decisions_path = structure_root / "decisions" / f"email_eps{source['epsilon0']}_seed{seed}.csv"
        decisions = read_csv(decisions_path)
        if len(decisions) != graph.vcount():
            raise ValueError("saved node decisions do not cover the original training graph")
        by_old = {int(row["prepared_local_id"]): row for row in decisions}
        if set(by_old) != set(range(graph.vcount())):
            raise ValueError("saved prepared IDs are not a bijection")
        key = "personalized_new_id" if arm == "personalized" else "uniform_new_id"
        selected = [old for old in range(graph.vcount()) if by_old[old][key] != ""]
        if [int(by_old[old][key]) for old in selected] != list(range(len(selected))):
            raise ValueError("saved new IDs do not match the induced graph ordering")
        if any(int(by_old[i]["original_id"]) != graph.vs[i]["original_id"] for i in by_old):
            raise ValueError("saved original IDs do not match prepared node IDs")
        source_paths.append(decisions_path)
    induced, _ = induce_nodes(graph, selected)
    payload_path = structure_root / "subgraphs" / f"email_eps{source['epsilon0']}_seed{seed}_{arm}.json"
    payload = json.loads(payload_path.read_text())
    source_paths.append(payload_path)
    node_sets = payload["node_sets"]
    freq = Counter(node for nodes in node_sets for node in nodes)
    if payload["frequencies"] != [freq[i] for i in range(induced.vcount())]:
        raise ValueError("saved frequency vector disagrees with saved subgraphs")
    if max(freq.values(), default=0) > int(source["M"]):
        raise ValueError("saved subgraphs violate the original frequency bound")
    if (induced.vcount(), induced.ecount(), len(node_sets)) != (
        int(source["nodes"]), int(source["edges"]), int(source["subgraphs"])
    ):
        raise ValueError("saved decisions or subgraphs disagree with the structure record")
    if [payload["stage1_count"], payload["stage2_count"]] != [int(source["scs_successes"]), int(source["bes_successes"])]:
        raise ValueError("saved SCS/BES stage counts disagree")
    for i, nodes in enumerate(node_sets):
        size = int(source["n"]) if i < payload["stage1_count"] else max(1, int(source["n"]) // 2)
        if len(nodes) != len(set(nodes)) or len(nodes) != size or any(v < 0 or v >= induced.vcount() for v in nodes):
            raise ValueError("invalid saved subgraph node IDs or sizes")
    if len(node_sets) < 16:
        raise ValueError("saved container cannot satisfy the fixed B=16 protocol")
    inputs.update({str(p.resolve()): file_hash(p) for p in source_paths})
    return dict(graph=induced, node_sets=node_sets, source=source, original_ids=induced.vs["original_id"],
                input_sha256=inputs, saved_subgraphs_verified=True)


def check_message_graph(graph, message_graph, coverage_graph):
    rt = runtime()
    torch = rt["torch"]
    n = graph.vcount()
    expected = Counter(graph.get_edgelist())
    if not graph.is_directed():
        expected.update((v, u) for u, v in graph.get_edgelist())
    src, dst = message_graph.edges()
    message_edges = Counter(zip(src.tolist(), dst.tolist()))
    if message_edges != Counter({(v, u): count for (u, v), count in expected.items()}):
        raise ValueError("message graph is not the required reversed coverage graph")
    if Counter(coverage_graph.get_edgelist()) != expected or coverage_graph.vcount() != n:
        raise ValueError("coverage graph direction or indexing is wrong")
    features = message_graph.ndata["feat"]
    if message_graph.num_nodes() != n or tuple(features.shape) != (n, 32) or not torch.equal(features, torch.ones(n, 32)):
        raise ValueError("node features must be aligned 32-dimensional all-one inputs")


def prepare_message_graph(graph):
    rt = runtime()
    dg = rt["graph_util"].get_rev_dgl(graph, "ones", 32, graph.is_directed(), False)
    cg, dg = rt["load_graph"].connect_node_to_graph(list(range(graph.vcount())), dg)
    check_message_graph(graph, dg, cg)
    adjacency = rt["graph_util"].get_adj_mat(cg)
    return dg, cg, adjacency


def prepare_container(case):
    rt = runtime()
    parent = rt["graph_util"].get_rev_dgl(case["graph"], "ones", 32, case["graph"].is_directed(), False)
    containers = []
    for nodes in case["node_sets"]:
        cg, dg = rt["load_graph"].connect_node_to_graph(nodes, parent)
        if dg.ndata[rt["dgl"].NID].tolist() != nodes:
            raise ValueError("DGL reordered saved subgraph IDs")
        local = {old: new for new, old in enumerate(nodes)}
        edges = [(local[u], local[v]) for u, v in case["graph"].get_edgelist() if u in local and v in local]
        import igraph
        expected = igraph.Graph(n=len(nodes), edges=edges, directed=case["graph"].is_directed())
        check_message_graph(expected, dg, cg)
        adjacency = rt["graph_util"].get_adj_mat(cg)
        wanted = rt["torch"].eye(len(nodes))
        for u, v in cg.get_edgelist():
            wanted[u, v] = 1
        if not rt["torch"].equal(adjacency, wanted):
            raise ValueError("coverage adjacency is not aligned with subgraph features")
        containers.append((dg, cg, adjacency))
    return containers


def new_model(seed):
    rt = runtime()
    rt["torch"].manual_seed(seed)
    return rt["GRAT3"](32, 32, 32, variant="legacy_released", initialization="released_default")


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(model, test_graph, test_dgl, denominator, seed, arm):
    rt = runtime()
    torch = rt["torch"]
    model.eval()
    with torch.no_grad():
        logits = rt["evaluate"].ranking_scores(model, test_dgl, test_dgl.ndata["feat"]).reshape(-1)
        probabilities = model(test_dgl, test_dgl.ndata["feat"]).reshape(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise ValueError("non-finite model output on the common test graph")
        selected = rt["evaluate"].select_topk_nodes(logits, 50)
        records = rt["evaluate"].evaluate_model(
            model, 1, "GRAT3", seed, "email_test_edges.txt", test_graph, test_dgl,
            [50], {50: denominator}, repeat=1,
        )
    record = records[0]
    if not math.isfinite(record["coverage_ratio_percent"]) or not math.isfinite(record["n_covered"]):
        raise ValueError("existing evaluation did not return finite spread and coverage")
    # Independently validate the existing one-step IC p=1 coverage calculation.
    covered = set(selected.tolist())
    for node in selected.tolist():
        covered.update(test_graph.successors(node))
    if len(covered) != record["n_covered"]:
        raise ValueError("influence evaluation disagrees with explicit one-step coverage")
    return dict(influence_spread=int(record["n_covered"]),
                coverage_ratio_percent=float(record["coverage_ratio_percent"]),
                selected_seed_count=len(selected), unique_seed_count=len(set(selected.tolist())),
                finite_outputs=True, test_ranking_logit_std=record["ranking_logit_std"],
                test_boundary_tie_count=record["ranking_boundary_tie_count"])


def train_case(case, arm, seed, internal, test_graph, test_dgl, denominator, test_hash):
    rt = runtime()
    torch = rt["torch"]
    started = time.perf_counter()
    containers = prepare_container(case)
    preparation_seconds = time.perf_counter() - started
    model = new_model(seed)
    initial_hash = state_hash(model)
    shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    initial_evaluation = evaluate(model, test_graph, test_dgl, denominator, seed, arm)
    loss_fn = rt["Loss"](0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1.0)
    update_count = 0
    def count_step(*args):
        nonlocal update_count
        update_count += 1
    step_handle = optimizer.register_step_post_hook(count_step)
    batch_seed = stream_seed(seed, "email", "utility-batch") % (2**63 - 1)
    generator = torch.Generator().manual_seed(batch_seed)
    dataset = [dict(idx=i, edge=None) for i in range(len(containers))]
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=16, generator=generator)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=sampler,
                                        collate_fn=rt["train"].my_collate_fn, generator=generator)
    case_dir = internal / f"seed{seed}_{arm}"
    case_dir.mkdir(mode=0o700)
    def container_loss():
        model.eval()
        with torch.no_grad():
            return statistics.mean(float(loss_fn(model(dg, dg.ndata["feat"]).squeeze(1), adj, 50))
                                   for dg, _, adj in containers)
    initial_loss = container_loss()
    losses = []
    batch_indices = []
    started = time.perf_counter()
    for iteration in range(100):
        indices = next(iter(loader))["idx"]
        batch_indices.append(indices)
        batch = [containers[i] for i in indices]
        step_losses = rt["train"].train_single(
            model, optimizer, scheduler, loss_fn,
            [x[2] for x in batch], [x[1] for x in batch], [x[0] for x in batch],
            [None] * 16, [None] * 16, [None] * 16, 50, iteration,
            clip_gradients=False, average_non_private_gradients=True,
        )
        if not all(math.isfinite(value) for value in step_losses) or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("non-finite training loss or parameters")
        losses.append(dict(update=iteration+1, mean_batch_loss=statistics.mean(step_losses)))
    training_seconds = time.perf_counter() - started
    step_handle.remove()
    final_loss = container_loss()
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise ValueError("non-finite full-container diagnostic loss")
    result = evaluate(model, test_graph, test_dgl, denominator, seed, arm)
    torch.save(model.state_dict(), case_dir / "model.pt")
    (case_dir / "model.pt").chmod(0o600)
    write_csv(case_dir / "losses.csv", losses)
    (case_dir / "losses.csv").chmod(0o600)
    write_json(case_dir / "diagnostics.json", dict(
        initial_container_loss=initial_loss, final_container_loss=final_loss,
        batch_indices=batch_indices, final_state_sha256=state_hash(model),
        parameter_shapes=shapes, configuration=CONFIG,
    ))
    source = case["source"]
    row = dict(seed=seed, arm=arm, updates=update_count, model_seed=seed, batch_seed=batch_seed,
               initial_state_sha256=initial_hash, parameter_shapes=shapes,
               parameter_count=sum(p.numel() for p in model.parameters()), test_graph_sha256=test_hash,
               saved_subgraphs_verified=True, **result,
               initial_public_coverage_ratio_percent=initial_evaluation["coverage_ratio_percent"],
               subgraph_container_size=len(containers), scs_subgraphs=int(source["scs_successes"]),
               bes_subgraphs=int(source["bes_successes"]), training_seconds=training_seconds,
               graph_preparation_seconds=preparation_seconds,
               node_retention=float(source["node_retention"]), edge_retention=float(source["edge_retention"]),
               largest_wcc_fraction=float(source["largest_wcc_fraction"]),
               isolated_fraction=float(source["isolated_fraction"]), mean_degree=float(source["mean_degree"]),
               rwr_success_rate=float(source["rwr_success_rate"]),
               node_seed=source["node_seed"], extraction_seed=source["extraction_seed"])
    return row


def validate_smoke(rows):
    if len(rows) != 3 or {r["arm"] for r in rows} != set(ARMS) or {r["seed"] for r in rows} != {0}:
        raise ValueError("smoke gate requires seed=0 for all three arms")
    for key in ("parameter_shapes", "initial_state_sha256", "test_graph_sha256"):
        if any(row[key] != rows[0][key] for row in rows):
            raise ValueError(f"smoke gate mismatch: {key}")
    for row in rows:
        if (row["updates"] != 100 or not row["finite_outputs"] or not row["saved_subgraphs_verified"]
            or row["selected_seed_count"] != 50 or row["unique_seed_count"] != 50
            or not math.isfinite(row["coverage_ratio_percent"]) or not math.isfinite(row["influence_spread"])):
            raise ValueError("smoke training/evaluation checks failed")


def paired_differences(rows):
    result = []
    for seed in sorted({r["seed"] for r in rows}):
        lookup = {r["arm"]: r["coverage_ratio_percent"] for r in rows if r["seed"] == seed}
        if set(lookup) != set(ARMS):
            raise ValueError("incomplete paired seed")
        for arm in ARMS[1:]:
            result.append(dict(seed=seed, arm=arm, delta_vs_original_pp=lookup[arm]-lookup["original"],
                               personalized_minus_uniform_pp=lookup["personalized"]-lookup["uniform_matched"]))
    return result


def summarize(rows):
    summaries = []
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summary = dict(arm=arm, runs=len(selected))
        for key in ("influence_spread", "coverage_ratio_percent", "training_seconds", "subgraph_container_size",
                    "scs_subgraphs", "bes_subgraphs", "node_retention", "edge_retention", "largest_wcc_fraction",
                    "mean_degree", "initial_public_coverage_ratio_percent"):
            values = [r[key] for r in selected]
            summary[key+"_mean"] = statistics.mean(values)
            summary[key+"_std"] = statistics.stdev(values) if len(values)>1 else None
        summaries.append(summary)
    return summaries


def report(rows, summaries, pairs, denominator):
    def show(row, key):
        mean, sd = row[key+"_mean"], row[key+"_std"]
        return f"{mean:.3f} ± {sd:.3f}" if sd is not None else f"{mean:.3f}"
    lines = ["# Email ε₀=2：无噪声 GRAT 效用诊断", "",
             "仅为经验效用诊断，不具有或宣称 DP/PDP 保证。所有文件仅保存在本地权限受限目录，未上传发布。", "",
             "## 固定协议", "",
             "GRAT3 legacy_released；32维全一输入；B=16，100次更新，λ=0.5，lr=0.005，momentum=0，无噪声、无额外梯度裁剪、无学习率衰减。复用原 train_single 的 batch-mean SGD 分支。",
             "直接读取结构实验 ε₀=2 的已保存节点决策、映射和 SCS/BES 子图，不重新分配预算、不重新抽样。三组相同模型种子，批采样使用独立同种子流；训练图与测试图按原始ID隔离。",
             f"测试图固定为既有 Email test；IC p=1，单步，top-50，CELF spread={denominator}。种子0先完成三组 smoke gate 后，才运行其余种子。", "",
             "## 每种子结果", "",
             "| seed | 方法 | Spread | Coverage % | 初始模型Coverage % | SCS/BES | 训练秒 |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['influence_spread']} | {r['coverage_ratio_percent']:.3f} | {r['initial_public_coverage_ratio_percent']:.3f} | {r['scs_subgraphs']}/{r['bes_subgraphs']} | {r['training_seconds']:.3f} |")
    lines += ["", "## 均值与样本标准差", "", "| 方法 | n | Spread | Coverage % | 训练秒 |", "|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['arm']} | {r['runs']} | {show(r,'influence_spread')} | {show(r,'coverage_ratio_percent')} | {show(r,'training_seconds')} |")
    lines += ["", "## 配对差值", "", "Coverage变化以百分点(pp)表示，不是相对百分比。", "",
              "| seed | 方法 | 相对Original Δpp | Personalized−Uniform Δpp |", "|---:|---|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['arm']} | {p['delta_vs_original_pp']:+.3f} | {p['personalized_minus_uniform_pp']:+.3f} |")
    for arm in ARMS[1:]:
        values=[p["delta_vs_original_pp"] for p in pairs if p["arm"]==arm]
        sd=statistics.stdev(values) if len(values)>1 else None
        text=f"{statistics.mean(values):+.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
        lines += ["", f"- {arm} 相对Original配对变化：{text} pp。"]
    lines += ["", "## 结构与解释边界", "",
              "| 方法 | 节点保留% | 边保留% | 最大WCC% | 平均总度 | 子图数 |", "|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['arm']} | {100*r['node_retention_mean']:.2f} | {100*r['edge_retention_mean']:.2f} | {100*r['largest_wcc_fraction_mean']:.2f} | {r['mean_degree_mean']:.2f} | {show(r,'subgraph_container_size')} |")
    lines += ["", "- **实验观察**：以每种子配对差值为主，同时对照初始公开模型；三组的测试图、初始化和优化参数一致，只有保存的训练图/子图容器不同。",
              "- **可能解释**：边和平均度下降、容器缩小、SCS/BES比例变化都可能影响学习；若某种子三组表现一起偏低，应考虑共有初始化因素。这里不根据3个种子声称因果关系。",
              "- **尚未证明**：3个配对种子不足以证明效用等价或稳定优越；没有隐私认证，也没有带噪声效用结果。",
              "- **下一步**：只在本诊断显示采样后的模型仍能学到有效排序时，考虑 ε₀=2 的小规模带噪声探索；本入口没有带噪声选项，不会自动运行。", "",
              "## 文件与内部诊断", "",
              "results.csv 为每种子结果，summary.csv 为样本均值/标准差，paired_differences.csv 为逐种子差值。",
              "internal/ 中保存模型权重、每步损失、固定容器前后损失和批索引，仅供本地内部诊断；报告不导出真实训练损失。",
              "manifest.json 记录固定配置、环境、来源哈希和 smoke gate；已有源码及结构实验产物在运行前后进行内容校验。", ""]
    return "\n".join(lines)


def protected_snapshot(structure_root):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {ROOT / p for p in tracked if p and (ROOT / p).is_file()}
    files.update(p for p in Path(structure_root).rglob("*") if p.is_file())
    files.update((ROOT / "AGENTS.md", ROOT / "experiments/node_sampling_structure.py",
                  ROOT / "tests/test_node_sampling_structure.py"))
    return {str(p.resolve()): file_hash(p) for p in sorted(files)}


def verify_snapshot(snapshot):
    if any(not Path(path).is_file() or file_hash(path) != digest for path,digest in snapshot.items()):
        raise ValueError("an existing source/result file changed during the experiment")


def run_experiment(structure_root, prepared_root, output, smoke_only=False):
    structure_root, prepared_root, output = map(Path, (structure_root, prepared_root, output))
    internal = create_output(output)
    snapshot = protected_snapshot(structure_root)
    rt = runtime()
    import numpy as np
    import igraph
    manifest = dict(configuration=CONFIG, started_at=datetime.now(timezone.utc).isoformat(),
                    status="running", privacy_claim=False, source_experiment=str(structure_root.resolve()),
                    environment=dict(python=sys.version, torch=rt["torch"].__version__, dgl=rt["dgl"].__version__,
                                     numpy=np.__version__, igraph=igraph.__version__, threads=1),
                    source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in (
                        Path(__file__),ROOT/"tests/test_node_sampling_utility.py")}, input_sha256={})
    write_json(output/"manifest.json",manifest)
    write_json(internal/"protected_before.json",snapshot)
    _, test_graph, directed = rt["loader"].load_prepared_graph("email", "test", prepared_root)
    test_dgl = rt["graph_util"].get_rev_dgl(test_graph, "ones", 32, directed, False)
    test_inputs = [prepared_root/"email"/name for name in ("email_test_edges.txt", "test_nodes.csv", "metadata.json")]
    for p in test_inputs:
        manifest["input_sha256"][str(p.resolve())]=file_hash(p)
    test_hash=hashlib.sha256("".join(file_hash(p) for p in test_inputs).encode()).hexdigest()
    test_ids={int(row["original_id"]) for row in read_csv(prepared_root/"email/test_nodes.csv")}
    denominator=rt["evaluate"].calculate_celf_coverages(test_graph,1,[50])[50]
    if not math.isfinite(denominator) or denominator<=0:
        raise ValueError("invalid CELF denominator")
    rows=[]
    for seed in ([0] if smoke_only else [0,1,2]):
        for arm in ARMS:
            case=load_saved_case(structure_root,prepared_root,arm,seed)
            if test_ids.intersection(case["original_ids"]):
                raise ValueError("training and test original node IDs overlap")
            manifest["input_sha256"].update(case["input_sha256"])
            row=train_case(case,arm,seed,internal,test_graph,test_dgl,denominator,test_hash)
            rows.append(row)
            write_csv(output/"results.csv",rows)
            print(f"Email seed={seed} arm={arm}: updates={row['updates']}, spread={row['influence_spread']}, coverage={row['coverage_ratio_percent']:.3f}%",flush=True)
        if seed==0:
            validate_smoke(rows)
            verify_snapshot(snapshot)
            write_json(output/"smoke_passed.json",dict(passed=True,checks=["100_updates","finite_outputs","same_shapes","same_initial_weights","top50","spread_and_coverage","unchanged_existing_files"]))
            manifest["status"]="smoke_passed"
            write_json(output/"manifest.json",manifest)
            print("SMOKE PASSED: all seed=0 arms; subsequent seeds are now permitted.",flush=True)
    verify_snapshot(snapshot)
    for path,digest in manifest["input_sha256"].items():
        if file_hash(path)!=digest:
            raise ValueError("input changed during experiment")
    summaries=summarize(rows)
    pairs=paired_differences(rows)
    write_csv(output/"summary.csv",summaries)
    write_csv(output/"paired_differences.csv",pairs)
    (output/"report.md").write_text(report(rows,summaries,pairs,denominator),encoding="utf-8")
    for p in output.glob("*"):
        if p.is_file():p.chmod(0o600)
    manifest.update(status="smoke_only_completed" if smoke_only else "completed",completed_at=datetime.now(timezone.utc).isoformat(),
                    records=len(rows),celf_spread=denominator,smoke_passed=True,existing_files_unchanged=True)
    write_json(output/"manifest.json",manifest)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root",type=Path,default=ROOT/"outputs/structure_sampling/personalized-node-20261030-v2")
    parser.add_argument("--prepared-root",type=Path,default=ROOT/"dataset/prepared")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--smoke-only",action="store_true")
    args=parser.parse_args()
    run_experiment(args.structure_root,args.prepared_root,args.output,args.smoke_only)


if __name__=="__main__":
    main()
