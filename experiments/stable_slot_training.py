"""Exactly 12 Email partition-0 fixed-slot, zero-noise utility fits.

Independent entry: original model/loss/samplers/PrivateSGD/accountant unchanged.
Private weights, gradients, node IDs and diagnostics stay under restricted/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import stable_slot_gradients as gd
from experiments import node_sampling_utility as u
from experiments.node_sampling_structure import write_csv
import numpy as np
import torch

ARMS = gd.ARMS
BRANCHES = ("flat", "double")


def configuration():
    return dict(dataset="email", partition_seed=0, model_seeds=[0, 1, 2], arms=list(ARMS),
                branches=list(BRANCHES), model="GRAT3", variant="legacy_released",
                initialization="released_default", feature_protocol="ones32", C=3., B=16, T=100,
                learning_rate=.005, momentum=0., loss_lambda=.5, gaussian_std=0.,
                slots=4, quota_per_slot=4, updates_per_step=1, final_batch_clipping=False,
                batch_policy="slot-isolated continuation of prior diagnostic RNG, shared across branches and model seeds",
                shortage_policy="sample min(m,4), then pad None to four; never top up from another slot",
                nonfinite_policy="replace entire position gradient with zero, record; nonfinite parameters abort",
                test_protocol=dict(dataset="email", split="test", ic_p=1, steps=1, k=50, celf_spread=432),
                privacy_claim=False, certified=False, device="cpu", threads=1)


def safe_clip(gradients, templates):
    raw = torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                     for g, p in zip(gradients, templates)])
    count = int((~torch.isfinite(raw)).sum())
    if count:
        zero = torch.zeros_like(raw)
        info = dict(pre_norm=None, legacy_pre_norm=None, post_norm=0., coefficient=0.,
                    clip_triggered=False, missing_parameters=sum(g is None for g in gradients),
                    missing_coordinates=sum(p.numel() for p, g in zip(templates, gradients) if g is None),
                    zero_parameters=0, zero_coordinates=int((raw == 0).sum()), zero_gradient=False,
                    nonfinite_coordinates=count, coordinate_count=raw.numel(), bound_ok=True,
                    zero_substituted=True, reason="nonfinite_gradient")
        return zero, info, raw.clone()
    clipped, info = gd.clip_joint(gradients, templates)
    info.update(zero_substituted=False, reason="")
    return clipped, info, raw.clone()


def make_schedule(sizes, iterations=100):
    if len(sizes) != 4 or any(n < 0 for n in sizes):
        raise ValueError("four fixed nonnegative container sizes required")
    schedule = [[None] * 4 for _ in range(iterations)]
    streams = []
    for j, size in enumerate(sizes):
        seed = int.from_bytes(hashlib.sha256(f"email-slot-gradient-batch-v1|0|{j}".encode()).digest()[:8], "big")
        rng = random.Random(seed)
        before = rng.getstate()
        for t in range(iterations):
            schedule[t][j] = rng.sample(range(size), min(size, 4)) + [None] * max(0, 4 - size)
        streams.append(dict(slot=j, seed=seed, initial_state=before, final_state=rng.getstate(),
                            index_trace_sha256=gd.slots.digest([s[j] for s in schedule])))
    return schedule, streams


@contextmanager
def training_guards():
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("use only explicit zero-noise SGD update")))
        for name in ("randn", "randn_like", "normal"):
            stack.enter_context(patch.object(torch, name, side_effect=AssertionError("Gaussian noise forbidden")))
        stack.enter_context(patch.object(gd.slots.sampling, "sample_privim_star", side_effect=AssertionError("resampling forbidden")))
        yield


def batch_gradients(model, batch, loss_builder):
    templates = [p for p in model.parameters() if p.requires_grad]
    raw, clipped, records = [], [], []
    calls = 0
    if len(batch) != 4 or any(len(slot) != 4 for slot in batch):
        raise ValueError("fixed 4x4 positions required")
    for j in range(4):
        for i in range(4):
            model.zero_grad(set_to_none=True)
            if any(p.grad is not None for p in templates):
                raise AssertionError("gradient reset failed")
            item = batch[j][i]
            loss_value, output_finite = None, True
            if item is not None:
                loss = loss_builder(model, item)
                if isinstance(loss, tuple):
                    loss, output_finite = loss
                if output_finite and bool(torch.isfinite(loss)):
                    loss_value = float(loss.detach())
                    loss.backward()
                    calls += 1
                    a, info, g = safe_clip([p.grad for p in templates], templates)
                else:
                    a, info, g = safe_clip([torch.zeros_like(p) for p in templates], templates)
                    info.update(zero_substituted=True, reason="nonfinite_output_or_loss")
            else:
                a, info, g = safe_clip([torch.zeros_like(p) for p in templates], templates)
                info.update(zero_substituted=True, reason="missing_candidate")
            raw.append(g); clipped.append(a)
            records.append(dict(slot=j, position=i, index=None if item is None else item.get("index"),
                                stage="padding" if item is None else item.get("stage", "synthetic"),
                                loss=loss_value, output_finite=output_finite, **info))
    model.zero_grad(set_to_none=True)
    d = sum(p.numel() for p in templates)
    a = torch.stack(clipped).reshape(4, 4, d)
    return dict(raw=torch.stack(raw).reshape(4, 4, d), a=a, samples=records,
                result=gd.aggregate(a, templates), backward_calls=calls)


def apply_update(parameters, signal):
    parameters = list(parameters)
    if signal.numel() != sum(p.numel() for p in parameters) or not bool(torch.isfinite(signal).all()):
        raise ValueError("finite joint update with exact layout required")
    before = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
    with torch.no_grad():
        for p, block in zip(parameters, signal.split([p.numel() for p in parameters])):
            p.add_(block.reshape(p.shape), alpha=-.005 / 16)
    after = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
    expected = before.double() - (.005 / 16) * signal.double()
    if not bool(torch.isfinite(after).all()) or not torch.allclose(after.double(), expected, atol=2e-7, rtol=2e-6):
        raise AssertionError("update differs from fixed-denominator SGD or became nonfinite")
    return dict(updates=1, denominator=16, coordinates=signal.numel(), changed=not torch.equal(before, after),
                update_norm=gd.norm(after - before), formula_max_error=float((after.double() - expected).abs().max()),
                before=before, after=after)


def synthetic_preflight():
    p = torch.nn.Parameter(torch.zeros(1000)); q = torch.nn.Parameter(torch.zeros(1340))
    raw = torch.zeros(2340); raw[0] = 3.; raw[1000] = 4.
    a, info, _ = safe_clip([raw[:1000], raw[1000:]], [p, q])
    expected_a = raw.double() * min(1., 3. / (5. + 1e-6))
    if not torch.allclose(a.double(), expected_a, atol=1e-6):
        raise AssertionError("joint clip preflight failed")
    batch = torch.zeros(4, 4, 2340)
    for j in range(4):
        batch[j, :, j] = 3.
    result = gd.aggregate(batch, [p, q])
    expected_flat = batch.double().sum((0, 1))
    expected_double = expected_flat * (3. / (12. + 1e-6))
    if not torch.allclose(result["flat"].double(), expected_flat, atol=1e-6) or not torch.allclose(result["slot"].double(), expected_double, atol=1e-6):
        raise AssertionError("aggregation preflight failed")
    if gd.norm(result["slot"]) <= 3. or any(gd.norm(v) > 3.00001 for v in result["v"]):
        raise AssertionError("wrong clipping level")
    audit = apply_update([p, q], result["slot"])
    return dict(passed=True, coordinates=2340, positions=[4] * 4, updates_per_step=audit["updates"],
                denominator=16, double_batch_norm=gd.norm(result["slot"]), gaussian_draws=0,
                update_formula_max_error=audit["formula_max_error"])


def load_containers(path):
    saved = json.loads(Path(path).read_text())
    rt = u.runtime()
    if len(saved["slots"]) != 4:
        raise ValueError("four stable slots required")
    containers, members = [], []
    for j, packet in enumerate(saved["slots"]):
        source, cache = packet["input"], packet["cache"]
        graph = gd.runtime()["igraph"].Graph(n=len(source["original_ids"]), edges=source["edges"], directed=source["directed"])
        graph.vs["original_id"] = source["original_ids"]
        graph.vs["prepared_local_id"] = source["prepared_ids"]
        if gd.slots.input_payload(graph) != source or not graph.is_directed():
            raise ValueError("slot graph/features/direction/mapping changed")
        frequency = Counter(v for nodes in cache["node_sets"] for v in nodes)
        if cache["frequencies"] != [frequency[i] for i in range(graph.vcount())]:
            raise ValueError("saved frequency state changed")
        prepared = u.prepare_container(dict(graph=graph, node_sets=cache["node_sets"]))
        slot = []
        for index, (message, coverage, adjacency) in enumerate(prepared):
            stage = "SCS" if index < cache["stage1_count"] else "BES"
            if message.num_nodes() != (40 if stage == "SCS" else 20):
                raise ValueError("cached subgraph size changed")
            slot.append(dict(index=index, stage=stage, graph=message, adjacency=adjacency))
        containers.append(slot)
        members.append(dict(slot=j, original_ids=source["original_ids"], node_sets=cache["node_sets"]))
    return containers, members


def evaluate(model, context, seed, arm):
    rt = u.runtime()
    result = u.evaluate(model, context["test_graph"], context["test_dgl"], 432, seed, arm)
    with torch.no_grad():
        scores = rt["evaluate"].ranking_scores(model, context["test_dgl"], context["test_dgl"].ndata["feat"]).reshape(-1)
        selected = rt["evaluate"].select_topk_nodes(scores, 50).tolist()
    if len(set(selected)) != 50:
        raise AssertionError("invalid top-50")
    return result, selected


def first_step_check(packet, seed, arm, diagnostic_root, initial_hash):
    path = Path(diagnostic_root) / "restricted" / f"partition0_model{seed}_{arm}" / "gradients.npz"
    with np.load(path) as previous:
        comparisons = dict(raw=np.array_equal(packet["raw"].numpy(), previous["raw"]),
                           clipped=np.array_equal(packet["a"].numpy(), previous["clipped_subgraphs"]),
                           flat=np.array_equal(packet["result"]["flat"].numpy(), previous["S_flat"]),
                           slot=np.array_equal(packet["result"]["slot"].numpy(), previous["S_slot"]))
    with (Path(diagnostic_root) / "restricted/conditions.csv").open() as f:
        match = next(r for r in csv.DictReader(f) if r["partition_seed"] == "0" and int(r["model_seed"]) == seed and r["arm"] == arm)
    comparisons["initial_hash"] = initial_hash == match["initial_state_sha256"]
    if not all(comparisons.values()):
        raise AssertionError(f"first step does not match historical gradient diagnostic: {comparisons}")
    return comparisons


def train_case(case, context):
    seed, arm, branch = case
    directory = context["restricted"] / f"seed{seed}_{arm}_{branch}"
    directory.mkdir(mode=0o700)
    rt = gd.runtime()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = rt["GRAT3"](32, 32, 32, variant="legacy_released", initialization="released_default")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 2340:
        raise AssertionError("GRAT layout changed")
    initial_hash = gd.state_hash(model)
    np.savez_compressed(directory / "initial.npz", **{n: v.detach().numpy() for n, v in model.state_dict().items()})
    initial_eval, initial_top = evaluate(model, context, seed, arm)
    loss_fn = rt["Loss"](.5)
    def loss_builder(net, item):
        out = net(item["graph"], item["graph"].ndata["feat"]).squeeze(1)
        return loss_fn(out, item["adjacency"], 50), bool(torch.isfinite(out).all())
    records, sample_records, slot_records, frames = [], [], [], []
    first = None
    started = time.perf_counter()
    for step, indices in enumerate(context["schedules"][arm], 1):
        model.train()
        batch = [[None if index is None else context["containers"][arm][j][index] for index in indices[j]] for j in range(4)]
        packet = batch_gradients(model, batch, loss_builder)
        result = packet["result"]
        if step == 1:
            first = first_step_check(packet, seed, arm, context["diagnostic_root"], initial_hash)
        chosen = result["flat"] if branch == "flat" else result["slot"]
        audit = apply_update(parameters, chosen)
        losses = [r["loss"] for r in packet["samples"] if r["loss"] is not None]
        record = dict(step=step, phase="early" if step <= 10 else "middle" if 46 <= step <= 55 else "late" if step >= 91 else "other",
                      **result["metrics"], loss=statistics.mean(losses) if losses else None,
                      chosen_norm=gd.norm(chosen), update_norm=audit["update_norm"], update_formula_error=audit["formula_max_error"],
                      updates=audit["updates"], parameters_changed=audit["changed"], backward_calls=packet["backward_calls"],
                      zero_positions=sum(r["zero_substituted"] for r in packet["samples"]),
                      nonfinite_gradient_coordinates=sum(r["nonfinite_coordinates"] for r in packet["samples"]),
                      finite_outputs=all(r["output_finite"] for r in packet["samples"]),
                      clipping_bounds_ok=all(r["bound_ok"] for r in packet["samples"] + result["slot_info"]))
        records.append(record)
        sample_records.extend(dict(step=step, **r) for r in packet["samples"])
        slot_records.extend(dict(step=step, slot=j, **r) for j, r in enumerate(result["slot_info"]))
        frames.append(dict(raw=packet["raw"].numpy(), u=result["u"].numpy(), v=result["v"].numpy(),
                           flat=result["flat"].numpy(), slot=result["slot"].numpy(),
                           before=audit["before"].numpy(), after=audit["after"].numpy()))
    seconds = time.perf_counter() - started
    final_eval, final_top = evaluate(model, context, seed, arm)
    final_hash = gd.state_hash(model)
    if len(records) != 100 or sum(r["updates"] for r in records) != 100 or final_hash == initial_hash:
        raise AssertionError("incomplete training or no parameter change")
    np.savez_compressed(directory / "audit_vectors.npz", **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
    torch.save(model.state_dict(), directory / "final_model.pt")
    gd.write_json(directory / "top50.json", dict(initial=initial_top, final=final_top))
    gd.write_json(directory / "first_step.json", first)
    for name, data in (("steps", records), ("subgraphs", sample_records), ("slots", slot_records)):
        write_csv(directory / f"{name}.csv", data)
    row = dict(seed=seed, arm=arm, branch=branch, influence_spread=final_eval["influence_spread"],
               coverage_ratio_percent=final_eval["coverage_ratio_percent"], initial_coverage=initial_eval["coverage_ratio_percent"],
               updates=100, backward_calls=sum(r["backward_calls"] for r in records), training_seconds=seconds,
               initial_sha256=initial_hash, final_sha256=final_hash, parameters_changed=True,
               first_step_exact=True, batch_sha256=gd.slots.digest(context["schedules"][arm]),
               finite_outputs=all(r["finite_outputs"] for r in records) and final_eval["finite_outputs"],
               nonfinite_gradient_coordinates=sum(r["nonfinite_gradient_coordinates"] for r in records),
               zero_positions=sum(r["zero_positions"] for r in records),
               slot_clip_fraction=statistics.mean(r["slot_clip_fraction"] for r in records),
               all_updates_verified=True, no_noise=True)
    gd.write_json(directory / "result.json", row)
    return dict(row=row, selected=final_top, steps=records)


def case_order():
    return [(seed, arm, branch) for seed in (0, 1, 2) for arm in ARMS for branch in BRANCHES]


def dispatch(fit, smoke_gate):
    result = []
    for case in case_order():
        result.append(fit(case))
        if len(result) == 4 and not smoke_gate(result):
            raise ValueError("four-case smoke gate failed; remaining fits forbidden")
    return result


def smoke_gate(fits, path):
    rows = [f["row"] for f in fits]
    passed = len({r["initial_sha256"] for r in rows}) == 1 and all(
        r["updates"] == 100 and r["finite_outputs"] and r["first_step_exact"] and r["parameters_changed"] and
        r["nonfinite_gradient_coordinates"] == 0 for r in rows)
    for arm in ARMS:
        passed &= len({r["batch_sha256"] for r in rows if r["arm"] == arm}) == 1
    gd.write_json(path, dict(passed=bool(passed), fits=4, updates=400, rows=rows))
    print(f"four-case smoke gate: {bool(passed)}", flush=True)
    return bool(passed)


def save_reports(restricted, fits, background):
    rows, phases, pairs, graph_pairs = [], [], [], []
    metrics = ("flat_norm", "slot_norm", "slot_flat_norm_ratio", "flat_slot_cosine", "slot_clip_fraction", "loss", "update_norm")
    for fit in fits:
        row = fit["row"]
        flat = next(f for f in fits if f["row"]["seed"] == row["seed"] and f["row"]["arm"] == row["arm"] and f["row"]["branch"] == "flat")
        row["top50_overlap_with_flat"] = len(set(fit["selected"]) & set(flat["selected"])) / 50
        gd.write_json(restricted / f"seed{row['seed']}_{row['arm']}_{row['branch']}" / "result.json", row)
        rows.append(row)
        for phase in ("early", "middle", "late"):
            subset = [r for r in fit["steps"] if r["phase"] == phase]
            summary = dict(seed=row["seed"], arm=row["arm"], branch=row["branch"], phase=phase, steps=len(subset))
            for key in metrics:
                values = [r[key] for r in subset if r[key] is not None]
                summary[key + "_mean"] = statistics.mean(values) if values else None
                summary[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            phases.append(summary)
    for seed in range(3):
        for arm in ARMS:
            a, b = [next(r for r in rows if r["seed"] == seed and r["arm"] == arm and r["branch"] == branch) for branch in BRANCHES]
            if a["initial_sha256"] != b["initial_sha256"] or a["batch_sha256"] != b["batch_sha256"]:
                raise AssertionError("paired initial state or batch mismatch")
            pairs.append(dict(seed=seed, arm=arm, spread_delta=b["influence_spread"] - a["influence_spread"],
                              coverage_delta_pp=b["coverage_ratio_percent"] - a["coverage_ratio_percent"],
                              top50_overlap=b["top50_overlap_with_flat"]))
        for branch in BRANCHES:
            a, b = [next(r for r in rows if r["seed"] == seed and r["arm"] == arm and r["branch"] == branch) for arm in ARMS]
            if a["initial_sha256"] != b["initial_sha256"]:
                raise AssertionError("cross-graph initial states differ")
            graph_pairs.append(dict(seed=seed, branch=branch, spread_delta=b["influence_spread"] - a["influence_spread"],
                                    coverage_delta_pp=b["coverage_ratio_percent"] - a["coverage_ratio_percent"]))
    summaries = []
    for arm in ARMS:
        for branch in BRANCHES:
            group = [r for r in rows if r["arm"] == arm and r["branch"] == branch]
            summary = dict(arm=arm, branch=branch, seeds=3)
            for key in ("influence_spread", "coverage_ratio_percent", "training_seconds", "top50_overlap_with_flat"):
                values = [r[key] for r in group if r[key] is not None]
                summary[key + "_mean"] = statistics.mean(values) if values else None
                summary[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
                summary[key + "_n"] = len(values)
            summaries.append(summary)
    for name, data in (("results", rows), ("summary", summaries), ("paired_double_minus_flat", pairs),
                       ("paired_personalized_minus_original", graph_pairs), ("phase_summary", phases), ("historical_background", background)):
        write_csv(restricted / f"{name}.csv", data)
    lines = ["# Email 固定槽位无噪声训练（本地受限，未认证）", "",
             "固定分区seed=0；模型seed=0/1/2；两图条件×Flat/Double，共12次训练，每次100次全局更新。",
             "没有高斯噪声、DP-SGD优化器或PDP会计。所有权重、梯度、损失、成员和统计均非认证DP输出。", "",
             "## 逐种子效用", "", "|seed|图条件|聚合|Spread|Coverage%|初始Coverage%|与Flat的top50重合|", "|---:|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['seed']}|{r['arm']}|{r['branch']}|{r['influence_spread']}|{r['coverage_ratio_percent']:.4f}|{r['initial_coverage']:.4f}|{r['top50_overlap_with_flat']:.2%}|")
    lines += ["", "## 均值和样本标准差", "", "|图条件|聚合|Spread|Coverage%|", "|---|---|---:|---:|"]
    for r in summaries:
        lines.append(f"|{r['arm']}|{r['branch']}|{r['influence_spread_mean']:.3f} ± {r['influence_spread_std']:.3f}|{r['coverage_ratio_percent_mean']:.4f} ± {r['coverage_ratio_percent_std']:.4f}|")
    lines += ["", "## 配对差值与解释", ""]
    for arm in ARMS:
        group = [r for r in pairs if r["arm"] == arm]
        values = [r["coverage_delta_pp"] for r in group]
        lines.append(f"- {arm}：Double−Flat覆盖差（百分点）{values}；均值{statistics.mean(values):+.4f}，样本标准差{statistics.stdev(values):.4f}。")
    lines += ["", "phase_summary.csv按预先固定的早期1–10、中期46–55、末期91–100步报告梯度范数、方向、裁剪比例和损失。",
              "每条轨迹的S_flat/S_slot诊断来自该轨迹当前同一参数点和同一batch；第二步起不得把两轨迹的梯度当作同点比较。",
              "每槽位独立缓存和Random流；Flat/Double以及三个模型初始化共用每图的完整100步batch。",
              "首步原始梯度、逐子图裁剪向量、S_flat、S_slot与上一轮对应诊断逐项一致。",
              "空槽位/不足配额固定补零、非有限位置梯度整体置零均有合成测试；实际异常数量见results.csv。",
              "historical_background.csv为原PrivIM裁剪零噪声结果，仅作背景，使用不同容器，不能视为本轮Flat的配对对照。",
              "仅一个公开分区、三个初始化，不能作强泛化结论；无噪声效用不能替代认证机制条件、理想高斯实现或PDP证明。", ""]
    (restricted / "report.md").write_text("\n".join(lines))
    return rows


def run(output, source_root, diagnostic_root, prepared_root, background_root):
    output, source_root, diagnostic_root, prepared_root, background_root = map(Path, (output, source_root, diagnostic_root, prepared_root, background_root))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    protected = {}
    for directory in (source_root, diagnostic_root, background_root):
        protected.update({str(p.resolve()): gd.file_hash(p) for p in directory.rglob("*") if p.is_file()})
    dm = json.loads((diagnostic_root / "restricted/manifest.json").read_text())
    for path, expected in dm["inputs"].items():
        if gd.file_hash(Path(path)) != expected:
            raise ValueError(f"historical input/source mismatch: {path}")
        protected[path] = expected
    for path in (Path(__file__), ROOT / "tests/test_stable_slot_training.py", ROOT / "experiments/node_sampling_utility.py"):
        protected[str(path.resolve())] = gd.file_hash(path)
    output.mkdir(parents=True, mode=0o755)
    gd.write_json(output / "public_configuration.json", configuration())
    restricted = output / "restricted"; restricted.mkdir(mode=0o700)
    oldmask = os.umask(0o077)
    try:
        with training_guards():
            preflight = synthetic_preflight()
        gd.write_json(restricted / "synthetic_preflight.json", preflight)
        rt = u.runtime()
        torch.set_num_threads(1)
        _, test_graph, directed = rt["loader"].load_prepared_graph("email", "test", prepared_root)
        test_dgl = rt["graph_util"].get_rev_dgl(test_graph, "ones", 32, directed, False)
        test_paths = [prepared_root / "email" / name for name in ("email_test_edges.txt", "test_nodes.csv", "metadata.json")]
        protected.update({str(p.resolve()): gd.file_hash(p) for p in test_paths})
        test_hash = hashlib.sha256("".join(gd.file_hash(p) for p in test_paths).encode()).hexdigest()
        if rt["evaluate"].calculate_celf_coverages(test_graph, 1, [50])[50] != 432:
            raise ValueError("CELF reference changed")
        background = [r for r in u.read_csv(background_root / "results.csv") if r["arm"] in ("original", "personalized")]
        if any(r["test_graph_sha256"] != test_hash for r in background):
            raise ValueError("independent test graph differs from historical evaluation")
        test_ids = {int(r["original_id"]) for r in u.read_csv(prepared_root / "email/test_nodes.csv")}
        containers, schedules = {}, {}
        for arm in ARMS:
            path = source_root / "restricted" / f"email_{arm}_seed0_slots.json"
            source_manifest = json.loads((source_root / "restricted/manifest.json").read_text())
            if gd.file_hash(path) != source_manifest["artifact_sha256"][path.name]:
                raise ValueError("P2 slot cache hash mismatch")
            containers[arm], members = load_containers(path)
            if test_ids & {node for m in members for node in m["original_ids"]}:
                raise ValueError("training/test stable node IDs overlap")
            schedule, streams = make_schedule([len(x) for x in containers[arm]])
            schedules[arm] = schedule
            gd.write_json(restricted / f"batch_{arm}.json", dict(schedule=schedule, streams=streams, members=members))
        context = dict(test_graph=test_graph, test_dgl=test_dgl, containers=containers, schedules=schedules,
                       restricted=restricted, diagnostic_root=diagnostic_root)
        gd.write_json(restricted / "protected_before.json", protected)
        def fit(case):
            result = train_case(case, context)
            print(f"completed seed={case[0]} arm={case[1]} branch={case[2]}: spread={result['row']['influence_spread']} updates=100", flush=True)
            return result
        with training_guards():
            fits = dispatch(fit, lambda data: smoke_gate(data, restricted / "smoke_validation.json"))
        save_reports(restricted, fits, background)
        if any(gd.file_hash(Path(p)) != expected for p, expected in protected.items()):
            raise AssertionError("historical files or original sources changed")
        manifest = dict(status="completed", completed_at=datetime.now(timezone.utc).isoformat(), fits=12, updates=1200,
                        backward_calls=sum(f["row"]["backward_calls"] for f in fits), noise_draws=0, privacy_claim=False,
                        fixed_partition_seed=0, fixed_denominator=16, protected_sha256=protected,
                        historical_hashes_unchanged=True, test_graph_sha256=test_hash, torch=torch.__version__,
                        dgl=rt["dgl"].__version__, artifacts={str(p.relative_to(restricted)): gd.file_hash(p) for p in restricted.rglob("*") if p.is_file()})
        gd.write_json(restricted / "manifest.json", manifest)
        return fits
    finally:
        os.umask(oldmask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs/structure_sampling/stable-slots-l2h2-20261031-v1")
    parser.add_argument("--diagnostic-root", type=Path, default=ROOT / "outputs/gradient_diagnostics/email-stable-slots-clip3-20261031-v1")
    parser.add_argument("--prepared-root", type=Path, default=ROOT / "dataset/prepared")
    parser.add_argument("--background-root", type=Path, default=ROOT / "outputs/utility_sampling/email-eps2-clip3-zero-20261030-v1")
    args = parser.parse_args()
    run(args.output, args.source_root, args.diagnostic_root, args.prepared_root, args.background_root)


if __name__ == "__main__":
    main()
