"""Email-only fixed-parameter slot-gradient diagnostics. Never train or add noise.

Public configuration is separate from restricted raw gradients/initial weights.
All numerical diagnostics are uncertified private research artifacts, not DP output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "privim-utility-mpl"))
os.environ.setdefault("DGLDEFAULTDIR", str(Path(tempfile.gettempdir()) / "privim-utility-dgl"))
os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np
import torch
from experiments import stable_slot_structure as slots
from experiments.node_sampling_structure import file_hash, write_csv

ARMS = ("original", "personalized_eps2")


def configuration():
    return dict(dataset="email", partition_seeds=list(range(5)), model_seeds=[0, 1, 2],
                arms=list(ARMS), model="GRAT3", variant="legacy_released", initialization="released_default",
                feature_protocol="ones32", loss="KSetMaxCoverAdjLoss", loss_lambda=.5, k_train=50,
                C=3., B=16, slots=4, quota_per_slot=4, parameter_coordinates=2340,
                per_subgraph_clip="original joint block norm; min(1,3/(norm+1e-6))",
                per_slot_clip="same joint block norm and epsilon convention on sum of four clipped gradients",
                batch_rule="Random(seed).sample(range(saved_container_size),4), no replacement",
                batch_seed_rule="sha256('email-slot-gradient-batch-v1|partition_seed|slot')[:8] big endian",
                same_batch_across_model_initializations=True, device="cpu", threads=1,
                parameter_updates=0, noise_draws=0, training_performed=False, privacy_claim=False)


def norm(vector):
    return float(vector.detach().double().norm())


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return float(torch.dot(a.double(), b.double())) / denominator if denominator else None


def clip_joint(gradients, templates, C=3.):
    if len(gradients) != len(templates) or not templates:
        raise ValueError("gradient layout mismatch")
    values, squared = [], 0.
    missing_parameters = missing_coordinates = zero_parameters = zero_coordinates = 0
    for gradient, template in zip(gradients, templates):
        if gradient is None:
            missing_parameters += 1
            missing_coordinates += template.numel()
            values.append(torch.zeros_like(template).reshape(-1))
            continue
        if gradient.shape != template.shape or gradient.dtype != template.dtype:
            raise ValueError("gradient shape/dtype mismatch")
        gradient = gradient.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("non-finite gradient; stop without replacing diagnostic data")
        squared += gradient.norm(2).item() ** 2
        zero_parameters += not bool(torch.count_nonzero(gradient))
        zero_coordinates += int((gradient == 0).sum())
        values.append(gradient.reshape(-1).clone())
    raw = torch.cat(values)
    legacy_norm = squared ** .5
    coefficient = min(C / (legacy_norm + 1e-6), 1.)
    clipped = raw * coefficient
    info = dict(pre_norm=norm(raw), legacy_pre_norm=legacy_norm, post_norm=norm(clipped),
                coefficient=coefficient, clip_triggered=coefficient < 1.,
                missing_parameters=missing_parameters, missing_coordinates=missing_coordinates,
                zero_parameters=zero_parameters, zero_coordinates=zero_coordinates,
                zero_gradient=norm(raw) == 0., nonfinite_coordinates=0,
                coordinate_count=raw.numel(), bound_ok=norm(clipped) <= C + 1e-5)
    if not info["bound_ok"]:
        raise AssertionError("clip norm check failed")
    return clipped, info


def aggregate(a, templates):
    sizes = [p.numel() for p in templates]
    if tuple(a.shape) != (4, 4, sum(sizes)) or not bool(torch.isfinite(a).all()):
        raise ValueError("expected four isolated slots, each with four finite vectors")
    u, v, info = [], [], []
    for j in range(4):
        local = torch.zeros_like(a[j, 0])
        for i in range(4):
            local.add_(a[j, i])
        blocks = [b.reshape(p.shape) for b, p in zip(local.split(sizes), templates)]
        clipped, record = clip_joint(blocks, templates)
        u.append(local.clone())
        v.append(clipped.clone())
        info.append(record)
    u, v = torch.stack(u), torch.stack(v)
    flat, slot = torch.zeros_like(u[0]), torch.zeros_like(v[0])
    for j in range(4):
        flat.add_(u[j])
        slot.add_(v[j])
    nf, ns = norm(flat), norm(slot)
    metrics = dict(flat_norm=nf, slot_norm=ns, slot_flat_norm_ratio=ns / nf if nf else None,
                   flat_slot_cosine=cosine(flat, slot), flat_slot_distance=norm(flat - slot),
                   slot_clip_fraction=sum(r["clip_triggered"] for r in info) / 4,
                   flat_cancellation_ratio=nf / sum(norm(x) for x in u) if sum(norm(x) for x in u) else None,
                   slot_cancellation_ratio=ns / sum(norm(x) for x in v) if sum(norm(x) for x in v) else None,
                   flat_fp64_sum_error=norm(flat.double() - a.double().sum((0, 1))))
    if not torch.allclose(flat.double(), a.double().sum((0, 1)), atol=2e-5, rtol=2e-5):
        raise AssertionError("aggregation differs from independent FP64 reference")
    return dict(u=u, v=v, flat=flat, slot=slot, slot_info=info, metrics=metrics)


def noise_reference():
    # Independent analytic computation; never call or change the old accountant.
    from scipy.optimize import brentq
    orders = np.arange(1, 100, .1)[1:]
    def epsilon(std):
        rho = 100 * orders * 6 ** 2 / (2 * std ** 2)
        return float(np.min(rho - (math.log(1e-5) + np.log(orders)) / (orders - 1)
                            + np.log((orders - 1) / orders)))
    base = brentq(lambda std: epsilon(std) - 2, .01, 1e5)
    return dict(epsilon=2., delta=1e-5, T=100, flat_K=16, slot_K=2,
                flat_s_sum=16 * base, slot_s_sum=2 * base,
                flat_noise_rms=16 * base * math.sqrt(2340),
                slot_noise_rms=2 * base * math.sqrt(2340),
                noise_scale_ratio=8., gaussian_draws=0,
                assumption="conditional RDP; total mechanism and slot locality premises; no subsampling amplification")


def proxy_metrics(flat_norm, slot_norm, reference):
    flat_proxy = flat_norm / reference["flat_s_sum"]
    slot_proxy = slot_norm / reference["slot_s_sum"]
    return dict(flat_signal_over_coordinate_std=flat_proxy,
                slot_signal_over_coordinate_std=slot_proxy,
                flat_signal_over_noise_rms=flat_norm / reference["flat_noise_rms"],
                slot_signal_over_noise_rms=slot_norm / reference["slot_noise_rms"],
                proxy_gain_slot_over_flat=slot_proxy / flat_proxy if flat_proxy else None)


def choose_indices(container_size, partition_seed, slot):
    if container_size < 4:
        raise ValueError("saved slot cannot satisfy fixed quota; no top-up")
    seed = int.from_bytes(hashlib.sha256(
        f"email-slot-gradient-batch-v1|{partition_seed}|{slot}".encode()).digest()[:8], "big")
    rng = random.Random(seed)
    initial = rng.getstate()
    indices = rng.sample(range(container_size), 4)
    return indices, dict(seed=seed, initial_state=initial, final_state=rng.getstate())


@contextmanager
def forward_backward_only():
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizers forbidden")))
        for name in ("randn", "randn_like", "normal"):
            stack.enter_context(patch.object(torch, name, side_effect=AssertionError("Gaussian draws forbidden")))
        stack.enter_context(patch.object(slots.sampling, "sample_privim_star", side_effect=AssertionError("resampling forbidden")))
        yield


@lru_cache(maxsize=1)
def runtime():
    import dgl
    import igraph
    import graph_util
    import load_graph
    from model.GRAT import GRAT3
    from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
    return dict(dgl=dgl, igraph=igraph, graph_util=graph_util, load_graph=load_graph,
                GRAT3=GRAT3, Loss=KSetMaxCoverAdjLoss)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_batch(path, partition_seed):
    rt = runtime()
    saved = json.loads(Path(path).read_text())
    batch, selections = [], []
    if len(saved["slots"]) != 4:
        raise ValueError("four fixed slots required")
    for j, data in enumerate(saved["slots"]):
        source, cache = data["input"], data["cache"]
        graph = rt["igraph"].Graph(n=len(source["original_ids"]), edges=source["edges"], directed=source["directed"])
        graph.vs["original_id"] = source["original_ids"]
        graph.vs["prepared_local_id"] = source["prepared_ids"]
        if slots.input_payload(graph) != source or not graph.is_directed():
            raise ValueError("saved Email slot direction/features/mapping mismatch")
        sets = cache["node_sets"]
        frequencies = Counter(v for nodes in sets for v in nodes)
        if cache["frequencies"] != [frequencies[i] for i in range(graph.vcount())]:
            raise ValueError("saved frequency state mismatch")
        if len(sets) != cache["stage1_count"] + cache["stage2_count"]:
            raise ValueError("saved stage counts mismatch")
        indices, stream = choose_indices(len(sets), partition_seed, j)
        parent = rt["graph_util"].get_rev_dgl(graph, "ones", 32, True, False)
        selected = []
        for position, index in enumerate(indices):
            nodes = sets[index]
            stage = "SCS" if index < cache["stage1_count"] else "BES"
            if len(nodes) != (40 if stage == "SCS" else 20) or len(set(nodes)) != len(nodes):
                raise ValueError("saved subgraph size/membership invalid")
            coverage, message = rt["load_graph"].connect_node_to_graph(nodes, parent)
            if message.ndata[rt["dgl"].NID].tolist() != nodes:
                raise ValueError("DGL changed saved node order")
            local = {old: new for new, old in enumerate(nodes)}
            expected = Counter((local[u], local[v]) for u, v in graph.get_edgelist() if u in local and v in local)
            src, dst = message.edges()
            if Counter(zip(src.tolist(), dst.tolist())) != Counter({(v, u): c for (u, v), c in expected.items()}):
                raise ValueError("message edge direction changed")
            if Counter(coverage.get_edgelist()) != expected or coverage.vcount() != len(nodes):
                raise ValueError("coverage graph mapping changed")
            if not torch.equal(message.ndata["feat"], torch.ones(len(nodes), 32)):
                raise ValueError("node feature mismatch")
            adjacency = rt["graph_util"].get_adj_mat(coverage)
            batch.append(dict(slot=j, position=position, index=index, stage=stage,
                              graph=message, adjacency=adjacency))
            selected.append(dict(index=index, stage=stage, slot_local_nodes=nodes,
                                 original_ids=[source["original_ids"][i] for i in nodes]))
        selections.append(dict(slot=j, stream=stream, chosen=selected, container_size=len(sets)))
    return batch, selections


def diagnose(batch, model_seed, reference):
    rt = runtime()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = rt["GRAT3"](32, 32, 32, variant="legacy_released", initialization="released_default")
    # Legacy GRAT contains no dropout/batchnorm; eval keeps autograd enabled.
    if any(isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for m in model.modules()):
        raise AssertionError("training-mode dependent layer requires review")
    model.eval()
    initial = {name: x.detach().clone() for name, x in model.state_dict().items()}
    initial_hash = state_hash(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    templates = [p for _, p in named]
    if sum(p.numel() for p in templates) != 2340 or len(batch) != 16:
        raise AssertionError("parameter/batch dimensions changed")
    loss_fn = rt["Loss"](.5)
    raw, clipped, samples = [], [], []
    torch_before, python_before = torch.get_rng_state().clone(), random.getstate()
    for item in batch:
        model.zero_grad(set_to_none=True)
        if any(p.grad is not None for p in templates):
            raise AssertionError("stale gradients before forward")
        out = model(item["graph"], item["graph"].ndata["feat"]).squeeze(1)
        loss = loss_fn(out, item["adjacency"], 50)
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(loss)):
            raise ValueError("non-finite model output/loss; diagnostic stopped")
        loss.backward()
        gradients = [p.grad.detach().clone() if p.grad is not None else None for p in templates]
        a, info = clip_joint(gradients, templates)
        raw.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(gradients, templates)]))
        clipped.append(a)
        samples.append(dict(slot=item["slot"], position=item["position"], subgraph_index=item["index"],
                            stage=item["stage"], loss=float(loss.detach()),
                            missing_parameter_names=[name for (name, p), g in zip(named, gradients) if g is None], **info))
    model.zero_grad(set_to_none=True)
    if not all(torch.equal(initial[name], value) for name, value in model.state_dict().items()):
        raise AssertionError("model parameters or buffers changed")
    if not torch.equal(torch_before, torch.get_rng_state()) or python_before != random.getstate():
        raise AssertionError("forward/backward consumed an undeclared random stream")
    raw, a = torch.stack(raw).reshape(4, 4, 2340), torch.stack(clipped).reshape(4, 4, 2340)
    result = aggregate(a, templates)
    metrics = dict(**result["metrics"], **proxy_metrics(norm(result["flat"]), norm(result["slot"]), reference),
                   subgraph_clip_fraction=sum(r["clip_triggered"] for r in samples) / 16,
                   missing_gradient_parameters=sum(r["missing_parameters"] for r in samples),
                   missing_gradient_coordinates=sum(r["missing_coordinates"] for r in samples),
                   zero_gradient_subgraphs=sum(r["zero_gradient"] for r in samples),
                   zero_gradient_coordinates=sum(r["zero_coordinates"] for r in samples),
                   nonfinite_coordinates=sum(r["nonfinite_coordinates"] for r in samples),
                   initial_state_sha256=initial_hash, final_state_sha256=state_hash(model),
                   parameter_coordinates=2340, forward_backward_calls=16, parameter_updates=0,
                   parameters_unchanged=True, undeclared_random_draws=False,
                   gradient_reset_verified=True, all_clip_bounds_ok=all(r["bound_ok"] for r in samples + result["slot_info"]))
    slot_records, pairs = [], []
    for j in range(4):
        selected = [r for r in samples if r["slot"] == j]
        slot_records.append(dict(slot=j, scs_count=sum(r["stage"] == "SCS" for r in selected),
                                 bes_count=sum(r["stage"] == "BES" for r in selected),
                                 cosine_u_with_flat=cosine(result["u"][j], result["flat"]),
                                 cosine_v_with_slot=cosine(result["v"][j], result["slot"]),
                                 **result["slot_info"][j]))
        for k in range(j + 1, 4):
            pairs.append(dict(slot_a=j, slot_b=k, pre_cosine=cosine(result["u"][j], result["u"][k]),
                              post_cosine=cosine(result["v"][j], result["v"][k])))
    stages = []
    for stage in ("SCS", "BES"):
        group = [r for r in samples if r["stage"] == stage]
        indices = [i for i, r in enumerate(samples) if r["stage"] == stage]
        record = dict(stage=stage, count=len(group), clip_fraction=statistics.mean(r["clip_triggered"] for r in group) if group else None,
                      clipped_sum_norm=norm(a.reshape(16, 2340)[indices].sum(0)) if group else 0.)
        for field in ("pre_norm", "post_norm", "coefficient"):
            values = [r[field] for r in group]
            record[field + "_mean"] = statistics.mean(values) if values else None
            record[field + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        stages.append(record)
    arrays = dict(raw=raw.numpy(), clipped_subgraphs=a.numpy(), u=result["u"].numpy(),
                  v=result["v"].numpy(), S_flat=result["flat"].numpy(), S_slot=result["slot"].numpy())
    return dict(metrics=metrics, samples=samples, slots=slot_records, pairs=pairs, stages=stages, arrays=arrays,
                initial={name: value.numpy() for name, value in initial.items()},
                layout=[dict(name=n, shape=list(p.shape), coordinates=p.numel()) for n, p in named])


def condition_order():
    smoke = [(0, 0, arm) for arm in ARMS]
    all_cases = [(p, m, arm) for p in range(5) for m in range(3) for arm in ARMS]
    return smoke + [case for case in all_cases if case not in smoke]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


SUMMARY_KEYS = ("flat_norm", "slot_norm", "slot_flat_norm_ratio", "flat_slot_cosine", "flat_slot_distance",
                "slot_clip_fraction", "subgraph_clip_fraction", "proxy_gain_slot_over_flat",
                "flat_signal_over_coordinate_std", "slot_signal_over_coordinate_std",
                "flat_signal_over_noise_rms", "slot_signal_over_noise_rms")


def grouped_summary(records, group_keys):
    result = []
    for identity in sorted({tuple(r[k] for k in group_keys) for r in records}):
        group = [r for r in records if tuple(r[k] for k in group_keys) == identity]
        row = dict(zip(group_keys, identity), conditions=len(group))
        for key in SUMMARY_KEYS:
            values = [r[key] for r in group if r[key] is not None]
            row[key + "_mean"] = statistics.mean(values) if values else None
            row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(row)
    return result


def report(records, sample_rows, slot_rows, stage_rows, reference):
    f = lambda x: "NA" if x is None else f"{x:.4g}"
    lines = ["# Email 槽位整体裁剪梯度信号诊断（本地受限）", "",
             "**仅初始化参数下的forward/backward。无训练、参数更新、噪声注入或DP/PDP认证。**", "",
             "## 固定选择与可比性", "",
             "仅复用P2保存的Email四槽位输入和容器，不重新生成分区、预算、节点决策或训练子图。",
             "每槽位独立Random流均匀无放回选4个子图，选择种子只依赖分区seed和槽位；三个模型seed共享batch。",
             "同模型seed的所有条件使用同一初始化。原GRAT3 legacy_released及MaxCoverLoss(lambda=.5)保持不变。",
             "逐子图及槽位裁剪使用原联合范数和+1e-6约定；全部2340坐标联合处理，缺失梯度填零。",
             "所有原始梯度、初始化权重、成员和ID位于restricted/，不属于DP保护输出。", "",
             "## 完整30条件", "",
             "|分区seed|模型seed|图条件|S_flat范数|S_slot范数|范数比|余弦|槽位裁剪比例|信噪代理增益|",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"|{r['partition_seed']}|{r['model_seed']}|{r['arm']}|" +
                     "|".join(f(r[k]) for k in ("flat_norm", "slot_norm", "slot_flat_norm_ratio", "flat_slot_cosine", "slot_clip_fraction", "proxy_gain_slot_over_flat")) + "|")
    lines += ["", "## 汇总口径", "",
              "partition_summary.csv：每分区/图条件内三个初始化的均值和样本标准差。",
              "model_summary.csv：每初始化/图条件内五个分区配置的均值和样本标准差。",
              "paired.csv：同分区、同初始化下Personalized−Original差值；不能把30条件当作30个独立图样本。",
              "Personalized的分区seed与前阶段节点决策seed配对，五配置不能分离节点采样和分区的方差。", "",
              "## 裁剪与方向", "",
              f"槽位整体裁剪触发{sum(r['clip_triggered'] for r in slot_rows)}/{len(slot_rows)}；"
              f"逐子图裁剪触发{sum(r['clip_triggered'] for r in sample_rows)}/{len(sample_rows)}。",
              f"非有限坐标{sum(r['nonfinite_coordinates'] for r in sample_rows)}；"
              f"零梯度子图{sum(r['zero_gradient'] for r in sample_rows)}。",
              f"存在S_slot范数大于S_flat的条件数：{sum(r['slot_norm'] > r['flat_norm'] for r in records)}。",
              "每槽位贡献和相互余弦见slots.csv、slot_pairs.csv；真实向量保存在每条件gradients.npz。", "",
              "## SCS / BES（描述性，不作因果归因）", "",
              "|图条件|阶段|子图-初始化观测数|原梯度范数均值|裁剪后范数均值|逐子图裁剪比例|",
              "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for stage in ("SCS", "BES"):
            group = [r for r in sample_rows if r["arm"] == arm and r["stage"] == stage]
            lines.append(f"|{arm}|{stage}|{len(group)}|" + "|".join(
                f(statistics.mean(r[k] for r in group)) if group else "NA"
                for k in ("pre_norm", "post_norm", "clip_triggered")) + "|")
    lines += ["", "每条件阶段均值/样本标准差及阶段合计向量范数见stages.csv。",
              "SCS为40节点、BES为20节点；阶段、子图大小、结构和初始化作用混杂，不能把均值差直接归因为BES算法。", "",
              "## 解析认证噪声参照（没有抽样或注入）", "",
              f"epsilon=2, delta=1e-5, T=100：K=16的s_sum={reference['flat_s_sum']:.9f}；K=2的s_sum={reference['slot_s_sum']:.9f}。",
              "每坐标代理=聚合范数/s_sum；向量RMS代理=聚合范数/(sqrt(2340)*s_sum)。",
              "信噪代理增益=8*||S_slot||/||S_flat||；零分母记NA。所有实际值见conditions.csv。",
              "这是条件RDP理论机制下的数值参照，非当前代码认证，不使用历史H2效用外推。", "",
              "## 限制", "",
              "仅每个条件一个固定batch、仅模型初始化点；不能证明学习曲线、训练稳定性、IM效用或严格隐私。",
              "裁剪改变槽位权重，未必只缩放总向量；范数比和方向必须一起判断。",
              "更高信噪代理不表示绝对噪声水平可用；没有用本结果调整C、配额、分区或学习率。", ""]
    return "\n".join(lines)


def run(output, source_root):
    output, source_root = Path(output), Path(source_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_manifest_path = source_root / "restricted/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["status"] != "completed" or source_manifest["configs"] != 30:
        raise ValueError("P2 screen must be complete")
    inputs = {str(source_manifest_path): file_hash(source_manifest_path)}
    for path, expected in source_manifest["protected_sha256_before"].items():
        if path.endswith(".py") and ("/experiments/" in path or "/model/" in path):
            if file_hash(Path(path)) != expected:
                raise ValueError(f"historical core source changed: {path}")
            inputs[path] = expected
    for path, expected in source_manifest["inputs"].items():
        if "/email/" in path or "email_eps" in path:
            if file_hash(Path(path)) != expected:
                raise ValueError(f"Email historical input changed: {path}")
            inputs[path] = expected
    for seed in range(5):
        for arm in ARMS:
            path = source_root / "restricted" / f"email_{arm}_seed{seed}_slots.json"
            if file_hash(path) != source_manifest["artifact_sha256"][path.name]:
                raise ValueError("P2 cache seal mismatch")
            inputs[str(path)] = file_hash(path)
    sources = [Path(__file__), ROOT / "tests/test_stable_slot_gradients.py",
               ROOT / "experiments/stable_slot_structure.py"]
    inputs.update({str(p): file_hash(p) for p in sources})
    output.mkdir(parents=True, mode=0o755)
    write_json(output / "public_configuration.json", configuration())
    restricted = output / "restricted"
    restricted.mkdir(mode=0o700)
    oldmask = os.umask(0o077)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = noise_reference()
        records, samples, slot_rows, pairs, stages = [], [], [], [], []
        initial_hashes, batches = {}, {}
        manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), status="running", inputs=inputs,
                        torch=torch.__version__, privacy_claim=False, parameter_updates=0, noise_draws=0)
        write_json(restricted / "noise_reference.json", reference)
        with forward_backward_only():
            for partition_seed, model_seed, arm in condition_order():
                key = (partition_seed, arm)
                if key not in batches:
                    source = source_root / "restricted" / f"email_{arm}_seed{partition_seed}_slots.json"
                    batches[key], selection = load_batch(source, partition_seed)
                    write_json(restricted / f"selection_partition{partition_seed}_{arm}.json", selection)
                started = time.perf_counter()
                result = diagnose(batches[key], model_seed, reference)
                identity = dict(partition_seed=partition_seed, model_seed=model_seed, arm=arm)
                metrics = dict(**identity, **result["metrics"], seconds=time.perf_counter() - started)
                expected_hash = initial_hashes.setdefault(model_seed, metrics["initial_state_sha256"])
                if metrics["initial_state_sha256"] != expected_hash or not metrics["all_clip_bounds_ok"]:
                    raise AssertionError("initialization mismatch or clipping check failed")
                records.append(metrics)
                for target, name in ((samples, "samples"), (slot_rows, "slots"), (pairs, "pairs"), (stages, "stages")):
                    target.extend(dict(**identity, **r) for r in result[name])
                case_dir = restricted / f"partition{partition_seed}_model{model_seed}_{arm}"
                case_dir.mkdir(mode=0o700)
                np.savez_compressed(case_dir / "gradients.npz", **result["arrays"])
                if not (restricted / f"initial_model_seed{model_seed}.npz").exists():
                    np.savez_compressed(restricted / f"initial_model_seed{model_seed}.npz", **result["initial"])
                    write_json(restricted / "parameter_layout.json", result["layout"])
                if len(records) == 2:
                    smoke = dict(passed=all(r["parameters_unchanged"] and r["all_clip_bounds_ok"] and
                                           r["parameter_coordinates"] == 2340 and r["nonfinite_coordinates"] == 0 for r in records),
                                 same_initialization=records[0]["initial_state_sha256"] == records[1]["initial_state_sha256"],
                                 conditions=records, total_forward_backward_calls=32, parameter_updates=0, noise_draws=0)
                    write_json(restricted / "smoke_validation.json", smoke)
                    if not smoke["passed"] or not smoke["same_initialization"]:
                        raise AssertionError("smoke gate failed; remaining conditions forbidden")
                    print("two-condition smoke gate passed; continuing approved remaining 28 conditions", flush=True)
                print(f"completed partition={partition_seed} model={model_seed} arm={arm}; weights unchanged", flush=True)
        for name, rows in (("conditions", records), ("subgraphs", samples), ("slots", slot_rows),
                           ("slot_pairs", pairs), ("stages", stages)):
            write_csv(restricted / f"{name}.csv", rows)
        write_csv(restricted / "partition_summary.csv", grouped_summary(records, ("partition_seed", "arm")))
        write_csv(restricted / "model_summary.csv", grouped_summary(records, ("model_seed", "arm")))
        paired = []
        for p in range(5):
            for m in range(3):
                a, b = [next(r for r in records if r["partition_seed"] == p and r["model_seed"] == m and r["arm"] == arm) for arm in ARMS]
                paired.append(dict(partition_seed=p, model_seed=m, **{key + "_delta": b[key] - a[key]
                              if a[key] is not None and b[key] is not None else None for key in SUMMARY_KEYS}))
        write_csv(restricted / "paired.csv", paired)
        (restricted / "report.md").write_text(report(records, samples, slot_rows, stages, reference))
        if any(file_hash(Path(path)) != expected for path, expected in inputs.items()):
            raise AssertionError("source/historical input changed")
        manifest.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(), conditions=30,
                        forward_backward_calls=len(samples), slot_vectors=len(slot_rows),
                        all_parameters_unchanged=all(r["parameters_unchanged"] for r in records),
                        historical_hashes_unchanged=True, no_optimizer_constructed=True,
                        initial_state_sha256=initial_hashes,
                        artifacts={str(p.relative_to(restricted)): file_hash(p) for p in restricted.rglob("*") if p.is_file()})
        write_json(restricted / "manifest.json", manifest)
        return records
    finally:
        torch.set_num_threads(threads)
        os.umask(oldmask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs/structure_sampling/stable-slots-l2h2-20261031-v1")
    args = parser.parse_args()
    run(args.output, args.source_root)


if __name__ == "__main__":
    main()
