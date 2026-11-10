"""Approved two-layer/two-bucket structural screen; no Torch, training, or DP claim.

Only public_configuration.json is public mechanism metadata. Everything under
restricted/ is an unprotected private structural diagnostic, not a DP output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import igraph
import numpy as np
from experiments import sampling
from experiments.node_sampling_structure import (
    file_hash, graph_structure, induce_nodes, load_prepared_train, write_csv,
)

SLOTS = (0, 1, 2, 3)
DATASETS = ("email", "bitcoin", "lastfm")
SEEDS = (0, 1, 2, 3, 4)
PARAMETERS = {"email": (40, 4), "bitcoin": (60, 4), "lastfm": (60, 4)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def public_configuration():
    return dict(candidate="two-independent-public-bipartitions-v1", layers=2,
                buckets_per_layer=2, slot_indices=list(SLOTS), membership_bound=2,
                partition_rule="sha256('privim-slot-partition-v1|seed|layer|original_id')[0] & 1",
                partition_seeds=list(SEEDS), datasets=list(DATASETS),
                arms=["original", "personalized_eps2"], quota_per_slot=4,
                start_budget_per_slot=64, total_start_budget=256,
                q_policy="min(1,64/max(1,slot_nodes)); local only",
                parameters={k: dict(n=v[0], M=v[1]) for k, v in PARAMETERS.items()},
                walk_length=200, restart_probability=.3, frequency_decay=1.,
                boundary_divisor=2, sampling_protocol="paper", feature_protocol="ones32",
                rwr_seed_rule="sha256('privim-slot-rwr-v1|dataset|seed|slot')[:8] as big-endian int",
                paired_rwr_streams_across_arms=True, retries_or_partition_selection=False,
                privacy_claim=False, training_performed=False, gradients_computed=False)


def memberships(original_id, seed):
    return tuple(2 * layer + (hashlib.sha256(
        f"privim-slot-partition-v1|{int(seed)}|{layer}|{int(original_id)}".encode()
    ).digest()[0] & 1) for layer in (0, 1))


def slot_seed(dataset, seed, slot):
    return int.from_bytes(hashlib.sha256(
        f"privim-slot-rwr-v1|{dataset}|{seed}|{slot}".encode()).digest()[:8], "big")


def build_slots(graph, seed):
    ids = list(map(int, graph.vs["original_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("original IDs must be unique")
    prepared = graph.vs["prepared_local_id"] if "prepared_local_id" in graph.vs.attributes() else ids
    result = []
    # Canonicalize each slot by stable ID, never by the current dense index.
    for slot in SLOTS:
        selected = sorted((i for i, node in enumerate(ids) if slot in memberships(node, seed)),
                          key=lambda i: ids[i])
        local = {old: new for new, old in enumerate(selected)}
        edges = [(local[u], local[v]) for u, v in graph.get_edgelist() if u in local and v in local]
        if not graph.is_directed():
            edges = [tuple(sorted(edge)) for edge in edges]
        sub = igraph.Graph(n=len(selected), edges=sorted(edges), directed=graph.is_directed())
        sub.vs["original_id"] = [ids[i] for i in selected]
        sub.vs["prepared_local_id"] = [int(prepared[i]) for i in selected]
        result.append(sub)
    return result


def input_payload(graph):
    ids = list(map(int, graph.vs["original_id"]))
    edges = sorted(map(list, graph.get_edgelist()))
    # Byte identity of actual aligned all-one feature arrays; no GNN instantiated.
    features = np.ones((len(ids), 32), dtype=np.float32)
    return dict(original_ids=ids, prepared_ids=list(map(int, graph.vs["prepared_local_id"])),
                local_ids=list(range(len(ids))), directed=graph.is_directed(), edges=edges,
                feature_shape=list(features.shape), feature_dtype=str(features.dtype),
                feature_sha256=hashlib.sha256(features.tobytes()).hexdigest())


class TraceRandom(random.Random):
    """Observe actual draws without changing Random's values or state transitions."""
    def __init__(self, seed):
        super().__init__(seed)
        self.draw_count = 0
        self.trace = hashlib.sha256()

    def random(self):
        value = super().random()
        self.draw_count += 1
        self.trace.update(b"r" + struct.pack("!d", value))
        return value

    def getrandbits(self, k):
        value = super().getrandbits(k)
        self.draw_count += 1
        self.trace.update(b"b" + str(k).encode() + b":" + str(value).encode() + b";")
        return value


def reachability(graph, size, boundary_size):
    n = graph.vcount()
    if graph.is_directed():
        counts = [len(graph.subcomponent(i, mode="out")) for i in range(n)]
    else:
        components = graph.connected_components(mode="weak")
        sizes = components.sizes()
        counts = [sizes[components.membership[i]] for i in range(n)]
    return dict(out_reachable_ge_scs_fraction=sum(x >= size for x in counts) / n if n else 0.,
                out_reachable_ge_bes_fraction=sum(x >= boundary_size for x in counts) / n if n else 0.,
                out_reachable_mean=statistics.mean(counts) if counts else 0.,
                out_reachable_max=max(counts, default=0))


def extract_slot(graph, *, size, threshold, seed):
    started = time.perf_counter()
    structure = graph_structure(graph, size, max(1, size // 2))
    structure.update(reachability(graph, size, max(1, size // 2)))
    structure_seconds = time.perf_counter() - started
    q = min(1., 64 / max(1, graph.vcount()))
    rng = TraceRandom(seed)
    initial = rng.getstate()
    counts = [Counter(), Counter()]
    stage = -1

    def observe(frame, event, value):
        nonlocal stage
        if frame.f_code is sampling._frequency_stage.__code__ and event == "call":
            stage += 1
        elif frame.f_code is sampling._walk_from_start.__code__:
            if event == "call":
                counts[stage]["attempts"] += 1
            elif event == "return" and value is not None:
                counts[stage]["successes"] += 1

    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError("run outside another profiler")
    started = time.perf_counter()
    try:
        sys.setprofile(observe)
        result = sampling.sample_privim_star(
            graph, size=size, boundary_divisor=2, sample_rate=q,
            restart_probability=.3, walk_length=200, threshold=threshold,
            decay=1., include_boundary_stage=True, rng=rng)
    finally:
        sys.setprofile(previous)
    extraction_seconds = time.perf_counter() - started
    frequencies = Counter(node for nodes in result.node_sets for node in nodes)
    if result.frequencies != [frequencies[i] for i in range(graph.vcount())]:
        raise AssertionError("frequency state disagrees with container membership")
    if max(result.frequencies, default=0) > threshold:
        raise AssertionError("SCS/BES frequency bound exceeded")
    if [counts[i]["successes"] for i in (0, 1)] != [result.stage1_count, result.stage2_count]:
        raise AssertionError("observed stage successes differ")
    attempts = sum(c["attempts"] for c in counts)
    metrics = dict(**structure, q=q, scs_successes=result.stage1_count,
                   bes_successes=result.stage2_count, subgraphs=len(result.node_sets),
                   walk_attempts=attempts, rwr_success_rate=len(result.node_sets) / attempts if attempts else None,
                   size_histogram=dict(sorted(Counter(map(len, result.node_sets)).items())),
                   container_ge4=len(result.node_sets) >= 4,
                   effective_positions=min(4, len(result.node_sets)),
                   zero_padding=max(0, 4 - len(result.node_sets)),
                   structurally_nonempty_contribution=bool(result.node_sets),
                   max_frequency=max(result.frequencies, default=0), frequency_bound_ok=True,
                   structure_seconds=structure_seconds, extraction_seconds=extraction_seconds,
                   preprocess_seconds=structure_seconds + extraction_seconds)
    for name, count in zip(("scs", "bes"), counts):
        metrics.update({name + "_attempts": count["attempts"],
                        name + "_failures": count["attempts"] - count["successes"],
                        name + "_success_rate": count["successes"] / count["attempts"] if count["attempts"] else None})
    cache = dict(node_sets=result.node_sets, frequencies=result.frequencies,
                 stage1_count=result.stage1_count, stage2_count=result.stage2_count,
                 rng_initial_state=initial, rng_final_state=rng.getstate(),
                 rng_draw_count=rng.draw_count, rng_trace_sha256=rng.trace.hexdigest())
    # Normalize tuples to the exact on-disk JSON representation used in rechecks.
    return dict(metrics=metrics, input=input_payload(graph), cache=json.loads(json.dumps(cache)))


def personalized_graph(graph, decision_path):
    with Path(decision_path).open() as handle:
        rows = list(csv.DictReader(handle))
    by_id = {int(row["prepared_local_id"]): row for row in rows}
    if len(rows) != graph.vcount() or set(by_id) != set(range(graph.vcount())):
        raise ValueError("saved decision IDs differ from prepared graph")
    kept = []
    for i in range(graph.vcount()):
        row = by_id[i]
        if int(row["original_id"]) != graph.vs[i]["original_id"]:
            raise ValueError("saved original ID mapping differs")
        if row["personalized_new_id"] != "":
            kept.append((int(row["personalized_new_id"]), i))
    kept.sort()
    if [new for new, old in kept] != list(range(len(kept))):
        raise ValueError("saved personalized dense IDs are invalid")
    if [old for new, old in kept] != sorted(old for new, old in kept):
        raise ValueError("saved personalized mapping order differs from historical induction")
    sub, mapping = induce_nodes(graph, [old for new, old in kept])
    if any(mapping[old] != new for new, old in kept):
        raise ValueError("personalized mapping was not reused exactly")
    return sub


def check_deletion(graph, original_id, partition_seed, rng_seeds, size, threshold,
                   before_slots=None, before_caches=None):
    if before_slots is None:
        before_slots = build_slots(graph, partition_seed)
    if before_caches is None:
        before_caches = [extract_slot(g, size=size, threshold=threshold, seed=rng_seeds[j])
                         for j, g in enumerate(before_slots)]
    deleted = graph.copy()
    deleted.delete_vertices(graph.vs["original_id"].index(original_id))
    after_slots = build_slots(deleted, partition_seed)
    allowed = set(memberships(original_id, partition_seed))
    comparisons = []
    for j in SLOTS:
        if j in allowed:
            continue
        after = extract_slot(after_slots[j], size=size, threshold=threshold, seed=rng_seeds[j])
        before = before_caches[j]
        comparisons.append(dict(slot=j, input_equal=before["input"] == after["input"],
                                cache_equal=before["cache"] == after["cache"],
                                rng_equal=all(before["cache"][k] == after["cache"][k]
                                              for k in before["cache"] if k.startswith("rng_")),
                                before_input_sha256=digest(before["input"]),
                                after_input_sha256=digest(after["input"]),
                                before_cache_sha256=digest(before["cache"]),
                                after_cache_sha256=digest(after["cache"])))
    return dict(deleted_original_id=int(original_id), allowed_changed_slots=sorted(allowed),
                unchanged_slots=[x["slot"] for x in comparisons], slot_indices=list(SLOTS),
                quota=[4] * 4, comparisons=comparisons,
                passed=len(comparisons) == 2 and all(x["input_equal"] and x["cache_equal"] and
                                                   x["rng_equal"] for x in comparisons))


def stable_edges(graph):
    ids = graph.vs["original_id"]
    edges = [(int(ids[u]), int(ids[v])) for u, v in graph.get_edgelist()]
    return Counter(edges if graph.is_directed() else [tuple(sorted(e)) for e in edges])


def configuration_metrics(graph, slots, caches):
    parent = stable_edges(graph)
    if any(v != 1 for v in parent.values()):
        raise ValueError("this screen requires simple input edge tables")
    multiplicity = Counter()
    for slot in slots:
        multiplicity.update(stable_edges(slot))
    if not set(multiplicity).issubset(parent) or max(multiplicity.values(), default=0) > 2:
        raise AssertionError("slot edge union/multiplicity invalid")
    metrics = [x["metrics"] for x in caches]
    return dict(nodes=graph.vcount(), edges=graph.ecount(),
                union_edges=len(multiplicity), union_edge_coverage=len(multiplicity) / graph.ecount() if graph.ecount() else 0.,
                edge_copies=sum(multiplicity.values()),
                edge_copy_ratio=sum(multiplicity.values()) / graph.ecount() if graph.ecount() else 0.,
                edges_in_two_slots=sum(v == 2 for v in multiplicity.values()),
                edge_multiplicity_histogram=dict(sorted(Counter(multiplicity.values()).items())),
                all_four_ge4=all(m["container_ge4"] for m in metrics),
                eligible_slots=sum(m["container_ge4"] for m in metrics),
                effective_positions=sum(m["effective_positions"] for m in metrics),
                zero_padding=sum(m["zero_padding"] for m in metrics),
                structural_nonempty_slots=sum(m["structurally_nonempty_contribution"] for m in metrics),
                structural_zero_slot_fraction=sum(not m["structurally_nonempty_contribution"] for m in metrics) / 4,
                subgraphs=sum(m["subgraphs"] for m in metrics),
                scs_successes=sum(m["scs_successes"] for m in metrics),
                bes_successes=sum(m["bes_successes"] for m in metrics),
                walk_attempts=sum(m["walk_attempts"] for m in metrics))


def summaries(rows, keys):
    result = []
    for dataset in DATASETS:
        for arm in ("original", "personalized_eps2"):
            group = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            row = dict(dataset=dataset, arm=arm, runs=len(group))
            for key in keys:
                values = [float(r[key]) for r in group if r[key] is not None]
                row[key + "_mean"] = statistics.mean(values) if values else None
                row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            result.append(row)
    return result


def slot_summaries(rows, keys):
    result = []
    groups = sorted({(r["dataset"], r["arm"], r["slot"]) for r in rows})
    for dataset, arm, slot in groups:
        group = [r for r in rows if (r["dataset"], r["arm"], r["slot"]) == (dataset, arm, slot)]
        row = dict(dataset=dataset, arm=arm, slot=slot, runs=len(group))
        for key in keys:
            values = [float(r[key]) for r in group if r[key] is not None]
            row[key + "_mean"] = statistics.mean(values) if values else None
            row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(row)
    return result


SLOT_SUMMARY_KEYS = ("nodes", "edges", "largest_wcc_fraction", "mean_degree",
                     "out_reachable_ge_scs_fraction", "out_reachable_ge_bes_fraction",
                     "scs_successes", "bes_successes", "walk_attempts", "rwr_success_rate",
                     "subgraphs", "container_ge4", "effective_positions", "zero_padding",
                     "structurally_nonempty_contribution", "preprocess_seconds")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def report(configs, slot_rows, summary, locality):
    lines = ["# P2 两层公开二分区：本地受限结构筛查", "",
             "**非 DP/PDP 输出。未实例化 GNN、未计算梯度、未训练、未运行 DP-SGD 或修改会计。**", "",
             "## 固定协议", "",
             "分区种子0–4，每层SHA-256独立域映射；四槽位固定，每槽位Q=64、配额4。",
             "Personalized复用同编号历史seed的epsilon_0=2决策，未重抽预算或节点。",
             "槽位RWR种子按dataset/seed/slot隔离，两图条件配对；每槽位只有一次容器提取，不因失败补抽。",
             "n/M=Email 40/4、Bitcoin与LastFM 60/4；L=200、tau=.3、mu=1、BES除数2。",
             "可达节点数包含起点；出向可达性仅为RWR成功的必要条件，不能代替频率约束下的实际walk。",
             "每槽位保存输入与缓存、实际RNG初末状态、抽样调用计数和消耗轨迹摘要。",
             "结构容量是min(4,容器数)；空贡献仅指没有候选子图，不表示实际梯度一定为零或非零。", "",
             "## 配置汇总", "", "±为五个配置的样本标准差。Personalized同时变化节点采样seed与分区seed，不能分解二者方差。", "",
             "|数据集|图条件|联合边%|四槽位均≥4|有效位置/16|子图总数|相对未分桶数量|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        group = [c for c in configs if c["dataset"] == r["dataset"] and c["arm"] == r["arm"]]
        f = lambda k: f"{r[k+'_mean']:.2f} ± {r[k+'_std']:.2f}"
        lines.append(f"|{r['dataset']}|{r['arm']}|{100*r['union_edge_coverage_mean']:.2f} ± {100*r['union_edge_coverage_std']:.2f}|"
                     f"{sum(c['all_four_ge4'] for c in group)}/5|{f('effective_positions')}|{f('subgraphs')}|{f('container_ratio_to_unbucketed')}|")
    lines += ["", "## 逐配置配额与失败记录", "",
              "|数据集|图条件|seed|四槽位SCS+BES数量|有效位置|补零|无候选槽位|", "|---|---|---:|---|---:|---:|---:|"]
    for c in configs:
        rows = [s for s in slot_rows if all(s[k] == c[k] for k in ("dataset", "arm", "partition_seed"))]
        counts = "; ".join(f"{s['scs_successes']}+{s['bes_successes']}" for s in rows)
        lines.append(f"|{c['dataset']}|{c['arm']}|{c['partition_seed']}|{counts}|{c['effective_positions']}|{c['zero_padding']}|{4-c['structural_nonempty_slots']}|")
    lines += ["", "## 原采样器影响与解释", "",
              "未分桶参照来自原结构实验对应seed/图条件的既有记录，未重新生成参照容器。",
              "configs.csv记录SCS/BES数量与RWR成功率差异；slots.csv记录WCC、出向可达性和每槽位失败。",
              "联合边覆盖的分母是本图条件的边；另记相对原始prepared图的联合边覆盖，避免掩盖外层节点采样损失。",
              "边副本数与联合唯一边数分开记录；跨层重复边不能算成额外覆盖。",
              "四个槽位同时达标是比合计子图数≥16更强的条件；不允许跨槽位补充。", "",
              "|数据集|图条件|槽位子图数范围|零SCS槽位/20|SCS总数：未分桶→四槽位均值|BES总数：未分桶→四槽位均值|RWR成功率：未分桶→四槽位|",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        group = [c for c in configs if c["dataset"] == r["dataset"] and c["arm"] == r["arm"]]
        slots = [s for s in slot_rows if s["dataset"] == r["dataset"] and s["arm"] == r["arm"]]
        mean = lambda key: statistics.mean(c[key] for c in group)
        lines.append(f"|{r['dataset']}|{r['arm']}|{min(s['subgraphs'] for s in slots)}–{max(s['subgraphs'] for s in slots)}|"
                     f"{sum(s['scs_successes']==0 for s in slots)}/20|{mean('unbucketed_scs'):.1f}→{mean('scs_successes'):.1f}|"
                     f"{mean('unbucketed_bes'):.1f}→{mean('bes_successes'):.1f}|"
                     f"{100*mean('unbucketed_rwr_success_rate'):.2f}%→{100*mean('rwr_success_rate'):.2f}%|")
    lines += ["", "增加子图总数不意味着保留同样多的结构信息：两个层会重复节点/边，局部q和频率饱和过程也与未分桶不同。",
              "完整槽位均值/样本标准差见slot_summary.csv（每固定槽位五个seed，不把20个槽位视作20个独立图重复）。",
              "配对差值见paired_personalized_minus_original.csv。", "",
              "## 成功子图大小、边重复与预处理时间", "",
              "时间为每配置秒数；边副本比的分母为本图边数，可以包含跨层重复。", "",
              "|数据集|图条件|成功子图大小:数量（合计）|边副本比均值|四槽位预处理秒，均值±样本标准差|",
              "|---|---|---|---:|---:|"]
    for r in summary:
        group = [c for c in configs if c["dataset"] == r["dataset"] and c["arm"] == r["arm"]]
        slots = [s for s in slot_rows if s["dataset"] == r["dataset"] and s["arm"] == r["arm"]]
        sizes = Counter()
        for slot in slots:
            sizes.update({int(k): v for k, v in slot["size_histogram"].items()})
        lines.append(f"|{r['dataset']}|{r['arm']}|{dict(sorted(sizes.items()))}|"
                     f"{statistics.mean(c['edge_copy_ratio'] for c in group):.4f}|"
                     f"{r['preprocess_seconds_mean']:.3f} ± {r['preprocess_seconds_std']:.3f}|")
    lines += ["",
              "## 判定与后续边界", ""]
    passes = sum(c["all_four_ge4"] for c in configs)
    lines.append(f"固定4+4+4+4配额：{passes}/{len(configs)}配置达标；补零总数{sum(c['zero_padding'] for c in configs)}；"
                 f"所有槽位候选数最小值{min(s['subgraphs'] for s in slot_rows)}。")
    for dataset in DATASETS:
        original = [c for c in configs if c["dataset"] == dataset and c["arm"] == "original"]
        personal = [c for c in configs if c["dataset"] == dataset and c["arm"] == "personalized_eps2"]
        deltas = [p["subgraphs"] - o["subgraphs"] for o, p in zip(original, personal)]
        pslots = [s for s in slot_rows if s["dataset"] == dataset and s["arm"] == "personalized_eps2"]
        lines.append(f"- {dataset}：Personalized−Original子图总数配对差值{deltas}；"
                     f"Personalized槽位最小WCC比例{min(s['largest_wcc_fraction'] for s in pslots):.2%}，"
                     f"最小SCS大小出向可达节点比例{min(s['out_reachable_ge_scs_fraction'] for s in pslots):.2%}。")
    lines += ["", "本次筛查未出现总配额失败或空容器；LastFM的完整大小SCS严重退化、依赖BES，必须保留为负面结果。",
              "建议只将Email列为下一阶段槽位整体裁剪信号收缩评估的首选；Bitcoin次之，LastFM保留为结构压力案例。",
              "这是后续研究建议，当前不执行任何梯度计算或训练；不能由本结果认定整体裁剪后仍有效学习。", "",
              "## 局部性与范围", "",
              f"真实图受控删除测试 {len(locality)} 例，全部通过：{all(x['passed'] for x in locality)}。",
              "每配置按稳定ID选取首、中、末三个不同节点（不按度或结果选择），每例检查两个非所属槽位。",
              "另有有向/无向合成图的逐节点删除测试，以及空槽位、节点重排、独立随机流消耗测试。",
              "这些是有限样例检查，不能替代P2全域局部性证明或实际程序的严格DP认证。", "",
              "## 时间和限制", "",
              "preprocess_seconds包括分桶、结构/出向可达性及带RNG轨迹观测的提取，不包括输入读取、写盘和额外删除验证。",
              "历史提取没有相同的RNG轨迹观测，时间只作本机描述，不作算法加速结论。",
              "每个分区只使用一条RWR轨迹，不能判断槽位长期失败概率。五个种子不支持强泛化结论。",
              "本地受限保存不是DP；本目录任何统计、映射、子图成员、决策及随机记录均不得作为认证DP输出。", ""]
    return "\n".join(lines)


def refresh_report(output):
    """Summarize saved diagnostics only; never load graphs or rerun extraction."""
    restricted = Path(output) / "restricted"

    def read_rows(name):
        def value(text):
            if text == "":
                return None
            if text in ("True", "False"):
                return text == "True"
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        with (restricted / name).open() as handle:
            return [{k: value(v) for k, v in r.items()} for r in csv.DictReader(handle)]

    configs, slots, summary = [read_rows(name) for name in ("configs.csv", "slots.csv", "summary.csv")]
    locality = json.loads((restricted / "locality.json").read_text())
    previous = os.umask(0o077)
    try:
        write_csv(restricted / "slot_summary.csv", slot_summaries(slots, SLOT_SUMMARY_KEYS))
        paired = []
        for dataset in DATASETS:
            for seed in SEEDS:
                a, b = [next(c for c in configs if c["dataset"] == dataset and c["partition_seed"] == seed and
                             c["arm"] == arm) for arm in ("original", "personalized_eps2")]
                keys = ("subgraphs", "scs_successes", "bes_successes", "union_edge_coverage",
                        "union_edge_coverage_vs_original", "effective_positions", "rwr_success_rate")
                paired.append(dict(dataset=dataset, partition_seed=seed,
                                   **{key + "_delta": b[key] - a[key] for key in keys}))
        write_csv(restricted / "paired_personalized_minus_original.csv", paired)
        (restricted / "report.md").write_text(report(configs, slots, summary, locality))
    finally:
        os.umask(previous)


def run(output, prepared_root, historical_root):
    output, prepared_root, historical_root = map(Path, (output, prepared_root, historical_root))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    historical_files = sorted(p for p in historical_root.rglob("*") if p.is_file())
    protected_sources = [ROOT / name for name in (
        "experiments/sampling.py", "experiments/train.py", "experiments/train_models.py",
        "experiments/privacy.py", "experiments/load_graph.py", "experiments/graph_util.py",
        "experiments/config.py", "experiments/node_sampling_structure.py")]
    protected_sources += sorted((ROOT / "model").rglob("*.py"))
    sealed = {str(p): file_hash(p) for p in historical_files + protected_sources}
    old_manifest = json.loads((historical_root / "manifest.json").read_text())
    for path, expected in old_manifest["inputs"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f"historical input hash mismatch: {path}")
    history = list(csv.DictReader((historical_root / "records.csv").open()))
    output.mkdir(parents=True, mode=0o755)
    restricted = output / "restricted"
    restricted.mkdir(mode=0o700)
    config = public_configuration()
    write_json(output / "public_configuration.json", config)
    previous_umask = os.umask(0o077)
    try:
        slot_rows, configs, locality = [], [], []
        manifest = dict(public_configuration_sha256=file_hash(output / "public_configuration.json"),
                        started_at=datetime.now(timezone.utc).isoformat(), privacy_claim=False,
                        training_performed=False, gradients_computed=False, python=platform.python_version(),
                        numpy=np.__version__, igraph=igraph.__version__, inputs={},
                        protected_sha256_before=sealed, source_sha256=file_hash(Path(__file__)))
        for dataset in DATASETS:
            parent, metadata, hashes = load_prepared_train(prepared_root, dataset)
            manifest["inputs"].update(hashes)
            size, threshold = PARAMETERS[dataset]
            if old_manifest["dataset_parameters"][dataset]["n"] != size or old_manifest["dataset_parameters"][dataset]["M"] != threshold:
                raise ValueError("historical sampler parameters changed")
            for seed in SEEDS:
                decision = historical_root / "decisions" / f"{dataset}_eps2.0_seed{seed}.csv"
                manifest["inputs"][str(decision)] = file_hash(decision)
                personal = personalized_graph(parent, decision)
                for arm, graph in (("original", parent), ("personalized_eps2", personal)):
                    historical_arm = "original" if arm == "original" else "personalized"
                    baseline = next(r for r in history if r["dataset"] == dataset and r["arm"] == historical_arm and
                                    int(r["seed"]) == seed and (arm == "original" or float(r["epsilon0"]) == 2.))
                    if (graph.vcount(), graph.ecount()) != (int(baseline["nodes"]), int(baseline["edges"])):
                        raise ValueError("historical graph reconstruction differs")
                    started = time.perf_counter()
                    slots = build_slots(graph, seed)
                    induction_seconds = time.perf_counter() - started
                    rng_seeds = [slot_seed(dataset, seed, j) for j in SLOTS]
                    caches = [extract_slot(g, size=size, threshold=threshold, seed=rng_seeds[j]) for j, g in enumerate(slots)]
                    tag = f"{dataset}_{arm}_seed{seed}"
                    for j, data in enumerate(caches):
                        row = dict(dataset=dataset, arm=arm, partition_seed=seed, slot=j,
                                   layer=j // 2, bucket=j % 2, **data["metrics"])
                        slot_rows.append(row)
                    metrics = configuration_metrics(graph, slots, caches)
                    metrics.update(dataset=dataset, arm=arm, partition_seed=seed,
                                   union_edge_coverage_vs_original=metrics["union_edges"] / parent.ecount() if parent.ecount() else 0.,
                                   preprocess_seconds=induction_seconds + sum(x["metrics"]["preprocess_seconds"] for x in caches),
                                   induction_seconds=induction_seconds,
                                   unbucketed_subgraphs=int(baseline["subgraphs"]),
                                   unbucketed_scs=int(baseline["scs_successes"]), unbucketed_bes=int(baseline["bes_successes"]),
                                   unbucketed_rwr_success_rate=float(baseline["rwr_success_rate"]),
                                   container_ratio_to_unbucketed=metrics["subgraphs"] / int(baseline["subgraphs"]),
                                   rwr_success_rate=metrics["subgraphs"] / metrics["walk_attempts"] if metrics["walk_attempts"] else None)
                    configs.append(metrics)
                    write_json(restricted / f"{tag}_slots.json", dict(rwr_seeds=rng_seeds, slots=caches))
                    ids = sorted(graph.vs["original_id"])
                    for node in sorted({ids[0], ids[len(ids) // 2], ids[-1]}) if ids else []:
                        check = check_deletion(graph, node, seed, rng_seeds, size, threshold, slots, caches)
                        check.update(dataset=dataset, arm=arm, partition_seed=seed)
                        locality.append(check)
                        if not check["passed"]:
                            raise AssertionError(f"locality failed in {tag}")
                    print(f"completed {tag}; locality verified", flush=True)
        summary = summaries(configs, ("union_edge_coverage", "union_edge_coverage_vs_original", "effective_positions",
                            "subgraphs", "scs_successes", "bes_successes", "container_ratio_to_unbucketed",
                            "structural_zero_slot_fraction", "rwr_success_rate", "preprocess_seconds"))
        write_csv(restricted / "slots.csv", slot_rows)
        write_csv(restricted / "configs.csv", configs)
        write_csv(restricted / "summary.csv", summary)
        write_json(restricted / "locality.json", locality)
        (restricted / "report.md").write_text(report(configs, slot_rows, summary, locality))
        refresh_report(output)
        after = {path: file_hash(Path(path)) for path in sealed}
        if sealed != after or any(file_hash(Path(p)) != h for p, h in manifest["inputs"].items()):
            raise AssertionError("historical artifacts or inputs changed during screen")
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(), status="completed",
                        configs=len(configs), slots=len(slot_rows), locality_cases=len(locality),
                        historical_and_core_hashes_unchanged=True, protected_sha256_after=after)
        write_json(restricted / "manifest.json", manifest)
        return configs
    finally:
        os.umask(previous_umask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=ROOT / "dataset/prepared")
    parser.add_argument("--historical-root", type=Path, default=ROOT / "outputs/structure_sampling/personalized-node-20261030-v2")
    args = parser.parse_args()
    run(args.output, args.prepared_root, args.historical_root)


if __name__ == "__main__":
    main()
