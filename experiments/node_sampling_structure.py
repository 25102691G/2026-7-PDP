"""Independent structural feasibility experiment. No GNN training or DP claim.

Only graph-only modules are imported: the existing load_graph/igraph_loader
modules also import Torch/DGL, so prepared train edges are read here using the
same metadata-driven vertex-count and direction convention.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import igraph
import numpy as np
from experiments import sampling
from experiments.config import select_indicator_parameters

LEVELS = (1, 2, 4)
RATIOS = (0.2, 0.4, 0.4)
DATASETS = ("email", "bitcoin", "lastfm")


def stream_seed(seed, dataset, role):
    text = f"node-structure-v1|{seed}|{dataset}|{role}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def retention_probabilities(budgets, epsilon0):
    if not math.isfinite(epsilon0) or epsilon0 <= 0:
        raise ValueError("epsilon0 must be finite and positive")
    probabilities = []
    for budget in budgets:
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("node budgets must be finite and non-negative")
        # Algebraically expm1(budget)/expm1(epsilon0), avoiding overflow.
        value = 1.0 if budget >= epsilon0 else (
            math.exp(budget - epsilon0)
            * -math.expm1(-budget) / -math.expm1(-epsilon0)
        )
        probabilities.append(min(1.0, max(0.0, value)))
    return probabilities


def assign_budgets(node_count, seed):
    """Largest-remainder quotas, randomly permuted without any graph access."""
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    quotas = [int(node_count * ratio) for ratio in RATIOS]
    order = sorted(
        range(len(RATIOS)),
        key=lambda i: (-(node_count * RATIOS[i] - quotas[i]), i),
    )
    for i in order[:node_count - sum(quotas)]:
        quotas[i] += 1
    budgets = [level for level, count in zip(LEVELS, quotas) for _ in range(count)]
    random.Random(seed).shuffle(budgets)
    return budgets


def bernoulli_nodes(probabilities, rng):
    # Exactly one independent coin per node, including probabilities zero/one.
    return [node for node, probability in enumerate(probabilities)
            if rng.random() < probability]


def induce_nodes(graph, nodes):
    selected = sorted(nodes)
    if len(set(selected)) != len(selected) or any(
        node < 0 or node >= graph.vcount() for node in selected
    ):
        raise ValueError("selected nodes must be distinct valid prepared IDs")
    induced = graph.induced_subgraph(selected)
    return induced, {old: new for new, old in enumerate(selected)}


def expected_retention(graph, probabilities):
    if len(probabilities) != graph.vcount():
        raise ValueError("one probability is required per node")
    expected_nodes = sum(probabilities) / graph.vcount() if graph.vcount() else 0.0
    expected_edges = sum(
        probabilities[u] if u == v else probabilities[u] * probabilities[v]
        for u, v in graph.get_edgelist()
    )
    return expected_nodes, expected_edges / graph.ecount() if graph.ecount() else 0.0


def graph_structure(graph, scs_size, bes_size):
    n, e = graph.vcount(), graph.ecount()
    components = graph.connected_components(mode="weak")
    sizes = components.sizes()
    degrees = graph.degree(mode="all", loops=True)
    return {
        "nodes": n, "edges": e, "directed": graph.is_directed(),
        "weak_components": len(sizes),
        "largest_wcc_nodes": max(sizes, default=0),
        "largest_wcc_fraction": max(sizes, default=0) / n if n else 0.0,
        "isolated_nodes": sum(degree == 0 for degree in degrees),
        "isolated_fraction": sum(degree == 0 for degree in degrees) / n if n else 0.0,
        "zero_outdegree_fraction": sum(d == 0 for d in graph.outdegree()) / n if n else 0.0,
        "mean_degree": 2 * e / n if n else 0.0,
        "mean_outdegree": e / n if n and graph.is_directed() else (2 * e / n if n else 0.0),
        "wcc_nodes_ge_scs_size_fraction": sum(s for s in sizes if s >= scs_size) / n if n else 0.0,
        "wcc_nodes_ge_bes_size_fraction": sum(s for s in sizes if s >= bes_size) / n if n else 0.0,
    }


def observe_extraction(graph, *, size, threshold, seed, sample_rate=None):
    """Observe call/return events without replacing or editing sampler functions.

    Every arm includes the same profiler overhead. 'Attempt' means entry into
    _walk_from_start; starts skipped at frequency M are counted separately.
    """
    q = min(1.0, 256 / max(1, graph.vcount())) if sample_rate is None else sample_rate
    counts = [Counter(), Counter()]
    stage = -1
    stage_code = sampling._frequency_stage.__code__
    starts_code = sampling._candidate_starts.__code__
    walk_code = sampling._walk_from_start.__code__

    def observe(frame, event, value):
        nonlocal stage
        code = frame.f_code
        if code is stage_code and event == "call":
            stage += 1
        elif code is starts_code and event == "return":
            counts[stage]["candidate_starts"] += len(value)
        elif code is walk_code:
            if event == "call":
                counts[stage]["attempts"] += 1
            elif event == "return":
                counts[stage]["successes" if value is not None else "failures"] += 1

    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError("run outside another Python profiler")
    started = time.perf_counter()
    try:
        sys.setprofile(observe)
        result = sampling.sample_privim_star(
            graph, size=size, boundary_divisor=2, sample_rate=q,
            restart_probability=0.3, walk_length=200, threshold=threshold,
            decay=1.0, include_boundary_stage=True, rng=random.Random(seed),
        )
    finally:
        sys.setprofile(previous)
    elapsed = time.perf_counter() - started
    attempts = sum(c["attempts"] for c in counts)
    successes = sum(c["successes"] for c in counts)
    assert [c["successes"] for c in counts] == [result.stage1_count, result.stage2_count]
    recomputed = Counter(node for nodes in result.node_sets for node in nodes)
    assert result.frequencies == [recomputed[i] for i in range(graph.vcount())]
    assert max(result.frequencies, default=0) <= threshold
    metrics = {
        "result": result, "q": q, "subgraph_seconds": elapsed,
        "walk_attempts": attempts, "walk_successes": successes,
        "walk_failures": attempts - successes,
        "rwr_success_rate": successes / attempts if attempts else None,
        "subgraphs": len(result.node_sets),
        "max_frequency": max(result.frequencies, default=0),
        "frequency_bound_ok": max(result.frequencies, default=0) <= threshold,
        "subgraph_node_coverage": len(recomputed) / graph.vcount() if graph.vcount() else 0.0,
        "subgraph_size_histogram": dict(sorted(Counter(map(len, result.node_sets)).items())),
        "container_ge_batch16": len(result.node_sets) >= 16,
    }
    for prefix, c in zip(("scs", "bes"), counts):
        metrics.update({
            f"{prefix}_candidate_starts": c["candidate_starts"],
            f"{prefix}_frequency_skipped": c["candidate_starts"] - c["attempts"],
            f"{prefix}_attempts": c["attempts"],
            f"{prefix}_successes": c["successes"],
            f"{prefix}_failures": c["failures"],
            f"{prefix}_success_rate": c["successes"] / c["attempts"] if c["attempts"] else None,
        })
    return metrics


def load_prepared_train(root, dataset):
    directory = Path(root) / dataset
    metadata_path = directory / "metadata.json"
    edge_path = directory / f"{dataset}_train_edges.txt"
    nodes_path = directory / "train_nodes.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["train_ratio"] != 0.5:
        raise ValueError("the existing 50:50 split is required")
    n = int(metadata.get("train_nodes", metadata.get("train_assigned_nodes")))
    with edge_path.open(encoding="utf-8") as handle:
        edges = [tuple(map(int, line.split()[:2])) for line in handle if line.strip()]
    if any(u < 0 or v < 0 or u >= n or v >= n for u, v in edges):
        raise ValueError("edge endpoints disagree with prepared vertex count")
    graph = igraph.Graph(n=n, edges=edges, directed=bool(metadata["directed"]))
    # Match the original RWR adapter's removal of self-loops, never drop vertices.
    graph.delete_edges([e.index for e in graph.es if e.source == e.target])
    with nodes_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {int(row["local_id"]): int(row["original_id"]) for row in rows}
    if set(mapping) != set(range(n)) or len(rows) != n or len(set(mapping.values())) != n:
        raise ValueError("prepared/original node mapping is not one-to-one")
    graph.vs["prepared_local_id"] = list(range(n))
    graph.vs["original_id"] = [mapping[i] for i in range(n)]
    paths = (metadata_path, edge_path, nodes_path)
    return graph, metadata, {str(p.resolve()): file_hash(p) for p in paths}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["dataset"], row["arm"], row["epsilon0"])].append(row)
    summaries = []
    excluded = {"seed", "extraction_seed", "node_seed", "uniform_seed", "epsilon0"}
    for (dataset, arm, base), rows in groups.items():
        summary = dict(dataset=dataset, arm=arm, epsilon0=base, runs=len(rows))
        for key, first in rows[0].items():
            if key in excluded or isinstance(first, bool):
                continue
            values = [row[key] for row in rows if isinstance(row[key], (int, float))
                      and not isinstance(row[key], bool)]
            if values:
                summary[key + "_mean"] = statistics.mean(values)
                summary[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        histogram = Counter()
        for row in rows:
            histogram.update(row["subgraph_size_histogram"])
        summary["subgraph_size_histogram_total"] = dict(sorted(histogram.items()))
        summary["frequency_bound_all_ok"] = all(r["frequency_bound_ok"] for r in rows)
        summary["batch16_eligible_runs"] = sum(r["container_ge_batch16"] for r in rows)
        summary["no_walk_runs"] = sum(r["walk_attempts"] == 0 for r in rows)
        summaries.append(summary)
    return summaries


def report_markdown(summaries, configuration):
    def fmt(row, key, percent=False):
        mean, sd = row.get(key + "_mean"), row.get(key + "_std")
        if mean is None:
            return "NA"
        scale = 100 if percent else 1
        return f"{mean * scale:.2f} ± {sd * scale:.2f}" if sd is not None else f"{mean * scale:.2f}"

    lines = [
        "# 个性化节点采样：结构与子图提取可行性", "",
        "本实验不进行 GNN 训练；结果不构成严格 Node-DP/PDP 认证或隐私证明。", "",
        "## 协议与口径", "",
        "- 只读取既有训练边表、节点映射及元数据；不改变 50:50 划分，不读取测试边表。",
        "- εv={1,2,4}，20/40/40 使用最大余数取整。预算标签由独立公开 budget_seed=0 固定打乱，所有 ε0 和重复实验复用，不使用拓扑。",
        "- 每节点恰好一次伯努利抽样；跨 ε0 复用同一节点随机数以便配对比较，不重采、不补点。均匀对照从全体训练节点中无放回选取相同数量节点。",
        "- 采样后保留全部入选节点，包括新产生的孤立节点；新 ID 到 prepared/original ID 的对应保存在 decisions/。",
        "- n、M 从原图原协议参数选择函数确定后固定；采样后 q=min(1,256/N)，N 包含孤立节点。L=200、τ=0.3、μ=1、s=2，均不调整。",
        "- 直接调用原 sample_privim_star；用 Python call/return profiler 观测原函数，不替换函数、不消费额外随机数。",
        "- RWR 成功率=成功子图数/实际进入 walk 的次数；频率已满而跳过的起点另列，零次尝试记 NA。SCS/BES 分别记录。",
        "- 最大弱连通分量占比的分母是采样后节点数；mean_degree 为总度 2E/N，有向图另记 mean_outdegree=E/N。",
        "- 时间包含节点抽样/诱导、结构指标、带观测的子图提取，不含共享输入读取和文件写入。全部实验同样观测；仅作本机描述性时间比较。",
        "- ± 为跨随机种子的样本标准差（ddof=1）。B=16 仅用来标记容器能否满足原训练入口，不实际训练。", "",
        "概率公式 p=min(1,expm1(εv)/expm1(ε0))：", "",
        "| ε0 | p(1) | p(2) | p(4) | 名义理论节点保留率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for base in configuration["base_epsilons"]:
        ps = retention_probabilities(LEVELS, base)
        lines.append(f"| {base:g} | {ps[0]:.6f} | {ps[1]:.6f} | {ps[2]:.6f} | {sum(p*r for p,r in zip(ps,RATIOS)):.4%} |")
    lines += ["", "## 结构结果", "", "保留率及占比单位为 %；original 是未做外层节点抽样的训练图。", "",
              "| 数据集 | ε0 | 方法 | 理论节点% | 实际节点% | 实际边% | 最大WCC% | 孤立节点% | 总平均度 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        base = row["epsilon0"] if row["epsilon0"] != "" else "—"
        values = [fmt(row,k,True) for k in ("theoretical_node_retention", "node_retention", "edge_retention", "largest_wcc_fraction", "isolated_fraction")]
        lines.append(f"| {row['dataset']} | {base} | {row['arm']} | " + " | ".join(values + [fmt(row,"mean_degree")]) + " |")
    lines += ["", "## 原 SCS/BES 提取结果", "",
              "| 数据集 | ε0 | 方法 | SCS数量 | BES数量 | RWR成功% | 容器≥16的次数 | 子图覆盖节点% | 预处理秒 | 相对原图耗时 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        base = row["epsilon0"] if row["epsilon0"] != "" else "—"
        values = [fmt(row,"scs_successes"), fmt(row,"bes_successes"), fmt(row,"rwr_success_rate",True),
                  f"{row['batch16_eligible_runs']}/{row['runs']}", fmt(row,"subgraph_node_coverage",True),
                  fmt(row,"preprocess_seconds"), fmt(row,"preprocess_time_ratio")]
        lines.append(f"| {row['dataset']} | {base} | {row['arm']} | " + " | ".join(values) + " |")
    lines += ["", "## 定位与解释", "",
              "- 优先观察节点/边保留率、孤立比例、WCC，再结合 RWR 成功率和 SCS/BES 数量；WCC 较大并不保证有向 walk 能到达足够节点。",
              "- 预算分配规则与拓扑无关，但只使用一个固定预算分配实现；随机分配的实际各组仍可能有度数/位置偏差。差异不能归因为采样方法稳定优于或劣于均匀抽样，也不能外推到偏好与拓扑相关的情况。",
              "- 成功子图的大小由原算法固定，大小分布只反映成功样本，不反映失败时差多少节点；失败次数、可容纳目标大小的 WCC 节点占比另在 records.csv。",
              "- q 在采样后增大，可能维持甚至增加子图数量；数量不等于结构质量，应同时看边密度、覆盖节点率和失败率。", ""]
    for dataset in configuration["datasets"]:
        original = next(r for r in summaries if r["dataset"] == dataset and r["arm"] == "original")
        for row in summaries:
            if row["dataset"] != dataset or row["arm"] != "personalized":
                continue
            uniform = next(r for r in summaries if r["dataset"] == dataset and r["arm"] == "uniform_matched" and r["epsilon0"] == row["epsilon0"])
            edge = row["edge_retention_mean"] * 100
            wcc = row["largest_wcc_fraction_mean"] * 100
            count_ratio = row["subgraphs_mean"] / original["subgraphs_mean"] if original["subgraphs_mean"] else 0
            eligible = row["batch16_eligible_runs"]
            if eligible < row["runs"]:
                verdict = "部分/全部种子达不到原 B=16 容器要求，暂不建议按原训练配置开展效用实验"
            elif row["epsilon0"] >= 4:
                verdict = "数量门槛通过，但应降为后续结构压力测试，不作为首轮效用候选"
            elif row["epsilon0"] >= 3:
                verdict = "保留为谨慎的后续候选，优先级低于 ε0=2"
            else:
                verdict = "值得优先进行受控小规模效用测试"
            bes_share = row["bes_successes_mean"] / row["subgraphs_mean"] if row["subgraphs_mean"] else 0
            lines.append(f"- **{dataset}, ε0={row['epsilon0']:g}**：边保留 {edge:.2f}%，最大WCC {wcc:.2f}%，容器数为原图的 {count_ratio:.2f} 倍，BES占平均子图数的 {bes_share:.2%}；{eligible}/{row['runs']} 达到16。与均匀对照边保留率差 {(row['edge_retention_mean']-uniform['edge_retention_mean'])*100:+.2f} 个百分点。{verdict}；这不是效用已得到验证。")
    lines += ["", "## 频率、大小与限制", ""]
    for dataset in configuration["datasets"]:
        rows = [r for r in summaries if r["dataset"] == dataset]
        hist = Counter()
        for row in rows:
            hist.update(row["subgraph_size_histogram_total"])
        lines.append(f"- {dataset}：所有成功子图合计大小分布 {dict(sorted(hist.items()))}；所有运行满足 M 上限：{all(r['frequency_bound_all_ok'] for r in rows)}。")
    lines += ["", "保留这些结构只证明原采样器仍有输入和可用候选，不证明模型能学习有效排序。下一步优先 ε0=2，配对比较原图/个性化/均匀对照，并保持模型与评估协议不变；ε0=3 谨慎保留，ε0=4 暂不作为首轮训练候选。当前底层 PrivIM* 未获严格 Node-DP 认证，不得把后续经验训练称为已证明的 PDP。", "",
              "## 复现与文件", "", "以下记录本次命令；重跑时必须将 --output 替换为不存在的新目录，同名目录会被拒绝。需要 NumPy 1.24.1 和 igraph 0.11.5，不需要 Torch/DGL。", "", "```sh", configuration["command"], "```", "",
              "- records.csv：逐种子完整指标；summary.csv：均值与样本标准差。",
              "- budgets/：固定公开预算分配；decisions/：逐节点概率、抽样决策和重编号映射。",
              "- subgraphs/：原采样器实际返回的节点集合和频率，节点 ID 对应 decisions 中的新 ID。",
              "- manifest.json：参数、输入/源码 SHA256、随机流种子和运行环境。", ""]
    return "\n".join(lines)


def run_experiment(*, prepared_root, output, datasets, base_epsilons, seeds):
    if not datasets or not base_epsilons or not seeds:
        raise ValueError("datasets, base epsilons and seeds must be nonempty")
    if len(set(datasets)) != len(datasets) or len(set(base_epsilons)) != len(base_epsilons) or len(set(seeds)) != len(seeds):
        raise ValueError("duplicate experiment cells are not allowed")
    for base in base_epsilons:
        retention_probabilities(LEVELS, base)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("budgets", "decisions", "subgraphs"):
        (output / directory).mkdir()
    command = ("PYTHONPATH=" + shlex.quote(os.environ.get("PYTHONPATH", ""))
               + " PYTHONDONTWRITEBYTECODE=1 " + shlex.join([
                   sys.executable, "experiments/node_sampling_structure.py", "--datasets", *datasets,
                   "--epsilon0", *map(str, base_epsilons), "--seeds", *map(str, seeds),
                   "--prepared-root", str(Path(prepared_root).resolve()), "--output", str(output.resolve()),
               ]))
    configuration = dict(datasets=list(datasets), base_epsilons=list(base_epsilons), seeds=list(seeds),
                         budgets=LEVELS, ratios=RATIOS, budget_seed=0, command=command,
                         walk_length=200, restart_probability=0.3, frequency_decay=1.0,
                         boundary_divisor=2, batch_reference=16, q_policy="min(1,256/post_sample_nodes)")
    manifest = dict(configuration=configuration, started_at=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), igraph=igraph.__version__, numpy=np.__version__,
                    platform=platform.platform(), privacy_claim=False, training_performed=False,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in (
                        Path(__file__), ROOT / "experiments/sampling.py", ROOT / "experiments/config.py")},
                    inputs={}, dataset_parameters={})
    records = []
    for dataset in datasets:
        started = time.perf_counter()
        graph, metadata, hashes = load_prepared_train(prepared_root, dataset)
        load_seconds = time.perf_counter() - started
        manifest["inputs"].update(hashes)
        size, threshold = select_indicator_parameters(dataset, metadata["original_nodes"])
        budget_seed = stream_seed(0, dataset, "budgets")
        budgets = assign_budgets(graph.vcount(), seed=budget_seed)
        manifest["dataset_parameters"][dataset] = dict(n=size, M=threshold, budget_seed=budget_seed,
                                                        budget_counts=dict(Counter(budgets)))
        write_csv(output / "budgets" / f"{dataset}.csv", [dict(
            prepared_local_id=i, original_id=graph.vs[i]["original_id"], epsilon_v=budget,
        ) for i, budget in enumerate(budgets)])

        for seed in seeds:
            node_seed = stream_seed(seed, dataset, "nodes")
            uniform_seed = stream_seed(seed, dataset, "uniform")
            extraction_seed = stream_seed(seed, dataset, "extraction")
            baseline_seconds = None
            started = time.perf_counter()
            all_nodes = list(range(graph.vcount()))
            cases = [("original", "", all_nodes, [1.0] * graph.vcount(),
                      time.perf_counter() - started)]
            decisions = {}
            for base in base_epsilons:
                started = time.perf_counter()
                probabilities = retention_probabilities(budgets, base)
                personal = bernoulli_nodes(probabilities, random.Random(node_seed))
                personal_seconds = time.perf_counter() - started
                started = time.perf_counter()
                uniform = sorted(random.Random(uniform_seed).sample(range(graph.vcount()), len(personal)))
                uniform_seconds = time.perf_counter() - started
                cases.extend((("personalized", base, personal, probabilities, personal_seconds),
                              ("uniform_matched", base, uniform, probabilities, uniform_seconds)))
                decisions[base] = (probabilities, personal, uniform)
            for arm, base, selected, probabilities, selection_seconds in cases:
                started = time.perf_counter()
                induced, mapping = induce_nodes(graph, selected)
                induction_seconds = time.perf_counter() - started
                started = time.perf_counter()
                structure = graph_structure(induced, size, max(1, size // 2))
                structure_seconds = time.perf_counter() - started
                observed = observe_extraction(induced, size=size, threshold=threshold, seed=extraction_seed)
                result = observed.pop("result")
                preprocess_seconds = selection_seconds + induction_seconds + structure_seconds + observed["subgraph_seconds"]
                if arm == "original":
                    baseline_seconds = preprocess_seconds
                node_theory, edge_theory = expected_retention(graph, probabilities)
                if arm == "uniform_matched":
                    n, k = graph.vcount(), len(selected)
                    node_theory = k / n if n else 0.0
                    edge_theory = k * (k - 1) / (n * (n - 1)) if n > 1 else 0.0
                variance = sum(p * (1-p) for p in probabilities)
                row = dict(dataset=dataset, arm=arm, epsilon0=base, seed=seed,
                           node_seed=node_seed, uniform_seed=uniform_seed, extraction_seed=extraction_seed,
                           n=size, M=threshold, **structure, **observed,
                           theoretical_node_retention=node_theory, theoretical_edge_retention=edge_theory,
                           node_retention=induced.vcount() / graph.vcount() if graph.vcount() else 0.0,
                           edge_retention=induced.ecount() / graph.ecount() if graph.ecount() else 0.0,
                           node_retention_theoretical_sd=math.sqrt(variance)/graph.vcount() if graph.vcount() and arm == "personalized" else 0.0,
                           mean_degree_ratio=structure["mean_degree"]/(2*graph.ecount()/graph.vcount()) if graph.ecount() else 0.0,
                           selection_seconds=selection_seconds,
                           induction_seconds=induction_seconds, structure_seconds=structure_seconds,
                           preprocess_seconds=preprocess_seconds, preprocess_time_ratio=preprocess_seconds/baseline_seconds,
                           load_seconds=load_seconds)
                for level in LEVELS:
                    assigned = sum(b == level for b in budgets)
                    retained = sum(budgets[i] == level for i in selected)
                    row[f"budget_{level}_assigned"] = assigned
                    row[f"budget_{level}_retained"] = retained
                    row[f"budget_{level}_retention"] = retained / assigned if assigned else None
                records.append(row)
                tag = f"{dataset}_eps{base}_seed{seed}_{arm}"
                (output / "subgraphs" / f"{tag}.json").write_text(json.dumps(dict(
                    node_sets=result.node_sets, frequencies=result.frequencies,
                    stage1_count=result.stage1_count, stage2_count=result.stage2_count,
                ), separators=(",", ":")) + "\n", encoding="utf-8")
                print(f"{tag}: V={induced.vcount()} E={induced.ecount()} SCS={result.stage1_count} BES={result.stage2_count}", flush=True)
            for base, (probabilities, personal, uniform) in decisions.items():
                pmap = {old: new for new, old in enumerate(personal)}
                umap = {old: new for new, old in enumerate(uniform)}
                write_csv(output / "decisions" / f"{dataset}_eps{base}_seed{seed}.csv", [dict(
                    prepared_local_id=i, original_id=graph.vs[i]["original_id"], epsilon_v=budgets[i],
                    probability=probabilities[i], personalized_new_id=pmap.get(i, ""), uniform_new_id=umap.get(i, ""),
                ) for i in range(graph.vcount())])
    summaries = summarize(records)
    write_csv(output / "records.csv", records)
    # Rows with zero walk attempts can have fewer numeric summary keys.
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    write_csv(output / "summary.csv", [{key: row.get(key) for key in fields} for row in summaries])
    (output / "report.md").write_text(report_markdown(summaries, configuration), encoding="utf-8")
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["records"] = len(records)
    manifest["status"] = "completed"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epsilon0", nargs="+", type=float, default=[2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--prepared-root", type=Path, default=ROOT / "dataset/prepared")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing paths are refused")
    args = parser.parse_args()
    run_experiment(prepared_root=args.prepared_root, output=args.output,
                   datasets=args.datasets, base_epsilons=args.epsilon0, seeds=args.seeds)


if __name__ == "__main__":
    main()
