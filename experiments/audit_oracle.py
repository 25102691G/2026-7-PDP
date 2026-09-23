#!/usr/bin/env python3
"""Audit the evaluation ceiling and ranking quality for one prepared dataset.

The paper defines CELF as the coverage-ratio denominator (Section V-A).  This
script makes that oracle explicit, compares it with a singleton-influence
(degree) ranking, validates the prepared node mapping, and optionally measures
saved model rankings against both references.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import igraph
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from baselines.heuristics import get_influence, greedy
from experiments.evaluate_models import (
    checkpoint_metadata,
    load_net,
    ranking_scores,
    select_topk_nodes,
)
from experiments.graph_util import get_rev_dgl
from experiments.igraph_loader import load_prepared_graph
from dataset.prepare_datasets import DATASETS, collect_original_graph


PAPER_GRAPH_STATS = {
    "hepph": {"nodes": 12008, "edges": 118489, "directed": False},
}


def topk_by_singleton_influence(graph: igraph.Graph, k: int) -> list[int]:
    """Rank by one-seed spread, breaking ties by ascending node ID."""
    scores = np.asarray(graph.outdegree(), dtype=np.int64) + 1
    return np.lexsort((np.arange(graph.vcount()), -scores))[:k].tolist()


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) != len(values_b):
        raise ValueError("rank vectors must have the same length")
    ranks_a = average_ranks(np.asarray(values_a))
    ranks_b = average_ranks(np.asarray(values_b))
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def validate_mapping(dataset_dir: Path, metadata: dict) -> dict:
    rows = list(csv.DictReader((dataset_dir / "node_mapping.csv").open()))
    by_split = {"train": [], "test": []}
    originals = {"train": set(), "test": set()}
    for row in rows:
        split = row["split"]
        local_id = int(row["local_id"])
        original_key = "original_id" if "original_id" in row else "artifact_local_id"
        original_id = int(row[original_key])
        by_split[split].append(local_id)
        originals[split].add(original_id)

    contiguous = {
        split: sorted(ids) == list(range(int(metadata[f"{split}_nodes"])))
        for split, ids in by_split.items()
    }
    return {
        "rows": len(rows),
        "train_local_ids_contiguous": contiguous["train"],
        "test_local_ids_contiguous": contiguous["test"],
        "original_id_overlap": len(originals["train"] & originals["test"]),
    }


def edge_endpoints_in_range(graph: igraph.Graph) -> bool:
    return all(
        0 <= source < graph.vcount() and 0 <= target < graph.vcount()
        for source, target in graph.get_edgelist()
    )


def graph_audit(dataset: str, prepared_root: Path, k: int) -> tuple[dict, igraph.Graph, list[int]]:
    dataset_dir = prepared_root / dataset
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    _, graph, directed = load_prepared_graph(dataset, "test", prepared_root)
    celf_seeds, celf_coverage = greedy(graph, k)
    degree_seeds = topk_by_singleton_influence(graph, k)
    degree_coverage = get_influence(graph, degree_seeds)
    expected = PAPER_GRAPH_STATS.get(dataset)
    within_edges = int(metadata["train_edges"]) + int(metadata["test_edges"])
    canonical_edges = int(metadata["canonical_edges"])

    report = {
        "dataset": dataset,
        "paper_reference": expected,
        "raw_matches_paper": (
            expected is None
            or (
                int(metadata["original_nodes"]) == expected["nodes"]
                and canonical_edges == expected["edges"]
                and bool(metadata["directed"]) == expected["directed"]
            )
        ),
        "prepared": {
            "split_source": metadata.get("split_source"),
            "directed": directed,
            "train_assigned_nodes": int(metadata["train_assigned_nodes"]),
            "test_assigned_nodes": int(metadata["test_assigned_nodes"]),
            "train_active_nodes": int(metadata["train_nodes"]),
            "test_active_nodes": graph.vcount(),
            "train_edges": int(metadata["train_edges"]),
            "test_edges": graph.ecount(),
            "cross_edges_removed": int(metadata["cross_edges_removed"]),
            "within_split_edge_fraction": within_edges / canonical_edges,
            "test_isolates_removed": int(metadata["test_isolated_nodes_removed"]),
            "edge_endpoints_in_range": edge_endpoints_in_range(graph),
            **validate_mapping(dataset_dir, metadata),
        },
        "oracle": {
            "k": k,
            "celf_coverage": celf_coverage,
            "celf_ratio_percent": 100.0,
            "degree_coverage": degree_coverage,
            "degree_ratio_percent": 100.0 * degree_coverage / celf_coverage,
            "degree_celf_topk_overlap": len(set(degree_seeds) & set(celf_seeds)),
        },
    }
    return report, graph, celf_seeds


def load_full_raw_graph(dataset: str, raw_root: Path) -> igraph.Graph:
    """Load the canonical full graph for a diagnostic, without changing data."""
    spec = DATASETS[dataset]
    edges, nodes, _ = collect_original_graph(
        raw_root / spec.relative_path,
        spec,
        remove_self_loops=True,
        keep_duplicate_edges=False,
    )
    ordered_nodes = sorted(nodes)
    local_id = {node: index for index, node in enumerate(ordered_nodes)}
    graph = igraph.Graph(n=len(ordered_nodes), directed=spec.directed)
    graph.add_edges([(local_id[u], local_id[v]) for u, v in edges])
    return graph


def oracle_audit(graph: igraph.Graph, k: int) -> tuple[dict, list[int]]:
    celf_seeds, celf_coverage = greedy(graph, k)
    degree_seeds = topk_by_singleton_influence(graph, k)
    degree_coverage = get_influence(graph, degree_seeds)
    return {
        "nodes": graph.vcount(),
        "edges": graph.ecount(),
        "k": k,
        "celf_coverage": celf_coverage,
        "celf_ratio_percent": 100.0,
        "degree_coverage": degree_coverage,
        "degree_ratio_percent": 100.0 * degree_coverage / celf_coverage,
        "degree_celf_topk_overlap": len(set(degree_seeds) & set(celf_seeds)),
    }, celf_seeds


def model_audit(
    graph: igraph.Graph,
    celf_seeds: list[int],
    model_paths: list[Path],
    k: int,
) -> list[dict]:
    celf_coverage = get_influence(graph, celf_seeds)
    singleton_scores = np.asarray(graph.outdegree(), dtype=float) + 1
    graph_cache = {}
    records = []
    for path in model_paths:
        metadata = checkpoint_metadata(path)
        feature_protocol = metadata.get("feature_protocol", "ones")
        input_dim = int(metadata.get("input_dim") or 32)
        key = (feature_protocol, input_dim)
        if key not in graph_cache:
            graph_cache[key] = get_rev_dgl(
                graph, feature_protocol, input_dim, graph.is_directed(), False
            )
        dglgraph = graph_cache[key]
        net = load_net(path, use_cuda=False)
        net.eval()
        with torch.no_grad():
            scores = ranking_scores(net, dglgraph, dglgraph.ndata["feat"])
        flat_scores = scores.detach().cpu().reshape(-1).numpy()
        seeds = select_topk_nodes(scores, k).detach().cpu().tolist()
        coverage = get_influence(graph, seeds)
        records.append({
            "checkpoint": str(path),
            "seed": metadata.get("seed"),
            "feature_protocol": feature_protocol,
            "coverage": coverage,
            "coverage_ratio_percent": 100.0 * coverage / celf_coverage,
            "celf_topk_overlap": len(set(seeds) & set(celf_seeds)),
            "spearman_score_vs_singleton_influence": spearman(
                flat_scores, singleton_scores
            ),
            "score_std": float(np.std(flat_scores)),
        })
    return records


def optimize_paper_loss_oracle(
    graph: igraph.Graph,
    celf_seeds: list[int],
    k: int,
    steps: int,
    seeds: int = 5,
    learning_rate: float = 0.05,
    penalty: float = 1.0,
) -> list[dict]:
    """Optimize Eq. (5) probabilities directly, without a GNN.

    This is a diagnostic upper bound for the repository's one-step loss, not
    a trainable or private method.  It distinguishes loss-objective behavior
    from GNN representation/generalization behavior.
    """
    sources = list(range(graph.vcount()))
    targets = list(range(graph.vcount()))
    for source, target in graph.get_edgelist():
        sources.append(source)
        targets.append(target)
        if not graph.is_directed():
            sources.append(target)
            targets.append(source)
    source_index = torch.tensor(sources, dtype=torch.long)
    target_index = torch.tensor(targets, dtype=torch.long)
    celf_coverage = get_influence(graph, celf_seeds)
    records = []
    for seed in range(seeds):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.nn.Parameter(
            torch.randn(graph.vcount(), generator=generator) * 0.01
        )
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            probabilities = torch.sigmoid(logits)
            log_not_selected = torch.log1p(
                -probabilities[source_index].clamp(max=1 - 1e-7)
            )
            log_not_covered = torch.zeros(graph.vcount())
            log_not_covered.index_add_(0, target_index, log_not_selected)
            loss = torch.exp(log_not_covered).sum() + penalty * probabilities.sum()
            loss.backward()
            optimizer.step()
        selected = select_topk_nodes(logits.detach(), k).tolist()
        coverage = get_influence(graph, selected)
        records.append({
            "seed": seed,
            "steps": steps,
            "penalty": penalty,
            "coverage": coverage,
            "coverage_ratio_percent": 100.0 * coverage / celf_coverage,
            "celf_topk_overlap": len(set(selected) & set(celf_seeds)),
            "probability_sum": float(torch.sigmoid(logits).sum()),
            "final_loss": float(loss),
        })
    return records


def markdown_report(report: dict) -> str:
    prepared = report["prepared"]
    oracle = report["oracle"]
    lines = [
        f"# {report['dataset']} Oracle 与数据链路审计",
        "",
        "论文口径：Section V-A，IC 概率 1、一步传播、k=50，Coverage Ratio 以 CELF 为分母。",
        "",
        "## 数据完整性",
        "",
        f"- 原始规模与论文 Table I 一致：`{report['raw_matches_paper']}`",
        f"- 当前测试图：{prepared['test_active_nodes']} 个活跃节点、{prepared['test_edges']} 条边",
        f"- 随机拆点后保留的原始边比例：{prepared['within_split_edge_fraction']:.2%}",
        f"- 被删除的跨 split 边：{prepared['cross_edges_removed']}",
        f"- 测试 split 删除的孤立节点：{prepared['test_isolates_removed']}",
        f"- 节点映射连续、train/test 原始 ID 不重叠：`{prepared['train_local_ids_contiguous'] and prepared['test_local_ids_contiguous'] and prepared['original_id_overlap'] == 0}`",
        "",
        "## Oracle 与简单排序",
        "",
        f"- CELF Oracle 覆盖：{oracle['celf_coverage']}（定义上为 100%）",
        f"- 单点影响力/度排序覆盖：{oracle['degree_coverage']}（{oracle['degree_ratio_percent']:.2f}%）",
        f"- 度排序与 CELF Top-{oracle['k']} 重合：{oracle['degree_celf_topk_overlap']}/{oracle['k']}",
    ]
    models = report.get("models", [])
    if models:
        lines.extend([
            "",
            "## Non-Private checkpoint 排序",
            "",
            "| seed | coverage ratio | CELF Top-50 overlap | Spearman(score, singleton spread) |",
            "|---:|---:|---:|---:|",
        ])
        for row in models:
            lines.append(
                f"| {row['seed']} | {row['coverage_ratio_percent']:.2f}% | "
                f"{row['celf_topk_overlap']}/50 | "
                f"{row['spearman_score_vs_singleton_influence']:.4f} |"
            )
        ratios = np.asarray([row["coverage_ratio_percent"] for row in models])
        overlaps = np.asarray([row["celf_topk_overlap"] for row in models])
        correlations = np.asarray([
            row["spearman_score_vs_singleton_influence"] for row in models
        ])
        lines.extend([
            "",
            f"均值 ± 样本标准差：{ratios.mean():.2f}% ± {ratios.std(ddof=1):.2f}%；"
            f"平均 CELF Top-50 重合 {overlaps.mean():.1f}/50；"
            f"平均 Spearman {correlations.mean():.4f}。",
        ])
    soft_oracle = report.get("paper_loss_oracle", [])
    if soft_oracle:
        lines.extend([
            "",
            "## 论文 Eq. (5) 自由概率上限诊断",
            "",
            "| lambda | coverage ratio | CELF Top-50 overlap | probability sum |",
            "|---:|---:|---:|---:|",
        ])
        for penalty in sorted({row["penalty"] for row in soft_oracle}):
            group = [row for row in soft_oracle if row["penalty"] == penalty]
            ratios = np.asarray([row["coverage_ratio_percent"] for row in group])
            overlaps = np.asarray([row["celf_topk_overlap"] for row in group])
            probability_sums = np.asarray([row["probability_sum"] for row in group])
            lines.append(
                f"| {penalty:g} | {ratios.mean():.2f}% ± {ratios.std(ddof=1):.2f}% | "
                f"{overlaps.mean():.1f}/50 | {probability_sums.mean():.1f} |"
            )
    full_raw = report.get("full_raw_diagnostic")
    if full_raw:
        oracle = full_raw["oracle"]
        lines.extend([
            "",
            "## 完整原始图诊断（非论文默认 split 解释）",
            "",
            f"完整图有 {oracle['nodes']} 个节点、{oracle['edges']} 条边；"
            f"CELF 覆盖 {oracle['celf_coverage']}。",
        ])
        full_models = full_raw.get("models", [])
        if full_models:
            ratios = np.asarray([
                row["coverage_ratio_percent"] for row in full_models
            ])
            lines.append(
                f"相同五个 checkpoint 在完整图上的覆盖率为 "
                f"{ratios.mean():.2f}% ± {ratios.std(ddof=1):.2f}%。"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hepph")
    parser.add_argument("--prepared-root", type=Path, default=PROJECT_ROOT / "dataset" / "prepared")
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--include-full-raw", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "raw_dataset")
    parser.add_argument("--soft-oracle-steps", type=int, default=0)
    parser.add_argument("--soft-oracle-penalty", type=float, nargs="+", default=[1.0])
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "diagnostics" / "hepph_oracle_audit.json")
    args = parser.parse_args()

    report, graph, celf_seeds = graph_audit(
        args.dataset, args.prepared_root, args.k
    )
    if args.models_root:
        model_paths = sorted(args.models_root.rglob("*.pt"))
        report["models"] = model_audit(
            graph, celf_seeds, model_paths, args.k
        )
    else:
        model_paths = []

    if args.soft_oracle_steps:
        report["paper_loss_oracle"] = []
        for penalty in args.soft_oracle_penalty:
            report["paper_loss_oracle"].extend(optimize_paper_loss_oracle(
                graph, celf_seeds, args.k, args.soft_oracle_steps,
                penalty=penalty,
            ))

    if args.include_full_raw:
        full_graph = load_full_raw_graph(args.dataset, args.raw_root)
        full_oracle, full_celf_seeds = oracle_audit(full_graph, args.k)
        report["full_raw_diagnostic"] = {"oracle": full_oracle}
        if model_paths:
            report["full_raw_diagnostic"]["models"] = model_audit(
                full_graph, full_celf_seeds, model_paths, args.k
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(report))
    print(markdown_report(report), end="")
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
