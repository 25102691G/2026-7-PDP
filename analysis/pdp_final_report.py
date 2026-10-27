"""Verify complete benchmark artifacts and export an unfiltered result report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/privim-pdp-mplconfig")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pdp.data import load_prepared, trusted_snapshot
from pdp.evaluation import celf_denominator
from pdp.positive_head import initialize_positive_head
from pdp.run import slot_configuration, training_configuration


RUNS = ["integer-pdp-pilot-v1", "stable-slots-pilot-v2",
        "positive-head-pilot-v3", "positive-head-confirmation-v1"]
DATASETS = ["email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla"]
METHODS = ["public_init", "non_private", "slots_zero_noise", "uniform_dp", "uniform_full", "pdp_slots"]
LABELS = {"public_init": "公开初始化", "non_private": "匹配零噪声",
          "slots_zero_noise": "全参与零噪声", "uniform_dp": "统一 DP：低参与",
          "uniform_full": "统一 DP：全参与", "pdp_slots": "Node-PDP"}


def report(output: Path) -> None:
    if output.exists():
        raise FileExistsError("never overwrite existing final reports")
    torch.set_num_threads(1)
    graphs, snapshots, denominators = {}, {}, {}
    for dataset in DATASETS:
        graphs[dataset] = load_prepared(dataset, "test", ROOT/"dataset/prepared")
        snapshots[dataset] = trusted_snapshot(dataset, ROOT/"dataset/prepared")
        denominators[dataset] = celf_denominator(graphs[dataset], 50)
    all_rows, final_rows, audit = [], [], []
    for run_id in RUNS:
        root = ROOT/"outputs/pdp_runs"/run_id
        manifest = json.loads((root/"manifest.json").read_text())
        protocol = manifest["registered_protocol"]
        if protocol.get("benchmark_data_public") is not True or not manifest.get("completed_at_utc"):
            raise ValueError("only complete public benchmarks can enter the report")
        expected = {(d, m, s) for d in protocol["datasets"] for m in protocol["methods"]
                    for s in protocol["model_seeds"]}
        observed = {(c["dataset"], c["method"], c["model_seed"]) for c in manifest["cells"]}
        if observed != expected or len(observed) != len(manifest["cells"]):
            raise ValueError("missing, extra or duplicate cells")
        for dataset in protocol["datasets"]:
            if snapshots[dataset] != manifest["identity"]["prepared_snapshots"][dataset]:
                raise ValueError("prepared split does not match recorded run")
        private_calls = reused = 0
        for cell in manifest["cells"]:
            if cell["status"] != "completed":
                raise ValueError("no successful-seed-only aggregation")
            graph, metrics = graphs[cell["dataset"]], cell["metrics"]
            selected = metrics["seed_local_ids"]
            if len(selected) != 50 or len(set(selected)) != 50:
                raise ValueError("invalid seed set")
            if min(selected) < 0 or max(selected) >= graph.vcount():
                raise ValueError("seed outside fixed public target")
            covered = set(selected)
            for identifier in selected:
                covered.update(graph.successors(identifier))
            if (metrics["spread"] != len(covered) or
                    metrics["celf_denominator"] != denominators[cell["dataset"]] or
                    abs(metrics["coverage_ratio_percent"]-100*len(covered)/denominators[cell["dataset"]]) > 1e-10):
                raise ValueError("independent spread/denominator check failed")
            if (metrics["diffusion_model"], metrics["edge_probability"],
                    metrics["diffusion_hops"], metrics["k"]) != ("IC", 1, 1, 50):
                raise ValueError("evaluation protocol changed")
            directory = root/"trusted/models"/cell["id"]
            for filename, digest in cell["artifact_sha256"].items():
                if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != digest:
                    raise ValueError("artifact checksum mismatch")
            certificate = json.loads((directory/"certificate.json").read_text())
            private = certificate.get("schema") in {"b0-integer-range-v1", "stable-slot-node-pdp-v1"}
            if private:
                private_calls += 1
                diagnostic = json.loads((root/"trusted"/(cell["id"]+"_diagnostics.json")).read_text())
                if diagnostic["completed_steps"] != protocol["training"]["private_steps"]:
                    raise ValueError("actual update count differs")
                if diagnostic.get("invalid_slot_queries", diagnostic.get("invalid_contributions", 0)):
                    raise ValueError("unexpected gradient fallback requires explicit investigation")
                if diagnostic.get("preprocessing_failed", False) or diagnostic.get("preprocessing_failed_slots", []):
                    raise ValueError("unexpected preprocessing failure")
                if run_id in RUNS[-2:]:
                    config = training_configuration(protocol, cell["dataset"], cell["method"], cell["model_seed"])
                    slots = slot_configuration(protocol, cell["method"])
                    initial = initialize_positive_head(config, slots.head_settings)
                    state = torch.load(directory/"parameters.pt", weights_only=True)
                    for name, parameter in initial.named_parameters():
                        if not parameter.requires_grad and not torch.equal(parameter, state[name]):
                            raise ValueError("public encoder was changed during private training")
                    if certificate["private_parameter_count"] != 33:
                        raise ValueError("unexpected private subspace dimension")
            reused += int("reused_from" in cell)
            row = {"run_id": run_id, "dataset": cell["dataset"], "method": cell["method"],
                   "model_seed": cell["model_seed"], **metrics}
            all_rows.append(row)
            if run_id in RUNS[-2:]:
                final_rows.append(row)
        audit.append({"run_id": run_id, "cells_verified": len(manifest["cells"]),
                      "private_calls": private_calls, "reused_controls": reused,
                      "registered_source_sha256": manifest["identity"]["implementation_sha256"],
                      "registered_protocol_sha256": manifest["identity"]["protocol_sha256"]})
    grouped = {}
    for dataset in DATASETS:
        for method in METHODS:
            rows = [r for r in final_rows if r["dataset"] == dataset and r["method"] == method]
            if not rows:
                continue
            if sorted(r["model_seed"] for r in rows) != [0, 1, 2, 3, 4]:
                raise ValueError("final comparison requires all five seeds")
            values = [r["coverage_ratio_percent"] for r in rows]
            grouped[(dataset, method)] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    pairs = []
    for dataset in DATASETS:
        candidate = {r["model_seed"]: r for r in final_rows if r["dataset"] == dataset and r["method"] == "pdp_slots"}
        for reference in ["public_init", "non_private", "uniform_dp", "uniform_full"]:
            other = {r["model_seed"]: r for r in final_rows if r["dataset"] == dataset and r["method"] == reference}
            differences = [candidate[s]["coverage_ratio_percent"]-other[s]["coverage_ratio_percent"] for s in range(5)]
            jaccards = []
            for seed in range(5):
                a, b = set(candidate[seed]["seed_local_ids"]), set(other[seed]["seed_local_ids"])
                jaccards.append(len(a & b)/len(a | b))
            pairs.append({"dataset": dataset, "reference": reference,
                          "mean_difference_pp": statistics.mean(differences),
                          "sample_std_difference": statistics.stdev(differences),
                          "all_five_differences": differences,
                          "mean_seed_set_jaccard": statistics.mean(jaccards)})
    output.mkdir(parents=True, exist_ok=False)
    with (output/"all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (output/"verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    (output/"paired_differences.json").write_text(json.dumps(pairs, indent=2)+"\n")
    text = ["# 最终六数据集结果", "", "全部五个模型 seed，均值 ± 样本标准差，单位 %。", "",
            "| 数据集 | 公开初始化 | 匹配零噪声 | 全参与零噪声 | 统一 DP：低参与 | 统一 DP：全参与 | Node-PDP |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        values = [f"{grouped[(dataset, method)]['mean']:.4f} ± {grouped[(dataset, method)]['std']:.4f}"
                  if (dataset, method) in grouped else "未预注册" for method in METHODS]
        text.append("| "+dataset+" | "+" | ".join(values)+" |")
    text += ["", "全参与零噪声控制只在四数据集确认阶段预注册；前两开发集没有补跑该单元。", "",
             "| 方法 | 六数据集等权均值 |", "|---|---:|"]
    for method in [m for m in METHODS if m != "slots_zero_noise"]:
        text.append(f"| {LABELS[method]} | {statistics.mean(grouped[(d, method)]['mean'] for d in DATASETS):.6f} |")
    text += ["", "未证实 Node-PDP 相对公开初始化或统一 DP 的额外 Coverage 收益。",
             "改进主要来自公开结构先验；全部开发失败轮与当前确认结果保留。",
             "模型证书是单次训练的保证，不是整套研究产物的联合隐私预算。"]
    (output/"results.md").write_text("\n".join(text)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    positions = range(6)
    for method, color, marker in [("public_init", "#666666", "o"), ("uniform_dp", "#E69F00", "s"),
                                  ("pdp_slots", "#0072B2", "^")]:
        axes[0].errorbar(list(positions), [grouped[(d, method)]["mean"] for d in DATASETS],
                         yerr=[grouped[(d, method)]["std"] for d in DATASETS], marker=marker,
                         color=color, label={"public_init":"Public initialization", "uniform_dp":"Uniform DP (low)", "pdp_slots":"Node-PDP"}[method],
                         capsize=3, linewidth=1.4)
    axes[0].set_ylabel("Coverage ratio (%)")
    axes[0].set_ylim(75, 101)
    axes[0].set_title("Fixed public encoder: mean and sample SD (n=5)")
    axes[0].legend(fontsize=8)
    selected = [next(p for p in pairs if p["dataset"] == d and p["reference"] == "public_init") for d in DATASETS]
    axes[1].errorbar(list(positions), [p["mean_difference_pp"] for p in selected],
                     yerr=[p["sample_std_difference"] for p in selected], marker="o", color="#0072B2", capsize=4)
    axes[1].axhline(0, color="#777777", linewidth=1, linestyle="--")
    axes[1].set_ylabel("PDP minus public initialization (percentage points)")
    axes[1].set_title("Paired differences; no added utility established")
    for ax in axes:
        ax.set_xticks(list(positions), [d.title() for d in DATASETS], rotation=25)
        ax.grid(axis="y", alpha=0.25)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output/f"coverage_comparison.{suffix}", dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.output)
