"""Read-only diagnostics for completed public-benchmark checkpoints.

These raw graph-dependent analyses are not covered by model DP certificates.
The script refuses runs not explicitly registered as public benchmarks.
"""

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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from experiments.graph_util import get_rev_dgl
from pdp.data import load_prepared, trusted_snapshot
from pdp.run import training_configuration
from pdp.training import initialize_model


def correlation(first, second):
    first = first-first.mean()
    second = second-second.mean()
    divisor = torch.linalg.vector_norm(first)*torch.linalg.vector_norm(second)
    return float(torch.dot(first, second)/divisor) if divisor > 0 else None


def analyze(run_id: str, output: Path) -> None:
    root = ROOT/"outputs/pdp_runs"/run_id
    manifest = json.loads((root/"manifest.json").read_text())
    protocol = manifest["registered_protocol"]
    if protocol.get("benchmark_data_public") is not True:
        raise ValueError("raw checkpoint diagnostics require an explicitly public benchmark")
    if not manifest.get("completed_at_utc"):
        raise ValueError("analyze all completed trials together, not a selected partial run")
    if output.exists():
        raise FileExistsError("preserve existing analysis; do not overwrite")
    torch.set_num_threads(1)
    graphs = {}
    for dataset in protocol["datasets"]:
        if trusted_snapshot(dataset, ROOT/"dataset/prepared") != manifest["identity"]["prepared_snapshots"][dataset]:
            raise ValueError("prepared files changed")
        graph = load_prepared(dataset, "test", ROOT/"dataset/prepared")
        graphs[dataset] = get_rev_dgl(graph, "ones", 32, graph.is_directed(), False)
    rows = []
    for cell in manifest["cells"]:
        if cell["status"] != "completed":
            raise ValueError("never drop a failed or incomplete seed from diagnostics")
        config = training_configuration(protocol, cell["dataset"], cell["method"], cell["model_seed"])
        path = root/"trusted/models"/cell["id"]/"parameters.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != cell["artifact_sha256"]["parameters.pt"]:
            raise ValueError("checkpoint checksum mismatch")
        model = initialize_model(config)
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        captured = []
        hook = model.grat.grat3.fc.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().double())
        )
        graph = graphs[cell["dataset"]]
        with torch.no_grad():
            scores = model.logits(graph, graph.ndata["feat"]).reshape(-1).double()
        hook.remove()
        features = captured[0]
        centered = features-features.mean(dim=0)
        singular = torch.linalg.svdvals(centered)
        energy = singular.square().sum()
        displacement = sum(
            float((value.detach().double()-initial[key].double()).square().sum())
            for key, value in model.state_dict().items()
        )**0.5
        rows.append({
            "dataset": cell["dataset"], "method": cell["method"],
            "model_seed": cell["model_seed"],
            "coverage_percent": cell["metrics"]["coverage_ratio_percent"],
            "parameter_displacement_l2": displacement,
            "head_score_vs_mean_activation_correlation": correlation(scores, features.mean(dim=1)),
            "head_feature_centered_pc1_energy_fraction": float(singular[0]**2/energy) if energy else None,
            "score_distinct_values": len(torch.unique(scores)),
            "score_finite_fraction": float(torch.isfinite(scores).double().mean()),
        })
    output.mkdir(parents=True, exist_ok=False)
    with (output/"checkpoints.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = []
    for dataset in protocol["datasets"]:
        for method in protocol["methods"]:
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            correlations = [r["head_score_vs_mean_activation_correlation"] for r in selected]
            pc1 = [r["head_feature_centered_pc1_energy_fraction"] for r in selected]
            grouped.append({
                "dataset": dataset, "method": method, "runs": len(selected),
                "coverage_mean": statistics.mean(r["coverage_percent"] for r in selected),
                "constant_score_runs": sum(r["score_distinct_values"] == 1 for r in selected),
                "negative_activation_correlation_runs": sum(c is not None and c < 0 for c in correlations),
                "absolute_activation_correlations": [abs(c) if c is not None else None for c in correlations],
                "centered_pc1_energy_fractions": pc1,
            })
    (output/"summary.json").write_text(json.dumps({
        "source_run": run_id, "role": "raw_public_benchmark_diagnostics_not_a_dp_release",
        "groups": grouped,
    }, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_id, args.output)
