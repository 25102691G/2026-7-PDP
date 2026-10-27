"""Conservative candidate-pool bound for the fixed positive-readout family.

Read-only public-graph analysis after confirmation, never a tuning procedure.
Bounds use all allowed readout weights, not the observed private parameter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("DGLBACKEND", "pytorch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.graph_util import get_rev_dgl
from pdp.data import load_prepared, trusted_snapshot
from pdp.positive_head import initialize_positive_head
from pdp.run import slot_configuration, training_configuration


def analyze(output: Path) -> None:
    if output.exists():
        raise FileExistsError("preserve existing analysis")
    torch.set_num_threads(1)
    records = []
    for run_id in ["positive-head-pilot-v3", "positive-head-confirmation-v1"]:
        root = ROOT/"outputs/pdp_runs"/run_id
        manifest = json.loads((root/"manifest.json").read_text())
        protocol = manifest["registered_protocol"]
        if not manifest.get("completed_at_utc") or protocol.get("benchmark_data_public") is not True:
            raise ValueError("requires completed public benchmarks")
        slots = slot_configuration(protocol, "pdp_slots")
        settings = slots.head_settings
        # Actual float32 clamp endpoints, represented exactly in float64.
        lower, upper = float(np.float32(settings.weight_lower)), float(np.float32(settings.weight_upper))
        for dataset in protocol["datasets"]:
            if trusted_snapshot(dataset, ROOT/"dataset/prepared") != manifest["identity"]["prepared_snapshots"][dataset]:
                raise ValueError("prepared data changed")
            graph = load_prepared(dataset, "test", ROOT/"dataset/prepared")
            message = get_rev_dgl(graph, "ones", 32, graph.is_directed(), False)
            for seed in protocol["model_seeds"]:
                config = training_configuration(protocol, dataset, "pdp_slots", seed)
                model = initialize_positive_head(config, settings)
                captured = []
                hook = model.grat.grat3.fc.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().double().numpy())
                )
                with torch.no_grad():
                    scores = model.logits(message, message.ndata["feat"]).reshape(-1)
                hook.remove()
                features = captured[0]
                if not np.isfinite(features).all() or (features < 0).any() or np.max(features) > 1e20:
                    raise ValueError("outside checked finite, nonnegative numerical diagnostic domain")
                selected = torch.argsort(scores, descending=True, stable=True)[:50].tolist()
                outside = [i for i in range(graph.vcount()) if i not in set(selected)]
                possible = np.zeros(len(outside), dtype=bool)
                universal_core = np.ones(len(selected), dtype=bool)
                for start in range(0, len(outside), 2048):
                    indices = np.asarray(outside[start:start+2048])
                    batch = features[indices]
                    for position, identifier in enumerate(selected):
                        difference = features[identifier]-batch
                        minimum = np.where(difference >= 0, lower*difference, upper*difference).sum(axis=1)
                        # Far larger than float64 arithmetic error and the
                        # standard 33-term float32 dot/add error envelope.
                        tolerance = 1e-4*(
                            upper*(np.abs(features[identifier]).sum()+np.abs(batch).sum(axis=1))
                            + 2*max(abs(settings.bias_lower), abs(settings.bias_upper))
                        )
                        fixed_tie = np.all(batch == features[identifier], axis=1) & (identifier < indices)
                        dominates = (minimum > tolerance) | fixed_tie
                        possible[start:start+len(indices)] |= ~dominates
                        universal_core[position] &= bool(dominates.all())
                candidates = set(selected) | {outside[i] for i in np.flatnonzero(possible)}
                covered = set(candidates)
                for identifier in candidates:
                    covered.update(graph.successors(identifier))
                baseline = next(c for c in manifest["cells"] if c["dataset"] == dataset
                                and c["method"] == "public_init" and c["model_seed"] == seed)
                denominator = baseline["metrics"]["celf_denominator"]
                records.append({
                    "dataset": dataset, "model_seed": seed,
                    "public_initialization_coverage": baseline["metrics"]["coverage_ratio_percent"],
                    "universally_dominating_selected_count": int(universal_core.sum()),
                    "possible_candidate_pool_size": len(candidates),
                    "candidate_pool_union_coverage_upper": 100*len(covered)/denominator,
                    "interpretation": "Conservative public-family diagnostic; not a privacy certificate or a guarantee of attainable coverage.",
                })
    output.mkdir(parents=True, exist_ok=False)
    (output/"bounds.json").write_text(json.dumps({
        "role": "post_confirmation_public_family_analysis_no_tuning",
        "numerical_assumptions": "finite nonnegative float32 features, standard IEEE arithmetic, conservative 1e-4 error envelope",
        "records": records,
    }, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output)
