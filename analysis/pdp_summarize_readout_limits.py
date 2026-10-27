"""Verify saved core relaxations and compare exact rational macro bounds."""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/privim-pdp-mplconfig")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.pdp_head_interval_bounds import core_coverage_relaxation
from pdp.data import load_prepared, trusted_snapshot


def encoded(value):
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal_approximation": float(value)}


def main():
    source = ROOT/"outputs/pdp_analyses/positive-head-core-relaxation/bounds.json"
    records = json.loads(source.read_text())["records"]
    manifests = [json.loads((ROOT/"outputs/pdp_runs"/name/"manifest.json").read_text())
                 for name in ["positive-head-pilot-v3", "positive-head-confirmation-v1"]]
    datasets = [d for m in manifests for d in m["registered_protocol"]["datasets"]]
    groups = []
    for dataset in datasets:
        manifest = next(m for m in manifests if dataset in m["registered_protocol"]["datasets"])
        if trusted_snapshot(dataset, ROOT/"dataset/prepared") != manifest["identity"]["prepared_snapshots"][dataset]:
            raise ValueError("prepared inputs changed")
        graph = load_prepared(dataset, "test", ROOT/"dataset/prepared")
        selected_records = sorted([r for r in records if r["dataset"] == dataset], key=lambda r:r["model_seed"])
        if [r["model_seed"] for r in selected_records] != [0, 1, 2, 3, 4]:
            raise ValueError("all five seeds are required")
        upper_values, actual_values, uniform_values = [], [], []
        for record in selected_records:
            maximum, count, exact = core_coverage_relaxation(
                graph, record["candidate_local_ids"], record["universal_core_local_ids"], 50,
            )
            if not exact or count != record["core_relaxation_combinations"]:
                raise ValueError("relaxation not exactly enumerated")
            cell = next(c for c in manifest["cells"] if c["dataset"] == dataset
                        and c["method"] == "pdp_slots" and c["model_seed"] == record["model_seed"])
            uniform = next(c for c in manifest["cells"] if c["dataset"] == dataset
                           and c["method"] == "uniform_dp" and c["model_seed"] == record["model_seed"])
            denominator = cell["metrics"]["celf_denominator"]
            upper = Fraction(100*maximum, denominator)
            if abs(float(upper)-record["core_constrained_coverage_upper"]) > 1e-10:
                raise ValueError("saved numeric upper differs from integer recomputation")
            upper_values.append(upper)
            actual_values.append(Fraction(100*cell["metrics"]["spread"], denominator))
            uniform_values.append(Fraction(100*uniform["metrics"]["spread"], denominator))
        upper = sum(upper_values, Fraction(0))/5
        actual = sum(actual_values, Fraction(0))/5
        uniform = sum(uniform_values, Fraction(0))/5
        groups.append({"dataset": dataset, "upper": encoded(upper), "actual_pdp": encoded(actual),
                       "uniform_low": encoded(uniform), "maximum_remaining_gain_pp": encoded(upper-actual),
                       "per_seed_maximum_coverage_ratio": [encoded(x) for x in upper_values]})
    def fraction(group, key):
        item = group[key]
        return Fraction(item["numerator"], item["denominator"])
    def aggregate(selected):
        upper = sum((fraction(g, "upper") for g in selected), Fraction(0))/len(selected)
        actual = sum((fraction(g, "actual_pdp") for g in selected), Fraction(0))/len(selected)
        uniform = sum((fraction(g, "uniform_low") for g in selected), Fraction(0))/len(selected)
        return {"upper": encoded(upper), "actual_pdp": encoded(actual),
                "maximum_gain_over_current_pdp_pp": encoded(upper-actual),
                "maximum_gain_over_uniform_low_pp": encoded(upper-uniform),
                "one_percentage_point_gain_over_current_pdp_possible": upper-actual >= 1}
    content = {
        "role": "utility_family_upper_bound_not_a_privacy_certificate",
        "proof": "docs/pdp_readout_interval_design.md",
        "scope": "same fixed positive encoders, same five public model seeds, same readout box, same public target graphs and standard FP32 arithmetic model",
        "caveat": "The combinatorial optimum can be unattainable by any linear readout. No impossibility claim about other model families.",
        "groups": groups, "six_dataset_macro": aggregate(groups),
        "development_macro": aggregate([g for g in groups if g["dataset"] in {"email", "bitcoin"}]),
    }
    path = source.parent/"exact_macro_certificate.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(content, indent=2)+"\n")
    print(json.dumps({key: content[key] for key in ["six_dataset_macro", "development_macro"]}, indent=2))


if __name__ == "__main__":
    main()
