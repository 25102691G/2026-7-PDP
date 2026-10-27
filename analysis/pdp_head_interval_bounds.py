"""Outward readout-family bounds; public read-only analysis, no training."""

from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/privim-pdp-mplconfig")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.graph_util import get_rev_dgl
from pdp.data import load_prepared, trusted_snapshot
from pdp.positive_head import initialize_positive_head
from pdp.run import slot_configuration, training_configuration


def down(value):
    return np.nextafter(value, -np.inf)


def up(value):
    return np.nextafter(value, np.inf)


def minimum_difference_lower(first, others, lower, upper):
    """Lower envelope of min_w w.(first-other), including every rounding."""
    difference = down(np.asarray(first, dtype=np.float64)-np.asarray(others, dtype=np.float64))
    terms = down(np.where(difference >= 0, lower, upper)*difference)
    total = np.zeros(terms.shape[:-1], dtype=np.float64)
    for index in range(terms.shape[-1]):
        total = down(total+terms[..., index])
    return total


def score_error_upper(features, upper_weight, maximum_bias):
    """Standard deterministic FP32 dot/add envelope; see the design note."""
    features = np.asarray(features, dtype=np.float64)
    dimension = features.shape[-1]
    operations = 2*dimension+2
    gamma = up(float(Fraction(operations, (1 << 24)-operations)))
    terms = up(np.abs(features)*upper_weight)
    magnitude = np.zeros(terms.shape[:-1], dtype=np.float64)
    for index in range(dimension):
        magnitude = up(magnitude+terms[..., index])
    magnitude = up(magnitude+maximum_bias)
    underflow = up(float(Fraction(2*operations, 1 << 126)))
    return up(up(gamma*magnitude)+underflow)


def candidate_pool_sets(features, selected, lower, upper, maximum_bias, chunk_size=2048):
    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(features).all() or np.max(np.abs(features)) > 1e20:
        raise ValueError("outside finite nonoverflow diagnostic domain")
    selected = list(selected)
    chosen = set(selected)
    outside = [i for i in range(len(features)) if i not in chosen]
    errors = score_error_upper(features, upper, maximum_bias)
    possible = np.zeros(len(outside), dtype=bool)
    core = np.ones(len(selected), dtype=bool)
    for start in range(0, len(outside), chunk_size):
        indices = np.asarray(outside[start:start+chunk_size], dtype=int)
        for position, identifier in enumerate(selected):
            minimum = minimum_difference_lower(features[identifier], features[indices], lower, upper)
            error = up(errors[identifier]+errors[indices])
            dominates = minimum > error
            possible[start:start+len(indices)] |= ~dominates
            core[position] &= bool(dominates.all())
    return (chosen | {outside[i] for i in np.flatnonzero(possible)},
            {selected[i] for i in np.flatnonzero(core)})


def candidate_pool(features, selected, lower, upper, maximum_bias, chunk_size=2048):
    pool, core = candidate_pool_sets(features, selected, lower, upper, maximum_bias, chunk_size)
    return pool, len(core)


def core_coverage_relaxation(graph, pool, core, k, maximum_combinations=1000000):
    """Exact max coverage over k-subsets containing core; ignores readout feasibility.

    When enumeration is too large, return the valid but weaker pool-union bound.
    Integer bitsets make all coverage/count computations exact.
    """
    pool, core = set(pool), set(core)
    if not core <= pool or not len(core) <= k <= len(pool):
        raise ValueError("invalid core/pool/cardinality")
    masks = {}
    for identifier in pool:
        raw = bytearray((graph.vcount()+7)//8)
        for node in [identifier, *graph.successors(identifier)]:
            raw[node//8] |= 1 << (node % 8)
        masks[identifier] = int.from_bytes(raw, "little")
    fixed = 0
    for identifier in core:
        fixed |= masks[identifier]
    remaining = sorted(pool-core)
    count = math.comb(len(remaining), k-len(core))
    if count > maximum_combinations:
        union = fixed
        for identifier in remaining:
            union |= masks[identifier]
        return union.bit_count(), count, False
    best = 0
    for selection in itertools.combinations(remaining, k-len(core)):
        union = fixed
        for identifier in selection:
            union |= masks[identifier]
        best = max(best, union.bit_count())
    return best, count, True


def analyze(output):
    if output.exists():
        raise FileExistsError("preserve prior analyses")
    torch.set_num_threads(1)
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("the analysis assumes standard FP32 products/additions")
    records = []
    for run_id in ["positive-head-pilot-v3", "positive-head-confirmation-v1"]:
        root = ROOT/"outputs/pdp_runs"/run_id
        manifest = json.loads((root/"manifest.json").read_text())
        protocol = manifest["registered_protocol"]
        if not manifest.get("completed_at_utc") or protocol.get("benchmark_data_public") is not True:
            raise ValueError("requires complete public benchmarks")
        slots = slot_configuration(protocol, "pdp_slots")
        settings = slots.head_settings
        lower = float(np.float32(settings.weight_lower))
        upper = float(np.float32(settings.weight_upper))
        for dataset in protocol["datasets"]:
            if trusted_snapshot(dataset, ROOT/"dataset/prepared") != manifest["identity"]["prepared_snapshots"][dataset]:
                raise ValueError("prepared inputs changed")
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
                selected = torch.argsort(scores, descending=True, stable=True)[:50].tolist()
                pool, core = candidate_pool_sets(captured[0], selected, lower, upper,
                                                 max(abs(settings.bias_lower), abs(settings.bias_upper)))
                covered = set(pool)
                for identifier in pool:
                    covered.update(graph.successors(identifier))
                reference = next(c for c in manifest["cells"] if c["dataset"] == dataset
                                 and c["method"] == "public_init" and c["model_seed"] == seed)
                maximum_coverage, combinations, enumerated = core_coverage_relaxation(graph, pool, core, 50)
                if maximum_coverage < reference["metrics"]["spread"]:
                    raise AssertionError("analysis excluded the observed public initialization")
                records.append({"dataset": dataset, "model_seed": seed,
                    "public_initialization_coverage": reference["metrics"]["coverage_ratio_percent"],
                    "possible_candidate_pool_size": len(pool), "universal_core_size": len(core),
                    "candidate_local_ids": sorted(pool), "universal_core_local_ids": sorted(core),
                    "candidate_pool_union_coverage_upper": 100*len(covered)/reference["metrics"]["celf_denominator"],
                    "core_constrained_coverage_upper": 100*maximum_coverage/reference["metrics"]["celf_denominator"],
                    "core_relaxation_combinations": combinations, "core_relaxation_exactly_enumerated": enumerated,
                    "topk_invariant_certified": len(pool) == 50})
    output.mkdir(parents=True, exist_ok=False)
    (output/"bounds.json").write_text(json.dumps({
        "role": "post_confirmation_read_only_analysis_no_tuning",
        "arithmetic_model": "standard FP32 dot/add, outward float64 envelopes, gamma_(2d+2), no reduced-precision replacement or overflow",
        "scope": "fixed encoder and readout box only; neither privacy certificate nor impossibility theorem for other models",
        "records": records,
    }, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output)
