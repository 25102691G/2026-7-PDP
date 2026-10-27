"""Pre-registered public-benchmark runner; all raw artifacts stay trusted."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import time
from typing import Any, Dict, List

import torch
import dgl
import igraph
import numpy

from .accounting import RangePrivacy
from .baselines import train_non_private
from .data import load_prepared, trusted_snapshot
from .data import DATASETS
from .evaluation import celf_denominator, evaluate_release
from .personalization import HashBudgetPolicy, NodeBudget, train_pdp
from .training import ParameterRelease, TrainingConfig, save_release, train_b0
from .stable_slots import StableSlots, slot_accounting, train_slots
from .positive_head import positive_initialization_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLOT_METHODS = {"uniform_dp", "uniform_full", "pdp_slots", "slots_zero_noise"}


def _uses_slots(protocol: Dict[str, Any], method: str) -> bool:
    return ("stable_slots" in protocol and
            method in protocol["stable_slots"]["appearance_limits_by_method"])


def slot_configuration(protocol: Dict[str, Any], method: str) -> StableSlots:
    values = dict(protocol["stable_slots"])
    limits = values.pop("appearance_limits_by_method")[method]
    uniform = method in values.pop("uniform_methods")
    return StableSlots(
        appearance_limits=tuple(limits),
        uniform_epsilon=protocol["base_epsilon"][method] if uniform else None,
        uniform_delta=protocol["delta_0"] if uniform else None,
        **values,
    )


def _write_json(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(
        json.dumps(content, sort_keys=True, indent=2)+"\n", encoding="utf-8",
    )
    temporary.replace(path)


def implementation_digest() -> str:
    sources = list((PROJECT_ROOT/"pdp").glob("*.py"))
    sources.extend(PROJECT_ROOT/name for name in [
        "experiments/sampling.py", "experiments/graph_util.py",
        "model/GRAT.py", "model/layer/GRATLayer.py",
        "model/functional/MaxCoverLoss.py", "baselines/heuristics.py",
    ])
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode()+b"\0")
        digest.update(path.read_bytes()+b"\0")
    return digest.hexdigest()


def load_protocol(path: Path) -> Dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != "pdp-preregistration-v1":
        raise ValueError("unknown pre-registration schema")
    if protocol.get("benchmark_data_public") is not True:
        raise ValueError("runner is only for explicitly public benchmarks")
    evaluation = protocol["evaluation"]
    if (evaluation["diffusion_model"], evaluation["edge_probability"],
            evaluation["diffusion_hops"]) != ("IC", 1, 1):
        raise ValueError("this runner implements the pre-registered one-hop IC protocol")
    if protocol["tuning"].get("test_seed_specific_tuning") is not False:
        raise ValueError("test-seed tuning is prohibited")
    for name in ["datasets", "methods", "model_seeds"]:
        values = protocol[name]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and contain no duplicates")
    if not set(protocol["datasets"]) <= set(DATASETS):
        raise ValueError("unknown dataset in protocol")
    methods = set(protocol["methods"])
    allowed_methods = {
        "public_init", "non_private", "clipped_no_noise",
        "uniform_dp", "pdp_base2", "pdp_base4",
    }
    if "stable_slots" in protocol:
        allowed_methods |= SLOT_METHODS
    if not methods <= allowed_methods:
        raise ValueError("unknown pre-registered method")
    budgets = [NodeBudget(**group) for group in protocol["node_budgets"]]
    if not budgets:
        raise ValueError("node budget policy must be nonempty")
    for uniform_method in methods & {"uniform_dp", "uniform_full"}:
        from decimal import Decimal
        epsilon = Decimal(protocol["base_epsilon"][uniform_method])
        delta = Decimal(protocol["delta_0"])
        if any(Decimal(str(group.epsilon)) < epsilon or
               Decimal(str(group.delta)) < delta for group in budgets):
            raise ValueError("uniform DP control must satisfy every node budget")
    if "stable_slots" in protocol:
        policy = HashBudgetPolicy(groups=tuple(budgets), salt=protocol["budget_policy_salt"])
        profiles = set(protocol["stable_slots"]["appearance_limits_by_method"])
        if not profiles <= SLOT_METHODS | {"non_private"}:
            raise ValueError("unknown stable-slot profile or data-dependent initialization control")
        if not (methods & SLOT_METHODS) <= profiles:
            raise ValueError("stable-slot methods require explicit appearance profiles")
        for dataset in protocol["datasets"]:
            for method in methods & profiles:
                slot_accounting(
                    training_configuration(protocol, dataset, method, protocol["model_seeds"][0]),
                    policy, slot_configuration(protocol, method),
                )
    return protocol


def training_configuration(protocol: Dict[str, Any], dataset: str,
                           method: str, seed: int) -> TrainingConfig:
    values = dict(protocol["training"])
    private_steps = values.pop("private_steps")
    nonprivate_steps = values.pop("non_private_steps")
    batch = values.pop("batch_size")
    clip = values.pop("clip_norm")
    grid = values.pop("grid_bits")
    size = values.pop("subgraph_size_by_dataset")[dataset]
    epsilon = protocol["base_epsilon"].get(method, "1")
    steps = nonprivate_steps if method == "non_private" else private_steps
    return TrainingConfig(
        privacy=RangePrivacy(
            epsilon=epsilon, delta=protocol["delta_0"], clip_norm=clip,
            batch_size=batch, steps=steps, grid_bits=grid,
        ),
        subgraph_size=size, model_seed=seed, **values,
    )


def planned_cells(protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for dataset in protocol["datasets"]:
        for method in protocol["methods"]:
            for seed in protocol["model_seeds"]:
                config = training_configuration(protocol, dataset, method, seed)
                result.append({
                    "id": f"{dataset}_{method}_seed{seed}",
                    "dataset": dataset, "method": method, "model_seed": seed,
                    "optimizer_steps": 0 if method == "public_init" else config.privacy.steps,
                    "status": "pending",
                })
    return result


def _reuse_public_control(protocol: Dict[str, Any], cell: Dict[str, Any],
                         config: TrainingConfig, identity: Dict[str, Any],
                         output_root: Path, directory: Path) -> ParameterRelease:
    """Reuse predeclared completed public controls without another training call."""
    if cell["method"] not in {"public_init", "non_private"}:
        raise ValueError("only explicitly non-private/public initialization controls may be reused")
    if (_uses_slots(protocol, cell["method"]) or
            protocol.get("stable_slots", {}).get("trainable_mode", "full") != "full"):
        raise ValueError("different encoder or slot-training control cannot reuse legacy controls")
    run_id = protocol["reuse_controls_from"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid reference run ID")
    root = Path(output_root)/run_id
    reference = json.loads((root/"manifest.json").read_text())
    if (not reference.get("completed_at_utc")
            or reference["registered_protocol"].get("benchmark_data_public") is not True):
        raise ValueError("reference must be a completed public benchmark")
    dataset = cell["dataset"]
    if (reference["identity"]["runtime"] != identity["runtime"] or
            reference["identity"]["prepared_snapshots"][dataset] !=
            identity["prepared_snapshots"][dataset]):
        raise ValueError("control reuse requires the same runtime and prepared split")
    # Unchanged executed baseline dependencies; other experimental modules
    # can evolve without forcing a pointless repeat of the same control.
    for name in ["training.py", "baselines.py", "evaluation.py", "data.py",
                 "accounting.py", "lattice.py"]:
        if (root/"trusted"/"source"/name).read_bytes() != (PROJECT_ROOT/"pdp"/name).read_bytes():
            raise ValueError("executed baseline source differs; control cannot be reused")
    candidates = [old for old in reference["cells"] if old["id"] == cell["id"]]
    if len(candidates) != 1 or candidates[0]["status"] != "completed":
        raise ValueError("reference control is not uniquely completed")
    old = candidates[0]
    source = root/"trusted"/"models"/cell["id"]
    for name in ["parameters.pt", "certificate.json"]:
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != old["artifact_sha256"][name]:
            raise ValueError("reference control checksum mismatch")
    certificate = json.loads((source/"certificate.json").read_text())
    expected = asdict(config)
    expected.pop("privacy")
    expected.update({
        "steps": 0 if cell["method"] == "public_init" else config.privacy.steps,
        "batch_size": config.privacy.batch_size, "clip_norm": None, "grid_bits": None,
    })
    if certificate.get("public_training_configuration") != expected:
        raise ValueError("reference control configuration differs")
    if certificate.get("privacy_guarantee") not in {"none", "data_independent"}:
        raise ValueError("a private mechanism is not an eligible reusable control")
    shutil.copytree(source, directory)
    cell["reused_from"] = {
        "run_id": run_id, "cell_id": old["id"],
        "implementation_sha256": reference["identity"]["implementation_sha256"],
        "new_training_calls": 0,
    }
    return ParameterRelease(torch.load(directory/"parameters.pt", weights_only=True), certificate)


def write_summaries(root: Path, manifest: Dict[str, Any]) -> None:
    records = []
    grouped = {}
    for cell in manifest["cells"]:
        row = {
            key: cell[key] for key in [
                "dataset", "method", "model_seed", "optimizer_steps", "status",
            ]
        }
        row.update(cell.get("metrics", {}))
        records.append(row)
        grouped.setdefault((cell["dataset"], cell["method"]), []).append(cell)
    fieldnames = [
        "dataset", "method", "model_seed", "optimizer_steps", "status",
        "diffusion_model", "edge_probability", "diffusion_hops", "k",
        "spread", "celf_denominator", "coverage_ratio_percent",
        "ranking_finite_fraction", "seed_local_ids",
    ]
    with (root/"records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    for (dataset, method), cells in grouped.items():
        complete = all(cell["status"] == "completed" for cell in cells)
        values = [cell["metrics"]["coverage_ratio_percent"] for cell in cells
                  if cell["status"] == "completed"]
        summaries.append({
            "dataset": dataset, "method": method,
            "expected_runs": len(cells), "completed_runs": len(values),
            "all_runs_complete": complete,
            # No successful-seed-only mean for an incomplete method.
            "coverage_mean": statistics.mean(values) if complete else None,
            "coverage_sample_std": (
                statistics.stdev(values) if complete and len(values) > 1 else None
            ),
        })
    _write_json(root/"summary.json", summaries)
    paired = []
    for dataset in sorted({key[0] for key in grouped}):
        for reference in ["uniform_dp", "public_init", "non_private"]:
            reference_cells = grouped.get((dataset, reference), [])
            if not reference_cells:
                continue
            reference_scores = {
                cell["model_seed"]: cell["metrics"]["coverage_ratio_percent"]
                for cell in reference_cells if cell["status"] == "completed"
            }
            for (candidate_dataset, method), cells in grouped.items():
                if candidate_dataset != dataset or method == reference:
                    continue
                complete = (
                    all(cell["status"] == "completed" for cell in cells)
                    and len(reference_scores) == len(reference_cells)
                    and {cell["model_seed"] for cell in cells} == set(reference_scores)
                )
                differences = [
                    cell["metrics"]["coverage_ratio_percent"]-reference_scores[cell["model_seed"]]
                    for cell in cells
                ] if complete else []
                paired.append({
                    "dataset": dataset, "method": method, "reference": reference,
                    "all_pairs_complete": complete,
                    "mean_difference_percentage_points": (
                        statistics.mean(differences) if complete else None
                    ),
                    "sample_std_difference": (
                        statistics.stdev(differences) if complete and len(differences)>1 else None
                    ),
                    "paired_differences": differences,
                })
    _write_json(root/"paired_comparisons.json", paired)


def run_protocol(protocol_path: Path, run_id: str, *, resume: bool = False,
                 output_root: Path = PROJECT_ROOT/"outputs"/"pdp_runs",
                 prepared_root: Path = PROJECT_ROOT/"dataset"/"prepared") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, hyphens and underscores")
    protocol = load_protocol(protocol_path)
    policy = HashBudgetPolicy(
        groups=tuple(NodeBudget(**group) for group in protocol["node_budgets"]),
        salt=protocol["budget_policy_salt"],
    )
    root = Path(output_root)/run_id
    path = root/"manifest.json"
    identity = {
        "protocol_sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
        "implementation_sha256": implementation_digest(),
        "runtime": {
            "torch": torch.__version__, "dgl": dgl.__version__,
            "igraph": igraph.__version__, "numpy": numpy.__version__,
            "device": "cpu",
        },
        "prepared_snapshots": {
            dataset: trusted_snapshot(dataset, prepared_root)
            for dataset in protocol["datasets"]
        },
    }
    if root.exists():
        if not resume:
            raise FileExistsError("run already exists; use resume or a new run_id")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("resume requires identical protocol, source and prepared files")
    else:
        if resume:
            raise FileNotFoundError("cannot resume a missing run")
        root.mkdir(parents=True, exist_ok=False)
        (root/"trusted").mkdir()
        (root/"trusted"/"models").mkdir()
        source_directory = root/"trusted"/"source"
        source_directory.mkdir()
        for source in (PROJECT_ROOT/"pdp").glob("*.py"):
            shutil.copyfile(source, source_directory/source.name)
        proof_directory = root/"trusted"/"proofs"
        proof_directory.mkdir()
        for name in ["b0_integer_privacy.md", "pdp_stable_slots_design.md", "pdp_positive_head_design.md",
                     "p0_pdp_feasibility.md", "pdp_privacy_design.md"]:
            shutil.copyfile(PROJECT_ROOT/"docs"/name, proof_directory/name)
        shutil.copyfile(protocol_path, root/"protocol.source.json")
        manifest = {
            "run_id": run_id, "identity": identity,
            "registered_protocol": protocol,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_boundary": "trusted_public_benchmark_research",
            "privacy_scope": "one_model_invocation; no joint per-model-budget claim",
            "cells": planned_cells(protocol),
        }
        _write_json(root/"preregistration.json", protocol)
        _write_json(path, manifest)
    torch.set_num_threads(1)
    graphs = {}
    for dataset in protocol["datasets"]:
        train_graph = load_prepared(dataset, "train", prepared_root)
        test_graph = load_prepared(dataset, "test", prepared_root)
        if trusted_snapshot(dataset, prepared_root) != identity["prepared_snapshots"][dataset]:
            raise ValueError("prepared files changed while loading the shared graphs")
        denominator = celf_denominator(test_graph, protocol["evaluation"]["k"])
        graphs[dataset] = (train_graph, test_graph, denominator)
    for cell in manifest["cells"]:
        if cell["status"] == "completed":
            continue
        if cell["status"] == "failed":
            raise RuntimeError("failed cell retained; investigate and register a new run")
        dataset, method, seed = cell["dataset"], cell["method"], cell["model_seed"]
        config = training_configuration(protocol, dataset, method, seed)
        train_graph, test_graph, denominator = graphs[dataset]
        directory = root/"trusted"/"models"/cell["id"]
        cell["status"] = "running"
        _write_json(path, manifest)
        try:
            if directory.exists():
                # Recovery after a completed private call: never retrain it.
                if not all((directory/name).is_file()
                           for name in ["certificate.json", "parameters.pt"]):
                    raise RuntimeError("partial model artifact; retained without overwrite")
                release = ParameterRelease(
                    torch.load(directory/"parameters.pt", weights_only=True),
                    json.loads((directory/"certificate.json").read_text()),
                )
                for filename in ["parameters.pt", "certificate.json"]:
                    observed = hashlib.sha256((directory/filename).read_bytes()).hexdigest()
                    recorded = cell.get("artifact_sha256", {}).get(filename)
                    if recorded is not None and observed != recorded:
                        raise ValueError("existing artifact checksum differs; refusing retraining")
                if _uses_slots(protocol, method):
                    _, expected = slot_accounting(config, policy, slot_configuration(protocol, method))
                    if slot_configuration(protocol, method).trainable_mode == "positive_head":
                        expected["proof"] = "docs/pdp_positive_head_design.md"
                    if method in {"slots_zero_noise", "non_private"}:
                        expected = {
                            "schema": "nonprivate-stable-slot-control-v1", "privacy_guarantee": "none",
                            "baseline": "stable_slots_zero_noise",
                            "public_training_configuration": expected["public_training_configuration"],
                            "public_slot_configuration": expected["public_slot_configuration"],
                        }
                    if any(release.certificate.get(key) != value for key, value in expected.items()):
                        raise ValueError("existing stable-slot certificate differs")
                elif method in {"uniform_dp", "pdp_base2", "pdp_base4"}:
                    if release.certificate.get("public_training_configuration") != asdict(config):
                        raise ValueError("existing private model has a different configuration")
                    expected = config.privacy.certificate(config.privacy.calibrate())
                    if any(release.certificate.get(key) != value
                           for key, value in expected.items()):
                        raise ValueError("existing private certificate differs")
                    if method != "uniform_dp" and release.certificate.get("personalization") != (
                        policy.certificate(config.privacy.epsilon, config.privacy.delta)
                    ):
                        raise ValueError("existing PDP policy differs")
            else:
                diagnostic = {}
                started = time.monotonic()
                if (protocol.get("reuse_controls_from")
                        and method in {"public_init", "non_private"}):
                    release = _reuse_public_control(
                        protocol, cell, config, identity, output_root, directory,
                    )
                    diagnostic["reused_from"] = cell["reused_from"]
                elif _uses_slots(protocol, method):
                    release = train_slots(
                        train_graph, config, policy, slot_configuration(protocol, method),
                        trusted_diagnostics=diagnostic,
                        noise_enabled=method not in {"slots_zero_noise", "non_private"},
                    )
                elif (method == "public_init" and "stable_slots" in protocol and
                      protocol["stable_slots"].get("trainable_mode") == "positive_head"):
                    slots = slot_configuration(protocol, next(iter(
                        protocol["stable_slots"]["appearance_limits_by_method"]
                    )))
                    release = positive_initialization_release(config, slots.head_settings)
                elif method in {"public_init", "non_private", "clipped_no_noise"}:
                    release = train_non_private(
                        train_graph, config,
                        clipped=method == "clipped_no_noise",
                        initialization_only=method == "public_init",
                        trusted_diagnostics=diagnostic,
                    )
                elif method == "uniform_dp":
                    release = train_b0(train_graph, config, trusted_diagnostics=diagnostic)
                else:
                    release = train_pdp(train_graph, config, policy,
                                        trusted_diagnostics=diagnostic)
                diagnostic["training_seconds"] = time.monotonic()-started
                if not directory.exists():
                    save_release(release, directory)
                _write_json(root/"trusted"/(cell["id"]+"_diagnostics.json"), diagnostic)
            cell["artifact_sha256"] = {
                filename: hashlib.sha256((directory/filename).read_bytes()).hexdigest()
                for filename in ["parameters.pt", "certificate.json"]
            }
            cell["metrics"] = evaluate_release(
                release, config, test_graph, k=protocol["evaluation"]["k"],
                denominator=denominator,
            )
            cell["status"] = "completed"
            print(
                f"{dataset} {method} model_seed={seed}: "
                f"coverage={cell['metrics']['coverage_ratio_percent']:.4f}%",
                flush=True,
            )
        except Exception as error:
            cell["status"] = "failed"
            cell["failure_type"] = type(error).__name__
            _write_json(path, manifest)
            write_summaries(root, manifest)
            raise
        _write_json(path, manifest)
        write_summaries(root, manifest)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, manifest)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=PROJECT_ROOT/"configs"/"pdp"/"pilot-v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.dry_run:
        print(json.dumps(planned_cells(protocol), indent=2))
        return
    run_protocol(args.protocol, args.run_id, resume=args.resume)


if __name__ == "__main__":
    main()
