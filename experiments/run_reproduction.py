#!/usr/bin/env python3
"""Run a manifest-scoped PrivIM reproduction matrix."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import (
        DATASETS,
        EPSILONS,
        PROTOCOL_PROFILES,
        RELEASED_COMPATIBILITY_PROTOCOL,
        released_compatibility_overrides,
    )
    from .privacy import (
        ACCOUNTANTS,
        DP_NOISE_STREAMS,
        LEGACY_GLOBAL_DP_NOISE_STREAM,
        PAPER_BINOMIAL,
    )
    from .seeding import SAMPLING_SEED_MODES, SEED_MODES, seed_protocol
except ImportError:  # Direct execution from the repository root.
    from config import (
        DATASETS,
        EPSILONS,
        PROTOCOL_PROFILES,
        RELEASED_COMPATIBILITY_PROTOCOL,
        released_compatibility_overrides,
    )
    from privacy import (
        ACCOUNTANTS,
        DP_NOISE_STREAMS,
        LEGACY_GLOBAL_DP_NOISE_STREAM,
        PAPER_BINOMIAL,
    )
    from seeding import SAMPLING_SEED_MODES, SEED_MODES, seed_protocol
from model.GRAT import GRAT_VARIANTS
from experiments.graph_util import FEATURE_PROTOCOLS
from experiments.sampling import (
    PAPER_SAMPLING_PROTOCOL,
    RELEASED_LEGACY_SAMPLING_PROTOCOL,
    SAMPLING_PROTOCOLS,
)


PREPARED_ROOT = PROJECT_ROOT / "dataset" / "prepared"


def subprocess_environment(gpu_id):
    """Return an isolated child environment pinned to one physical GPU."""
    environment = os.environ.copy()
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return environment


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_state():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def validate_prepared(
    dataset, isolated_policy, remove_self_loops, split_source="generated"
):
    path = PREPARED_ROOT / dataset / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prepared metadata for {dataset}: {path}"
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    actual_source = metadata.get("split_source", "generated")
    if actual_source != split_source:
        raise ValueError(
            f"{dataset}: prepared split_source={actual_source!r}, expected "
            f"{split_source!r}; rerun without --skip-prepare"
        )
    expected_policy = (
        "artifact_edge_list" if split_source == "paper-artifact"
        else isolated_policy
    )
    actual_policy = metadata.get("isolated_policy", "legacy-preserve")
    if actual_policy != expected_policy:
        raise ValueError(
            f"{dataset}: prepared isolated_policy={actual_policy!r}, expected "
            f"{expected_policy!r}; rerun without --skip-prepare"
        )
    if bool(metadata.get("remove_self_loops")) != remove_self_loops:
        raise ValueError(
            f"{dataset}: prepared remove_self_loops="
            f"{metadata.get('remove_self_loops')!r}, expected "
            f"{remove_self_loops!r}; rerun without --skip-prepare"
        )
    if float(metadata.get("train_ratio", -1)) != 0.5:
        raise ValueError(f"{dataset}: paper reproduction requires train_ratio=0.5")
    return metadata


def completed_models(
    run_root, dataset, setting, epsilon_dir, iterations,
    learning_rate=None, lr_decay=None, lr_decay_steps=None, model_variant=None,
    expected_seed_protocol=None,
    model_initialization=None,
    non_private_clipping=None,
    feature_protocol=None,
    protocol_profile=None,
    sampling_protocol=None,
    optimizer_update_cap=None,
    dp_noise_stream=None,
    parameter_averaging=None,
    ema_decay=None,
    batch_size=None,
    optimizer_momentum=None,
    loss_penalty=None,
):
    directory = run_root / dataset / setting / epsilon_dir
    if not directory.exists():
        return 0
    completed = 0
    for checkpoint in directory.rglob("model_*.pt"):
        metadata_path = checkpoint.with_suffix(".json")
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        configured_units = metadata.get(
            "configured_training_units", metadata.get("iterations", -1)
        )
        if int(configured_units) != iterations:
            continue
        if learning_rate is not None and float(
            metadata.get("learning_rate", -1)
        ) != learning_rate:
            continue
        schedule = metadata.get("learning_rate_schedule", {})
        if lr_decay is not None and float(schedule.get("gamma", -1)) != lr_decay:
            continue
        if lr_decay_steps is not None and int(
            schedule.get("step_size_updates", -1)
        ) != lr_decay_steps:
            continue
        if model_variant is not None and metadata.get("model_variant") != model_variant:
            continue
        if (
            model_initialization is not None
            and metadata.get("model_initialization") != model_initialization
        ):
            continue
        if (
            non_private_clipping is not None
            and bool(metadata.get("non_private_clipping", False))
            != non_private_clipping
        ):
            continue
        if (
            feature_protocol is not None
            and metadata.get("feature_protocol", "ones") != feature_protocol
        ):
            continue
        if (
            protocol_profile is not None
            and metadata.get("protocol_profile", "strict") != protocol_profile
        ):
            continue
        if (
            sampling_protocol is not None
            and metadata.get("sampling_protocol", "paper") != sampling_protocol
        ):
            continue
        if metadata.get("optimizer_update_cap") != optimizer_update_cap:
            continue
        if (
            dp_noise_stream is not None
            and metadata.get("dp_noise_stream", "independent")
            != dp_noise_stream
        ):
            continue
        if (
            parameter_averaging is not None
            and metadata.get("parameter_averaging", "none")
            != parameter_averaging
        ):
            continue
        if parameter_averaging == "ema" and float(
            metadata.get("ema_decay", -1)
        ) != ema_decay:
            continue
        if batch_size is not None and int(
            metadata.get("batch_size", -1)
        ) != batch_size:
            continue
        if optimizer_momentum is not None and float(
            metadata.get("momentum", 0.0)
        ) != optimizer_momentum:
            continue
        if loss_penalty is not None and float(
            metadata.get("loss_lambda", 1.0)
        ) != loss_penalty:
            continue
        if (
            expected_seed_protocol is not None
            and metadata.get("experiment", {}).get("seed_protocol")
            != expected_seed_protocol
        ):
            continue
        completed += 1
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-profile", choices=PROTOCOL_PROFILES, default="strict",
        help=(
            "strict is the paper-oriented unbounded node-DP protocol; "
            "released-compatibility reproduces audited revision 31166ca "
            "behavior and must not be reported as strict epsilon"
        ),
    )
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--epsilon", nargs="+", type=int, choices=EPSILONS, default=list(EPSILONS))
    parser.add_argument(
        "--settings", nargs="+",
        choices=("PrivIM", "PrivIM+SCS", "PrivIM+", "PrivIM*"),
        default=["PrivIM", "PrivIM+SCS", "PrivIM+"],
    )
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--model-seed-mode", choices=SEED_MODES, default="per-round"
    )
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--sampling-seed-mode", choices=SAMPLING_SEED_MODES, default="fixed",
        help="fixed reuses one subgraph container across model rounds",
    )
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument(
        "--batch-seed-mode", choices=SEED_MODES, default="per-round"
    )
    parser.add_argument("--dp-noise-seed", type=int, default=0)
    parser.add_argument(
        "--dp-noise-seed-mode", choices=SEED_MODES, default="per-round",
        help="hold Gaussian DP noise fixed or vary it independently by round",
    )
    parser.add_argument(
        "--dp-noise-stream", choices=DP_NOISE_STREAMS, default="independent",
        help=(
            "independent uses a dedicated seeded generator; legacy-global "
            "reproduces revision 31166ca and requires released-compatibility"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help=(
            "optimizer updates per run; the paper does not publish T, and "
            "500 is the smallest validated default that removes the 50-step "
            "underfitting observed on Email"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument(
        "--optimizer-momentum", type=float, default=0.0,
        help=(
            "SGD momentum applied after private gradient construction; "
            "strict default is 0 and released-compatibility forces 0.9"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="subgraphs per private optimizer update",
    )
    parser.add_argument(
        "--parameter-averaging", choices=("none", "ema"), default="none",
        help="save EMA parameters as a strict-DP post-processing diagnostic",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--lr-decay", type=float, default=1.0,
        help="StepLR multiplier; the paper reproduction keeps 0.005 fixed",
    )
    parser.add_argument("--lr-decay-steps", type=int, default=5)
    parser.add_argument(
        "--loss-penalty", type=float, default=1.0,
        help=(
            "lambda in paper Equation (5); 1.0 preserves released behavior"
        ),
    )
    parser.add_argument(
        "--grat-variant",
        choices=GRAT_VARIANTS,
        default="legacy_released",
    )
    parser.add_argument(
        "--grat-initialization",
        choices=("released_default", "structural_preserving"),
        default="released_default",
    )
    parser.add_argument(
        "--feature-protocol", choices=FEATURE_PROTOCOLS, default="ones"
    )
    parser.add_argument(
        "--sampling-protocol",
        choices=SAMPLING_PROTOCOLS,
        default=PAPER_SAMPLING_PROTOCOL,
        help=(
            "paper keeps Algorithms 1/3; released-legacy reproduces the "
            "audited revision 31166ca sampler and requires the compatibility "
            "profile"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--gpu-id", type=int,
        help=(
            "physical CUDA device exposed to training/evaluation; internally "
            "it is mapped to logical cuda:0"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-non-private", action="store_true")
    parser.add_argument("--only-non-private", action="store_true")
    parser.add_argument(
        "--clip-non-private",
        action="store_true",
        help=(
            "retain Algorithm 2 per-subgraph clipping in the zero-noise "
            "non-private baseline"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--isolated-policy", choices=("active", "preserve"), default="active"
    )
    parser.add_argument("--keep-self-loops", action="store_true")
    parser.add_argument(
        "--split-source",
        choices=("generated", "paper-artifact"),
        default="generated",
        help="use a seeded raw-data split or an available fixed author artifact",
    )
    parser.add_argument(
        "--privacy-accountant",
        choices=ACCOUNTANTS,
        default=PAPER_BINOMIAL,
    )
    parser.add_argument("--clip-norm", type=float, default=3.0)
    parser.add_argument(
        "--compatibility-update-cap", type=int,
        help=(
            "cap actual optimizer updates in released-epochs diagnostics "
            "without changing the 20-step privacy calibration"
        ),
    )
    args = parser.parse_args()
    if args.protocol_profile == RELEASED_COMPATIBILITY_PROTOCOL:
        for name, value in released_compatibility_overrides().items():
            setattr(args, name, value)
        # Revision 31166ca hard-coded lambda=1 in KSetMaxCoverAdjLoss.
        args.loss_penalty = 1.0
        args.training_protocol = "released-epochs"
        args.privacy_accounting_steps = 20
        args.optimizer_momentum = 0.9
        args.sensitivity_noise_factor = 2.0
        # Algorithm 1 with theta=10, r=1 cannot form the released n=40
        # subgraphs: its one-hop neighborhood has at most 11 vertices.  Keep
        # the tested paper sampler at r=3 while reproducing the released
        # manually supplied privacy bound (10) separately below.
        args.sampling_hops = 3
        args.compatibility_occurrence_hops = 1
        args.base_occurrence_bound = 10
    else:
        args.training_protocol = "fixed-iterations"
        args.privacy_accounting_steps = None
        args.sensitivity_noise_factor = 1.0
        args.sampling_hops = 3
        args.compatibility_occurrence_hops = None
        args.base_occurrence_bound = None
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.loss_penalty <= 0:
        parser.error("--loss-penalty must be positive")
    if args.optimizer_momentum < 0:
        parser.error("--optimizer-momentum must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0 < args.lr_decay <= 1:
        parser.error("--lr-decay must be in (0, 1]")
    if args.lr_decay_steps <= 0:
        parser.error("--lr-decay-steps must be positive")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1)")
    if (
        args.protocol_profile == RELEASED_COMPATIBILITY_PROTOCOL
        and args.parameter_averaging != "none"
    ):
        parser.error(
            "released-compatibility requires --parameter-averaging none"
        )
    if args.gpu_id is not None and args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    if args.gpu_id is not None and args.device == "cpu":
        parser.error("--gpu-id cannot be combined with --device cpu")
    if args.gpu_id is not None and args.device == "auto":
        # Fail loudly if the requested physical GPU is unavailable instead of
        # silently falling back to CPU after CUDA_VISIBLE_DEVICES remapping.
        args.device = "cuda"
    if args.split_source == "paper-artifact" and set(args.dataset) != {"facebook"}:
        parser.error(
            "--split-source paper-artifact currently supports only --dataset facebook"
        )
    if args.only_non_private and args.skip_non_private:
        parser.error("--only-non-private and --skip-non-private are mutually exclusive")
    if (
        args.compatibility_update_cap is not None
        and args.compatibility_update_cap <= 0
    ):
        parser.error("--compatibility-update-cap must be positive")
    if (
        args.compatibility_update_cap is not None
        and args.protocol_profile != RELEASED_COMPATIBILITY_PROTOCOL
    ):
        parser.error(
            "--compatibility-update-cap requires "
            "--protocol-profile released-compatibility"
        )
    if (
        args.dp_noise_stream == LEGACY_GLOBAL_DP_NOISE_STREAM
        and args.protocol_profile != RELEASED_COMPATIBILITY_PROTOCOL
    ):
        parser.error(
            "--dp-noise-stream legacy-global requires "
            "--protocol-profile released-compatibility"
        )
    if (
        args.sampling_protocol == RELEASED_LEGACY_SAMPLING_PROTOCOL
        and args.protocol_profile != RELEASED_COMPATIBILITY_PROTOCOL
    ):
        parser.error(
            "--sampling-protocol released-legacy requires "
            "--protocol-profile released-compatibility"
        )

    python = sys.executable
    # The fixed author artifact is consumed exactly as published, including
    # self-loops. Training removes them before subgraph sampling, matching the
    # historical implementation; evaluation retains the fixed edge list.
    remove_self_loops = (
        not args.keep_self_loops and args.split_source != "paper-artifact"
    )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = PROJECT_ROOT / "outputs" / "runs" / run_id
    evaluation_root = run_root / "evaluation"
    manifest_path = run_root / "manifest.json"

    commands = []
    if not args.skip_prepare:
        prepare = [
            python, "dataset/prepare_datasets.py", *args.dataset,
            "--seed", "0", "--train-ratio", "0.5", "--force",
            "--isolated-policy", args.isolated_policy,
            "--split-source", args.split_source,
        ]
        if remove_self_loops:
            prepare.append("--remove-self-loops")
        commands.append({"kind": "prepare", "command": prepare})

    for dataset in args.dataset:
        if not args.only_non_private:
            for setting in args.settings:
                for epsilon in args.epsilon:
                    commands.append({
                        "kind": "train",
                        "dataset": dataset,
                        "setting": setting,
                        "epsilon_dir": f"eps_{epsilon}",
                        "command": [
                            python, "experiments/train_models.py",
                            "--dataset", dataset,
                            "--settings", setting,
                            "--target_epsilon", str(epsilon),
                            "--n_iteration", str(args.iterations),
                            "--lr", str(args.learning_rate),
                            "--loss-penalty", str(args.loss_penalty),
                            "--batchsize", str(args.batch_size),
                            "--lr-decay", str(args.lr_decay),
                            "--lr-decay-steps", str(args.lr_decay_steps),
                            "--grat-variant", args.grat_variant,
                            "--grat-initialization", args.grat_initialization,
                            "--feature-protocol", args.feature_protocol,
                            "--sampling-protocol", args.sampling_protocol,
                            "--round", str(args.round),
                            "--seed", str(args.model_seed),
                            "--model-seed-mode", args.model_seed_mode,
                            "--sampling-seed", str(args.sampling_seed),
                            "--sampling-seed-mode", args.sampling_seed_mode,
                            "--batch-seed", str(args.batch_seed),
                            "--batch-seed-mode", args.batch_seed_mode,
                            "--dp-noise-seed", str(args.dp_noise_seed),
                            "--dp-noise-seed-mode", args.dp_noise_seed_mode,
                            "--dp-noise-stream", args.dp_noise_stream,
                            "--device", args.device,
                            "--auto-parameters",
                            "--privacy-accountant", args.privacy_accountant,
                            "--clip-norm", str(args.clip_norm),
                            "--protocol-profile", args.protocol_profile,
                            "--training-protocol", args.training_protocol,
                            "--optimizer-momentum", str(args.optimizer_momentum),
                            "--parameter-averaging", args.parameter_averaging,
                            "--ema-decay", str(args.ema_decay),
                            "--sensitivity-noise-factor",
                            str(args.sensitivity_noise_factor),
                            "--r", str(args.sampling_hops),
                            "--output-root", str(run_root),
                        ],
                    })
                    if args.privacy_accounting_steps is not None:
                        commands[-1]["command"].extend([
                            "--privacy-accounting-steps",
                            str(args.privacy_accounting_steps),
                        ])
                    if args.base_occurrence_bound is not None:
                        commands[-1]["command"].extend([
                            "--base-occurrence-bound",
                            str(args.base_occurrence_bound),
                        ])
                    if args.compatibility_update_cap is not None:
                        commands[-1]["command"].extend([
                            "--optimizer-update-cap",
                            str(args.compatibility_update_cap),
                        ])
        if not args.skip_non_private:
            non_private_command = [
                    python, "experiments/train_models.py",
                    "--dataset", dataset,
                    "--settings", "PrivIM+",
                    "--no-dp",
                    "--n_iteration", str(args.iterations),
                    "--lr", str(args.learning_rate),
                    "--loss-penalty", str(args.loss_penalty),
                    "--batchsize", str(args.batch_size),
                    "--lr-decay", str(args.lr_decay),
                    "--lr-decay-steps", str(args.lr_decay_steps),
                    "--grat-variant", args.grat_variant,
                    "--grat-initialization", args.grat_initialization,
                    "--feature-protocol", args.feature_protocol,
                    "--sampling-protocol", args.sampling_protocol,
                    "--round", str(args.round),
                    "--seed", str(args.model_seed),
                    "--model-seed-mode", args.model_seed_mode,
                    "--sampling-seed", str(args.sampling_seed),
                    "--sampling-seed-mode", args.sampling_seed_mode,
                    "--batch-seed", str(args.batch_seed),
                    "--batch-seed-mode", args.batch_seed_mode,
                    "--dp-noise-seed", str(args.dp_noise_seed),
                    "--dp-noise-seed-mode", args.dp_noise_seed_mode,
                    "--dp-noise-stream", args.dp_noise_stream,
                    "--device", args.device,
                    "--auto-parameters",
                    "--privacy-accountant", args.privacy_accountant,
                    "--clip-norm", str(args.clip_norm),
                    "--protocol-profile", args.protocol_profile,
                    "--training-protocol", args.training_protocol,
                    "--optimizer-momentum", str(args.optimizer_momentum),
                    "--parameter-averaging", args.parameter_averaging,
                    "--ema-decay", str(args.ema_decay),
                    "--sensitivity-noise-factor",
                    str(args.sensitivity_noise_factor),
                    "--r", str(args.sampling_hops),
                    "--output-root", str(run_root),
                ]
            if args.privacy_accounting_steps is not None:
                non_private_command.extend([
                    "--privacy-accounting-steps",
                    str(args.privacy_accounting_steps),
                ])
            if args.base_occurrence_bound is not None:
                non_private_command.extend([
                    "--base-occurrence-bound",
                    str(args.base_occurrence_bound),
                ])
            if args.compatibility_update_cap is not None:
                non_private_command.extend([
                    "--optimizer-update-cap",
                    str(args.compatibility_update_cap),
                ])
            if args.clip_non_private:
                non_private_command.append("--clip-non-private")
            commands.append({
                "kind": "train",
                "dataset": dataset,
                "setting": "PrivIM+",
                "epsilon_dir": "eps_10000",
                "command": non_private_command,
                "non_private_clipping": args.clip_non_private,
            })

    evaluation_settings = ["PrivIM+"] if args.only_non_private else args.settings
    commands.append({
        "kind": "evaluate",
        "command": [
            python, "experiments/evaluate_models.py",
            "--dataset", *args.dataset,
            "--settings", *evaluation_settings,
            "--device", args.device,
            "--k", "50",
            "--models-root", str(run_root),
            "--evaluation-root", str(evaluation_root),
        ],
    })

    if args.dry_run:
        print(f"run_root={run_root}")
        environment_prefix = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
            if args.gpu_id is not None else ""
        )
        for item in commands:
            print(
                environment_prefix
                + " ".join(str(part) for part in item["command"])
            )
        return

    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"run already exists: {run_root}; use --resume or a new --run-id"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    prepared_snapshot = {}
    if args.skip_prepare:
        for dataset in args.dataset:
            prepared_snapshot[dataset] = validate_prepared(
                dataset, args.isolated_policy, remove_self_loops,
                args.split_source,
            )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "paper_sha256": sha256_file(PROJECT_ROOT / "PrivIM_Report.pdf"),
        "git": git_state(),
        "configuration": vars(args),
        "run_root": str(run_root),
        "commands": [
            {**item, "command": [str(part) for part in item["command"]], "status": "pending"}
            for item in commands
        ],
        "prepared_datasets": prepared_snapshot,
    }
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_configuration = previous.get("configuration", {})
        seed_keys = (
            "model_seed", "model_seed_mode", "sampling_seed",
            "sampling_seed_mode", "batch_seed", "batch_seed_mode",
            "dp_noise_seed", "dp_noise_seed_mode",
            "dp_noise_stream",
            "clip_non_private",
            "feature_protocol",
            "protocol_profile",
            "sampling_protocol",
            "compatibility_update_cap",
            "parameter_averaging",
            "ema_decay",
            "batch_size",
            "optimizer_momentum",
            "loss_penalty",
        )
        compatibility_defaults = {
            # Manifests written before this field existed necessarily used
            # the paper sampler.  Preserve resume compatibility for them.
            "sampling_protocol": PAPER_SAMPLING_PROTOCOL,
            "dp_noise_stream": "independent",
            "parameter_averaging": "none",
            "ema_decay": 0.99,
            "batch_size": 16,
            "optimizer_momentum": 0.0,
            "loss_penalty": 1.0,
        }
        incompatible = {
            key: (
                previous_configuration.get(
                    key, compatibility_defaults.get(key)
                ),
                getattr(args, key),
            )
            for key in seed_keys
            if previous_configuration.get(
                key, compatibility_defaults.get(key)
            ) != getattr(args, key)
        }
        if incompatible:
            details = ", ".join(
                f"{key}: old={old!r}, requested={new!r}"
                for key, (old, new) in incompatible.items()
            )
            raise ValueError(
                "cannot resume a run created with a different or legacy "
                f"seed protocol ({details}); use a new --run-id"
            )
        manifest["created_at"] = previous.get("created_at", manifest["created_at"])
        manifest["resumed_at"] = datetime.now().isoformat()
    write_manifest(manifest_path, manifest)

    expected_seed_protocol = seed_protocol(
        model_seed=args.model_seed,
        model_seed_mode=args.model_seed_mode,
        sampling_seed=args.sampling_seed,
        batch_seed=args.batch_seed,
        batch_seed_mode=args.batch_seed_mode,
        sampling_seed_mode=args.sampling_seed_mode,
        dp_noise_seed=args.dp_noise_seed,
        dp_noise_seed_mode=args.dp_noise_seed_mode,
    )
    child_environment = subprocess_environment(args.gpu_id)

    for index, item in enumerate(commands):
        if item["kind"] == "prepare" and args.resume:
            for dataset in args.dataset:
                manifest["prepared_datasets"][dataset] = validate_prepared(
                    dataset, args.isolated_policy, remove_self_loops,
                    args.split_source,
                )
            manifest["commands"][index]["status"] = "skipped-existing"
            write_manifest(manifest_path, manifest)
            continue
        if item["kind"] == "train" and args.resume:
            count = completed_models(
                run_root, item["dataset"], item["setting"],
                item["epsilon_dir"], args.iterations,
                args.learning_rate, args.lr_decay, args.lr_decay_steps,
                args.grat_variant,
                expected_seed_protocol,
                args.grat_initialization,
                item.get("non_private_clipping", False),
                args.feature_protocol,
                args.protocol_profile,
                args.sampling_protocol,
                args.compatibility_update_cap,
                args.dp_noise_stream,
                args.parameter_averaging,
                args.ema_decay,
                args.batch_size,
                args.optimizer_momentum,
                args.loss_penalty,
            )
            if count >= args.round:
                manifest["commands"][index]["status"] = "skipped-existing"
                manifest["commands"][index]["checkpoint_count"] = count
                write_manifest(manifest_path, manifest)
                continue
        print(" ".join(str(part) for part in item["command"]), flush=True)
        manifest["commands"][index]["status"] = "running"
        write_manifest(manifest_path, manifest)
        try:
            subprocess.run(
                item["command"], cwd=PROJECT_ROOT, check=True,
                env=child_environment,
            )
        except BaseException:
            manifest["commands"][index]["status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise
        manifest["commands"][index]["status"] = "completed"
        write_manifest(manifest_path, manifest)

        if item["kind"] == "prepare":
            for dataset in args.dataset:
                manifest["prepared_datasets"][dataset] = validate_prepared(
                    dataset, args.isolated_policy, remove_self_loops,
                    args.split_source,
                )
            write_manifest(manifest_path, manifest)

    manifest["completed_at"] = datetime.now().isoformat()
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
