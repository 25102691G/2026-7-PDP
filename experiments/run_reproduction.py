#!/usr/bin/env python3
"""Run a manifest-scoped PrivIM reproduction matrix."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import DATASETS, EPSILONS
    from .privacy import ACCOUNTANTS, PAPER_BINOMIAL
    from .seeding import SEED_MODES, seed_protocol
except ImportError:  # Direct execution from the repository root.
    from config import DATASETS, EPSILONS
    from privacy import ACCOUNTANTS, PAPER_BINOMIAL
    from seeding import SEED_MODES, seed_protocol
from model.GRAT import GRAT_VARIANTS
from experiments.graph_util import FEATURE_PROTOCOLS


PREPARED_ROOT = PROJECT_ROOT / "dataset" / "prepared"


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
        if int(metadata.get("iterations", -1)) != iterations:
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
            expected_seed_protocol is not None
            and metadata.get("experiment", {}).get("seed_protocol")
            != expected_seed_protocol
        ):
            continue
        completed += 1
    return completed


def main():
    parser = argparse.ArgumentParser()
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
        "--sampling-seed-mode", choices=SEED_MODES, default="fixed",
        help="fixed reuses one subgraph container across model rounds",
    )
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument(
        "--batch-seed-mode", choices=SEED_MODES, default="per-round"
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
        "--lr-decay", type=float, default=1.0,
        help="StepLR multiplier; the paper reproduction keeps 0.005 fixed",
    )
    parser.add_argument("--lr-decay-steps", type=int, default=5)
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 < args.lr_decay <= 1:
        parser.error("--lr-decay must be in (0, 1]")
    if args.lr_decay_steps <= 0:
        parser.error("--lr-decay-steps must be positive")
    if args.split_source == "paper-artifact" and set(args.dataset) != {"facebook"}:
        parser.error(
            "--split-source paper-artifact currently supports only --dataset facebook"
        )
    if args.only_non_private and args.skip_non_private:
        parser.error("--only-non-private and --skip-non-private are mutually exclusive")

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
                            "--lr-decay", str(args.lr_decay),
                            "--lr-decay-steps", str(args.lr_decay_steps),
                            "--grat-variant", args.grat_variant,
                            "--grat-initialization", args.grat_initialization,
                            "--feature-protocol", args.feature_protocol,
                            "--round", str(args.round),
                            "--seed", str(args.model_seed),
                            "--model-seed-mode", args.model_seed_mode,
                            "--sampling-seed", str(args.sampling_seed),
                            "--sampling-seed-mode", args.sampling_seed_mode,
                            "--batch-seed", str(args.batch_seed),
                            "--batch-seed-mode", args.batch_seed_mode,
                            "--device", args.device,
                            "--auto-parameters",
                            "--privacy-accountant", args.privacy_accountant,
                            "--clip-norm", str(args.clip_norm),
                            "--output-root", str(run_root),
                        ],
                    })
        if not args.skip_non_private:
            non_private_command = [
                    python, "experiments/train_models.py",
                    "--dataset", dataset,
                    "--settings", "PrivIM+",
                    "--no-dp",
                    "--n_iteration", str(args.iterations),
                    "--lr", str(args.learning_rate),
                    "--lr-decay", str(args.lr_decay),
                    "--lr-decay-steps", str(args.lr_decay_steps),
                    "--grat-variant", args.grat_variant,
                    "--grat-initialization", args.grat_initialization,
                    "--feature-protocol", args.feature_protocol,
                    "--round", str(args.round),
                    "--seed", str(args.model_seed),
                    "--model-seed-mode", args.model_seed_mode,
                    "--sampling-seed", str(args.sampling_seed),
                    "--sampling-seed-mode", args.sampling_seed_mode,
                    "--batch-seed", str(args.batch_seed),
                    "--batch-seed-mode", args.batch_seed_mode,
                    "--device", args.device,
                    "--auto-parameters",
                    "--privacy-accountant", args.privacy_accountant,
                    "--clip-norm", str(args.clip_norm),
                    "--output-root", str(run_root),
                ]
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
        for item in commands:
            print(" ".join(str(part) for part in item["command"]))
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
            "clip_non_private",
            "feature_protocol",
        )
        incompatible = {
            key: (previous_configuration.get(key), getattr(args, key))
            for key in seed_keys
            if previous_configuration.get(key) != getattr(args, key)
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
    )

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
            subprocess.run(item["command"], cwd=PROJECT_ROOT, check=True)
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
