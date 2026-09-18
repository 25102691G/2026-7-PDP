#!/usr/bin/env python3
"""Create train/test edge lists directly from original node IDs.

Processing route:

1. Parse the raw edge endpoints without constructing an igraph graph.
2. Randomly split the original node IDs into train/test sets.
3. Keep only edges whose endpoints belong to the same split.
4. Apply the selected isolated-node policy.
5. Densely renumber the retained nodes in each induced graph to ``0..N-1``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, TextIO, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "raw_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "prepared"

PAPER_ARTIFACTS = {
    "facebook": {
        "revision": "31166ca",
        "train": "dataset/undirected/Facebook_train.txt",
        "test": "dataset/undirected/Facebook_test.txt",
        "original_nodes": 22470,
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_path: str
    source_format: str
    directed: bool
    has_header: bool = False
    too_large_for_memory: bool = False


DATASETS: Dict[str, DatasetSpec] = {
    "email": DatasetSpec(
        name="email",
        relative_path="directed/email-Eu-core.txt",
        source_format="whitespace",
        directed=True,
    ),
    "bitcoin": DatasetSpec(
        name="bitcoin",
        relative_path="directed/soc-sign-bitcoinotc.csv",
        source_format="csv",
        directed=True,
    ),
    "lastfm": DatasetSpec(
        name="lastfm",
        relative_path="undirected/lasftm_asia/lastfm_asia_edges.csv",
        source_format="csv",
        directed=False,
        has_header=True,
    ),
    "hepph": DatasetSpec(
        name="hepph",
        relative_path="undirected/ca-HepPh.txt",
        source_format="whitespace",
        directed=False,
    ),
    "facebook": DatasetSpec(
        name="facebook",
        relative_path="undirected/facebook_edges.csv",
        source_format="csv",
        directed=False,
        has_header=True,
    ),
    "gowalla": DatasetSpec(
        name="gowalla",
        relative_path="undirected/loc-gowalla_edges.txt",
        source_format="whitespace",
        directed=False,
    ),
    "friendster": DatasetSpec(
        name="friendster",
        relative_path="undirected/com-Friendster/com-Friendster.mtx",
        source_format="matrix-market",
        directed=False,
        too_large_for_memory=True,
    ),
}


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def iter_csv_edges(path: Path, has_header: bool) -> Iterator[Tuple[int, int]]:
    with open_text(path) as handle:
        rows = csv.reader(handle)
        if has_header:
            next(rows, None)
        first_data_line = 2 if has_header else 1
        for line_number, row in enumerate(rows, start=first_data_line):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_number}: expected two endpoints")
            try:
                yield int(row[0].strip()), int(row[1].strip())
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid endpoints {row[:2]!r}"
                ) from exc


def iter_whitespace_edges(path: Path) -> Iterator[Tuple[int, int]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected two endpoints")
            try:
                yield int(fields[0]), int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid endpoints {fields[:2]!r}"
                ) from exc


def iter_edges(path: Path, spec: DatasetSpec) -> Iterator[Tuple[int, int]]:
    if spec.source_format == "csv":
        yield from iter_csv_edges(path, spec.has_header)
    elif spec.source_format == "whitespace":
        yield from iter_whitespace_edges(path)
    else:
        raise ValueError(
            f"{spec.name}: Matrix Market input is not supported by the "
            "in-memory original-ID splitter"
        )


def collect_original_graph(
    source_path: Path,
    spec: DatasetSpec,
    remove_self_loops: bool,
    keep_duplicate_edges: bool,
) -> Tuple[List[Tuple[int, int]], Set[int], Dict[str, int]]:
    """Read endpoints while keeping their original IDs."""
    nodes: Set[int] = set()
    edges: List[Tuple[int, int]] = []
    unique_edges: Set[Tuple[int, int]] = set()
    raw_rows = 0
    self_loops_seen = 0
    self_loops_removed = 0
    duplicates_removed = 0

    for source, target in iter_edges(source_path, spec):
        raw_rows += 1
        nodes.add(source)
        nodes.add(target)

        if source == target:
            self_loops_seen += 1
            if remove_self_loops:
                self_loops_removed += 1
                continue

        if not spec.directed and source > target:
            source, target = target, source
        edge = (source, target)

        if keep_duplicate_edges:
            edges.append(edge)
        elif edge in unique_edges:
            duplicates_removed += 1
        else:
            unique_edges.add(edge)

    if not keep_duplicate_edges:
        edges = sorted(unique_edges)

    statistics = {
        "raw_rows": raw_rows,
        "canonical_edges": len(edges),
        "self_loops_seen": self_loops_seen,
        "self_loops_removed": self_loops_removed,
        "duplicates_removed": duplicates_removed,
    }
    return edges, nodes, statistics


def split_original_nodes(
    original_nodes: Set[int],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Shuffle original IDs reproducibly and split them by node count."""
    shuffled_nodes = sorted(original_nodes)
    random.Random(seed).shuffle(shuffled_nodes)
    split_index = int(len(shuffled_nodes) * train_ratio)
    return shuffled_nodes[:split_index], shuffled_nodes[split_index:]


def induced_original_edges(
    edges: Sequence[Tuple[int, int]],
    train_nodes: Set[int],
    test_nodes: Set[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
    train_edges: List[Tuple[int, int]] = []
    test_edges: List[Tuple[int, int]] = []
    cross_edges = 0

    for source, target in edges:
        if source in train_nodes and target in train_nodes:
            train_edges.append((source, target))
        elif source in test_nodes and target in test_nodes:
            test_edges.append((source, target))
        else:
            cross_edges += 1

    return train_edges, test_edges, cross_edges


def renumber_split_edges(
    original_edges: Sequence[Tuple[int, int]],
    split_order: Sequence[int],
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """Renumber the retained split nodes into a contiguous local ID space."""
    local_id = {original_id: index for index, original_id in enumerate(split_order)}
    renumbered = [
        (local_id[source], local_id[target]) for source, target in original_edges
    ]
    return renumbered, local_id


def retained_split_order(
    split_order: Sequence[int],
    split_edges: Sequence[Tuple[int, int]],
    isolated_policy: str,
) -> List[int]:
    if isolated_policy == "preserve":
        return list(split_order)
    if isolated_policy == "active":
        active = {node for edge in split_edges for node in edge}
        return [node for node in split_order if node in active]
    raise ValueError(f"unsupported isolated-node policy: {isolated_policy}")


def output_names(dataset_name: str) -> Tuple[str, ...]:
    return (
        f"{dataset_name}_train_edges.txt",
        f"{dataset_name}_test_edges.txt",
        "train_nodes.csv",
        "test_nodes.csv",
        "node_mapping.csv",
        "metadata.json",
    )


def reserve_output_directory(
    output_root: Path,
    dataset_name: str,
    force: bool,
) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / dataset_name
    expected_outputs = output_names(dataset_name)
    existing = [name for name in expected_outputs if (final_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{final_dir} already contains {', '.join(existing)}; "
            "use --force to replace them"
        )
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{dataset_name}.", dir=output_root))
    return final_dir, temp_dir


def write_edges(path: Path, edges: Sequence[Tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source, target in edges:
            handle.write(f"{source} {target}\n")


def write_node_split(path: Path, order: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("original_id", "local_id"))
        writer.writerows((original_id, local_id) for local_id, original_id in enumerate(order))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_outputs(
    temp_dir: Path,
    final_dir: Path,
    dataset_name: str,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in output_names(dataset_name):
        os.replace(temp_dir / name, final_dir / name)
    temp_dir.rmdir()


def _read_git_artifact(revision: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FileNotFoundError(
            f"paper split artifact is unavailable: {revision}:{path}"
        ) from exc


def _parse_artifact_edges(content: bytes, label: str) -> List[Tuple[int, int]]:
    edges = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"{label}:{line_number}: expected two endpoints")
        edges.append((int(fields[0]), int(fields[1])))
    if not edges:
        raise ValueError(f"{label}: artifact contains no edges")
    return edges


def prepare_paper_artifact(
    spec: DatasetSpec,
    output_root: Path,
    remove_self_loops: bool,
    force: bool,
) -> Dict[str, int]:
    """Publish the fixed split shipped in the authors' repository history."""
    artifact = PAPER_ARTIFACTS.get(spec.name)
    if artifact is None:
        raise ValueError(
            f"{spec.name}: no fixed paper split artifact is available"
        )
    revision = artifact["revision"]
    contents = {
        split: _read_git_artifact(revision, artifact[split])
        for split in ("train", "test")
    }
    split_edges = {
        split: _parse_artifact_edges(
            contents[split], f"{revision}:{artifact[split]}"
        )
        for split in ("train", "test")
    }
    self_loops_seen = sum(
        source == target
        for edges in split_edges.values()
        for source, target in edges
    )
    if remove_self_loops:
        split_edges = {
            split: [(u, v) for u, v in edges if u != v]
            for split, edges in split_edges.items()
        }

    node_counts = {
        split: max(max(edge) for edge in edges) + 1
        for split, edges in split_edges.items()
    }
    final_dir, temp_dir = reserve_output_directory(
        output_root, spec.name, force
    )
    try:
        write_edges(temp_dir / f"{spec.name}_train_edges.txt", split_edges["train"])
        write_edges(temp_dir / f"{spec.name}_test_edges.txt", split_edges["test"])
        for split in ("train", "test"):
            write_node_split(
                temp_dir / f"{split}_nodes.csv", range(node_counts[split])
            )
        with (temp_dir / "node_mapping.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("artifact_local_id", "split", "local_id"))
            for split in ("train", "test"):
                writer.writerows(
                    (node, split, node) for node in range(node_counts[split])
                )

        combined_hash = hashlib.sha256(
            contents["train"] + b"\0" + contents["test"]
        ).hexdigest()
        metadata = {
            "dataset": spec.name,
            "directed": spec.directed,
            "split_source": "paper-artifact",
            "artifact_revision": revision,
            "artifact_train_path": artifact["train"],
            "artifact_test_path": artifact["test"],
            "artifact_train_sha256": hashlib.sha256(contents["train"]).hexdigest(),
            "artifact_test_sha256": hashlib.sha256(contents["test"]).hexdigest(),
            "source_sha256": combined_hash,
            "seed": None,
            "train_ratio": 0.5,
            "remove_self_loops": remove_self_loops,
            "keep_duplicate_edges": False,
            "isolated_policy": "artifact_edge_list",
            "node_id_semantics": "artifact_local_ids",
            "original_nodes": int(artifact["original_nodes"]),
            "train_assigned_nodes": int(artifact["original_nodes"]) // 2,
            "test_assigned_nodes": int(artifact["original_nodes"]) // 2,
            "train_nodes": node_counts["train"],
            "test_nodes": node_counts["test"],
            "train_isolated_nodes_removed": None,
            "test_isolated_nodes_removed": None,
            "train_edges": len(split_edges["train"]),
            "test_edges": len(split_edges["test"]),
            "self_loops_seen": self_loops_seen,
            "self_loops_removed": self_loops_seen if remove_self_loops else 0,
            "cross_edges_removed": None,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_outputs(temp_dir, final_dir, spec.name)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "train_assigned_nodes": metadata["train_assigned_nodes"],
        "test_assigned_nodes": metadata["test_assigned_nodes"],
        "train_active_nodes": node_counts["train"],
        "test_active_nodes": node_counts["test"],
        "train_edges": len(split_edges["train"]),
        "test_edges": len(split_edges["test"]),
        "cross_edges_removed": 0,
    }


def prepare_dataset(
    spec: DatasetSpec,
    raw_root: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
    remove_self_loops: bool,
    keep_duplicate_edges: bool,
    force: bool,
    isolated_policy: str = "preserve",
) -> Dict[str, int]:
    if spec.too_large_for_memory:
        raise ValueError(
            "Friendster is intentionally excluded from this in-memory route. "
            "It requires a separate external-memory partitioning process."
        )

    source_path = raw_root / spec.relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"{spec.name}: source file not found: {source_path}")

    edges, original_nodes, source_stats = collect_original_graph(
        source_path=source_path,
        spec=spec,
        remove_self_loops=remove_self_loops,
        keep_duplicate_edges=keep_duplicate_edges,
    )
    train_order, test_order = split_original_nodes(
        original_nodes=original_nodes,
        train_ratio=train_ratio,
        seed=seed,
    )
    train_original_edges, test_original_edges, cross_edges = induced_original_edges(
        edges=edges,
        train_nodes=set(train_order),
        test_nodes=set(test_order),
    )
    retained_train_order = retained_split_order(
        train_order, train_original_edges, isolated_policy
    )
    retained_test_order = retained_split_order(
        test_order, test_original_edges, isolated_policy
    )
    train_edges, train_local_ids = renumber_split_edges(
        train_original_edges, retained_train_order
    )
    test_edges, test_local_ids = renumber_split_edges(
        test_original_edges, retained_test_order
    )

    final_dir, temp_dir = reserve_output_directory(
        output_root, spec.name, force
    )
    train_output_name, test_output_name = output_names(spec.name)[:2]
    try:
        write_edges(temp_dir / train_output_name, train_edges)
        write_edges(temp_dir / test_output_name, test_edges)
        write_node_split(temp_dir / "train_nodes.csv", retained_train_order)
        write_node_split(temp_dir / "test_nodes.csv", retained_test_order)
        with (temp_dir / "node_mapping.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("original_id", "split", "local_id"))
            writer.writerows((node, "train", local) for node, local in train_local_ids.items())
            writer.writerows((node, "test", local) for node, local in test_local_ids.items())
        metadata = {
            "dataset": spec.name,
            "directed": spec.directed,
            "split_source": "generated",
            "source": str(source_path.relative_to(raw_root)),
            "source_sha256": sha256_file(source_path),
            "seed": seed,
            "train_ratio": train_ratio,
            "remove_self_loops": remove_self_loops,
            "keep_duplicate_edges": keep_duplicate_edges,
            "isolated_policy": isolated_policy,
            **source_stats,
            "original_nodes": len(original_nodes),
            "train_assigned_nodes": len(train_order),
            "test_assigned_nodes": len(test_order),
            "train_nodes": len(retained_train_order),
            "test_nodes": len(retained_test_order),
            "train_isolated_nodes_removed": len(train_order) - len(retained_train_order),
            "test_isolated_nodes_removed": len(test_order) - len(retained_test_order),
            "train_edges": len(train_edges),
            "test_edges": len(test_edges),
            "cross_edges_removed": cross_edges,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        publish_outputs(temp_dir, final_dir, spec.name)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        **source_stats,
        "original_nodes": len(original_nodes),
        "train_assigned_nodes": len(train_order),
        "test_assigned_nodes": len(test_order),
        "train_active_nodes": len(retained_train_order),
        "test_active_nodes": len(retained_test_order),
        "train_edges": len(train_edges),
        "test_edges": len(test_edges),
        "cross_edges_removed": cross_edges,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split original node IDs into train/test sets, construct induced "
            "subgraphs, and densely renumber only the final edge lists."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=tuple(DATASETS) + ("all",),
        help="one or more dataset names, or 'all' (Friendster is skipped)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument(
        "--split-source",
        choices=("generated", "paper-artifact"),
        default="generated",
        help=(
            "generate a seeded split from raw data, or import an exact fixed "
            "split retained in the repository history"
        ),
    )
    parser.add_argument(
        "--remove-self-loops",
        action="store_true",
        help="remove self-loop edges before splitting",
    )
    parser.add_argument(
        "--keep-duplicate-edges",
        action="store_true",
        help="preserve duplicate/parallel edge rows",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing dataset-prefixed train/test edge files",
    )
    parser.add_argument(
        "--isolated-policy",
        choices=("active", "preserve"),
        default="active",
        help=(
            "active keeps only nodes incident to an edge after the split; "
            "preserve retains all assigned split members"
        ),
    )
    args = parser.parse_args(argv)
    return args


def selected_specs(dataset_names: Sequence[str]) -> List[DatasetSpec]:
    if dataset_names == ["all"]:
        print("Skipping Friendster: it requires external-memory processing.")
        return [
            spec for spec in DATASETS.values() if not spec.too_large_for_memory
        ]

    selected: List[DatasetSpec] = []
    seen: Set[str] = set()
    for name in dataset_names:
        if name not in seen:
            selected.append(DATASETS[name])
            seen.add(name)
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        raw_root = args.raw_root.resolve()
        output_root = args.output_root.resolve()
        for spec in selected_specs(args.datasets):
            if args.split_source == "paper-artifact":
                print(f"Preparing {spec.name} from its fixed paper artifact")
                stats = prepare_paper_artifact(
                    spec=spec,
                    output_root=output_root,
                    remove_self_loops=args.remove_self_loops,
                    force=args.force,
                )
            else:
                print(f"Preparing {spec.name} from {raw_root / spec.relative_path}")
                stats = prepare_dataset(
                    spec=spec,
                    raw_root=raw_root,
                    output_root=output_root,
                    train_ratio=args.train_ratio,
                    seed=args.seed,
                    remove_self_loops=args.remove_self_loops,
                    keep_duplicate_edges=args.keep_duplicate_edges,
                    force=args.force,
                    isolated_policy=args.isolated_policy,
                )
            print(
                f"Prepared {spec.name}: "
                f"assigned_nodes={stats['train_assigned_nodes']}/"
                f"{stats['test_assigned_nodes']}, "
                f"active_nodes={stats['train_active_nodes']}/"
                f"{stats['test_active_nodes']}, "
                f"edges={stats['train_edges']}/{stats['test_edges']}, "
                f"cross_edges_removed={stats['cross_edges_removed']}"
            )
        return 0
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
