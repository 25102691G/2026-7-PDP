"""Read existing prepared splits without preparing or overwriting anything."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict

import igraph


DATASETS = ("email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla")


def load_prepared(dataset: str, split: str, root: Path) -> igraph.Graph:
    if dataset not in DATASETS or split not in {"train", "test"}:
        raise ValueError("unknown prepared dataset or split")
    directory = Path(root)/dataset
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    count = int(metadata[f"{split}_nodes"])
    if count < 0:
        raise ValueError("node count must be nonnegative")
    edges = []
    with (directory/f"{dataset}_{split}_edges.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                fields = line.split()
                edges.append((int(fields[0]), int(fields[1])))
    graph = igraph.Graph(n=count, edges=edges, directed=bool(metadata["directed"]))
    identifiers = [None]*count
    with (directory/f"{split}_nodes.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            local = int(row["local_id"])
            if not 0 <= local < count or identifiers[local] is not None:
                raise ValueError("invalid prepared node mapping")
            identifiers[local] = int(row["original_id"])
    if any(value is None for value in identifiers) or len(set(identifiers)) != count:
        raise ValueError("node mapping must be complete and unique")
    graph.vs["stable_id"] = identifiers
    return graph


def trusted_snapshot(dataset: str, root: Path) -> Dict[str, str]:
    """Research provenance only: never attach input hashes to DP releases."""
    directory = Path(root)/dataset
    names = [
        "metadata.json", "train_nodes.csv", "test_nodes.csv",
        f"{dataset}_train_edges.txt", f"{dataset}_test_edges.txt",
    ]
    return {
        name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in names
    }
