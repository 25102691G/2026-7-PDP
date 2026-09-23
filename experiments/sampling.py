"""Subgraph samplers matching Algorithms 1 and 3 in the PrivIM paper.

The functions in this module operate only on :mod:`igraph` graphs.  Keeping
the sampling logic independent from DGL makes the privacy-relevant invariants
straightforward to test: fixed subgraph size, bounded walk length, r-hop
containment, and bounded node frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import igraph
import numpy as np


PAPER_SAMPLING_PROTOCOL = "paper"
RELEASED_LEGACY_SAMPLING_PROTOCOL = "released-legacy"
SAMPLING_PROTOCOLS = (
    PAPER_SAMPLING_PROTOCOL,
    RELEASED_LEGACY_SAMPLING_PROTOCOL,
)


@dataclass
class SamplingResult:
    node_sets: List[List[int]]
    frequencies: List[int]
    stage1_count: int
    stage2_count: int = 0


def project_max_indegree(
    graph: igraph.Graph, theta: int, rng: random.Random
) -> igraph.Graph:
    """Return the paper's theta-bounded projection of ``graph``.

    Incoming edges above ``theta`` are removed uniformly at random.  Vertex
    IDs are preserved, including isolated vertices.
    """
    if theta < 0:
        raise ValueError("theta must be non-negative")
    projected = graph.copy()
    for vertex in range(projected.vcount()):
        incoming = projected.incident(vertex, mode="in")
        if len(incoming) > theta:
            # Delete immediately.  For undirected graphs an edge is incident
            # to both endpoints; batching all deletions from degrees measured
            # on the original graph would over-prune the projection.
            projected.delete_edges(rng.sample(incoming, len(incoming) - theta))
    return projected


def _r_hop_nodes(graph: igraph.Graph, start: int, r: Optional[int]) -> set[int]:
    if r is None:
        return set(range(graph.vcount()))
    if r < 0:
        raise ValueError("r must be non-negative")
    return set(graph.neighborhood(start, order=r, mode="out"))


def _weighted_choice(
    candidates: Sequence[int], weights: Optional[Sequence[float]], rng: random.Random
) -> int:
    if weights is None:
        return rng.choice(candidates)
    total = sum(weights)
    if total <= 0:
        raise ValueError("sampling weights must have a positive sum")
    target = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if target <= cumulative:
            return candidate
    return candidates[-1]


def _walk_from_start(
    graph: igraph.Graph,
    start: int,
    size: int,
    restart_probability: float,
    walk_length: int,
    rng: random.Random,
    allowed: set[int],
    frequencies: Optional[Sequence[int]] = None,
    threshold: Optional[int] = None,
    decay: float = 1.0,
) -> Optional[List[int]]:
    """Run one RWR trial and return a fixed-size ordered node set."""
    if start not in allowed:
        return None
    visited = [start]
    seen = {start}
    current = start

    for _ in range(walk_length):
        if len(visited) == size:
            return visited
        if rng.random() < restart_probability:
            current = start

        candidates = [
            node
            for node in graph.neighbors(current, mode="out")
            if node in allowed
            and (frequencies is None or frequencies[node] < threshold)
        ]
        if not candidates:
            # A restart on the next iteration cannot help when the current
            # node is already the start; terminate this unsuccessful trial.
            if current == start:
                break
            current = start
            continue

        weights = None
        if frequencies is not None:
            weights = [1.0 / ((frequencies[node] + 1) ** decay) for node in candidates]
        current = _weighted_choice(candidates, weights, rng)
        if current not in seen:
            seen.add(current)
            visited.append(current)

    return visited if len(visited) == size else None


def _candidate_starts(
    vertices: Iterable[int], sample_rate: float, rng: random.Random
) -> List[int]:
    if not 0 <= sample_rate <= 1:
        raise ValueError("sample_rate must be in [0, 1]")
    # Algorithm 1/3 considers every vertex once and performs an independent
    # Bernoulli(q) decision.  It does not repeatedly draw starts until a fixed
    # number of subgraphs has been collected.
    return [vertex for vertex in vertices if rng.random() < sample_rate]


def sample_privim(
    graph: igraph.Graph,
    *,
    size: int,
    theta: int,
    r: int,
    sample_rate: float,
    restart_probability: float,
    walk_length: int,
    rng: random.Random,
) -> Tuple[igraph.Graph, SamplingResult]:
    """Algorithm 1: theta projection followed by r-hop constrained RWR."""
    projected = project_max_indegree(graph, theta, rng)
    node_sets: List[List[int]] = []
    frequencies = [0] * projected.vcount()
    for start in _candidate_starts(range(projected.vcount()), sample_rate, rng):
        nodes = _walk_from_start(
            projected,
            start,
            size,
            restart_probability,
            walk_length,
            rng,
            _r_hop_nodes(projected, start, r),
        )
        if nodes is not None:
            node_sets.append(nodes)
            for node in nodes:
                frequencies[node] += 1
    return projected, SamplingResult(node_sets, frequencies, len(node_sets))


def _frequency_stage(
    graph: igraph.Graph,
    vertices: Sequence[int],
    *,
    size: int,
    sample_rate: float,
    restart_probability: float,
    walk_length: int,
    threshold: int,
    decay: float,
    frequencies: List[int],
    rng: random.Random,
) -> List[List[int]]:
    node_sets: List[List[int]] = []
    allowed = set(vertices)
    for start in _candidate_starts(vertices, sample_rate, rng):
        if frequencies[start] >= threshold:
            continue
        nodes = _walk_from_start(
            graph,
            start,
            size,
            restart_probability,
            walk_length,
            rng,
            allowed,
            frequencies,
            threshold,
            decay,
        )
        if nodes is not None:
            node_sets.append(nodes)
            for node in nodes:
                frequencies[node] += 1
    return node_sets


def sample_privim_star(
    graph: igraph.Graph,
    *,
    size: int,
    boundary_divisor: int,
    sample_rate: float,
    restart_probability: float,
    walk_length: int,
    threshold: int,
    decay: float,
    include_boundary_stage: bool,
    rng: random.Random,
) -> SamplingResult:
    """Algorithm 3: SCS, optionally followed by BES on the remaining graph."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if decay < 0:
        raise ValueError("decay must be non-negative")
    if boundary_divisor <= 0:
        raise ValueError("boundary_divisor must be positive")

    frequencies = [0] * graph.vcount()
    vertices = list(range(graph.vcount()))
    stage1 = _frequency_stage(
        graph,
        vertices,
        size=size,
        sample_rate=sample_rate,
        restart_probability=restart_probability,
        walk_length=walk_length,
        threshold=threshold,
        decay=decay,
        frequencies=frequencies,
        rng=rng,
    )

    stage2: List[List[int]] = []
    if include_boundary_stage:
        remaining = [node for node in vertices if frequencies[node] < threshold]
        boundary_size = max(1, size // boundary_divisor)
        stage2 = _frequency_stage(
            graph,
            remaining,
            size=boundary_size,
            sample_rate=sample_rate,
            restart_probability=restart_probability,
            walk_length=walk_length,
            threshold=threshold,
            decay=decay,
            frequencies=frequencies,
            rng=rng,
        )

    if max(frequencies, default=0) > threshold:
        raise AssertionError("frequency sampler exceeded its privacy threshold")
    return SamplingResult(stage1 + stage2, frequencies, len(stage1), len(stage2))


def _released_legacy_walk_from_start(
    graph: igraph.Graph,
    start: int,
    *,
    size: int,
    restart_probability: float,
    max_attempts: int,
    threshold: int,
    decay: float,
    frequencies: Sequence[int],
    python_rng: random.Random,
    numpy_rng: np.random.RandomState,
) -> Optional[List[int]]:
    """Reproduce revision 31166ca's frequency-weighted walk.

    This intentionally differs from Algorithms 1 and 3: the starting vertex
    is not inserted into the sampled set, directed graphs are traversed with
    ``mode="all"``, and ``max_attempts`` counts repeated visits rather than a
    fixed walk length.  It exists only for released-code provenance tests.
    """
    current = start
    visited = set()
    attempts = 0
    while len(visited) < size and attempts < max_attempts:
        if python_rng.random() < restart_probability:
            current = start

        # igraph's historical default graph.neighbors(v) uses all incident
        # edges.  Preserve that behavior even for Email and Bitcoin.
        neighbors = list(set(graph.neighbors(current, mode="all")))
        candidates = [
            node for node in neighbors if frequencies[node] < threshold
        ]
        if not candidates:
            break

        candidate_frequencies = np.asarray(
            [frequencies[node] for node in candidates], dtype=float
        )
        if candidate_frequencies.sum() == 0:
            probabilities = np.ones_like(candidate_frequencies)
        else:
            probabilities = 1.0 / np.power(candidate_frequencies + 1.0, decay)
        probabilities /= probabilities.sum()
        current = int(numpy_rng.choice(candidates, p=probabilities))
        if current not in visited:
            visited.add(current)
            attempts = 0
        else:
            attempts += 1

    return list(visited) if len(visited) == size else None


def _released_legacy_frequency_stage(
    graph: igraph.Graph,
    *,
    size: int,
    target_count: int,
    restart_probability: float,
    max_attempts: int,
    threshold: int,
    decay: float,
    frequencies: List[int],
    python_rng: random.Random,
    numpy_rng: np.random.RandomState,
    candidate_limit: int = 1000,
) -> List[List[int]]:
    """Collect a fixed target count using the released retry semantics."""
    node_sets: List[List[int]] = []
    failed_starts = set()
    while len(node_sets) < target_count:
        candidates = []
        for node, frequency in enumerate(frequencies):
            if frequency != threshold and node not in failed_starts:
                candidates.append(node)
                if len(candidates) >= candidate_limit:
                    break
        if not candidates or min(frequencies, default=threshold) == threshold:
            break

        start = python_rng.choice(candidates)
        nodes = _released_legacy_walk_from_start(
            graph,
            start,
            size=size,
            restart_probability=restart_probability,
            max_attempts=max_attempts,
            threshold=threshold,
            decay=decay,
            frequencies=frequencies,
            python_rng=python_rng,
            numpy_rng=numpy_rng,
        )
        if nodes is None:
            failed_starts.add(start)
            continue
        node_sets.append(nodes)
        for node in nodes:
            frequencies[node] += 1

    return node_sets


def sample_released_legacy(
    graph: igraph.Graph,
    *,
    size: int,
    boundary_divisor: int,
    target_count: int,
    restart_probability: float,
    threshold: int,
    decay: float,
    include_boundary_stage: bool,
    python_rng: random.Random,
    numpy_rng: np.random.RandomState,
    max_attempts: int = 100,
) -> SamplingResult:
    """Reproduce the sampler shipped in revision 31166ca.

    The released implementation repeatedly selected a start from the first
    1,000 eligible vertex IDs until it collected ``q|V|`` subgraphs.  Its BES
    stage reused the original graph and targeted one quarter as many
    half-sized subgraphs.  These choices conflict with paper Algorithms 1/3,
    so callers must label this as a non-strict compatibility diagnostic.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if boundary_divisor <= 0:
        raise ValueError("boundary_divisor must be positive")
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if decay < 0:
        raise ValueError("decay must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    frequencies = [0] * graph.vcount()
    stage1 = _released_legacy_frequency_stage(
        graph,
        size=size,
        target_count=target_count,
        restart_probability=restart_probability,
        max_attempts=max_attempts,
        threshold=threshold,
        decay=decay,
        frequencies=frequencies,
        python_rng=python_rng,
        numpy_rng=numpy_rng,
    )

    stage2: List[List[int]] = []
    if include_boundary_stage:
        # The historical loop used ``count < target_count / 4``.  For an
        # integer counter this is equivalent to ceil(target_count / 4).
        boundary_target = (target_count + 3) // 4
        stage2 = _released_legacy_frequency_stage(
            graph,
            size=max(1, size // boundary_divisor),
            target_count=boundary_target,
            restart_probability=restart_probability,
            max_attempts=max_attempts,
            threshold=threshold,
            decay=decay,
            frequencies=frequencies,
            python_rng=python_rng,
            numpy_rng=numpy_rng,
        )

    if max(frequencies, default=0) > threshold:
        raise AssertionError("released sampler exceeded its frequency threshold")
    return SamplingResult(stage1 + stage2, frequencies, len(stage1), len(stage2))
