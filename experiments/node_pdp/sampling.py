"""Hidden one-shot coins and internal induced input. These objects are NOT releases."""
from dataclasses import dataclass
import math
import random

from experiments.node_dp_slots.adapter import Graph


def _threshold(p):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('probability must be in [0,1]')
    return int(p * (1 << 64))


class SecretBernoulli:
    def __init__(self):
        self.__rng = random.SystemRandom()

    def draw(self, p):
        return self.__rng.getrandbits(64) < _threshold(p)


class TestBernoulli:
    def __init__(self, seed):
        self.__rng = random.Random(seed)

    def draw(self, p):
        return self.__rng.getrandbits(64) < _threshold(p)


@dataclass(frozen=True)
class _Induced:
    graph: Graph
    local_to_stable: tuple

    @property
    def stable_to_local(self):
        return {v: i for i, v in enumerate(self.local_to_stable)}


def validate_graph(graph, spec):
    if type(graph) is not Graph:
        raise ValueError('typed graph required')
    ids = set(graph.nodes)
    if len(ids) != len(graph.nodes) or not ids <= set(spec.universe) or graph.directed != spec.directed:
        raise ValueError('graph outside public input domain')
    edges = graph.edges if graph.directed else tuple(tuple(sorted(e)) for e in graph.edges)
    if len(set(edges)) != len(edges) or any(u == v or u not in ids or v not in ids for u, v in edges):
        raise ValueError('simple induced graph domain required')


def induce(graph, retained):
    selected = set(retained)
    if not selected <= set(graph.nodes):
        raise ValueError('induction cannot add identities')
    ids = tuple(sorted(selected))
    edges = tuple(e for e in graph.edges if e[0] in selected and e[1] in selected)
    if not graph.directed: edges = tuple(tuple(sorted(e)) for e in edges)
    return _Induced(Graph(ids, tuple(sorted(edges)), graph.directed), ids)


def sample_once(graph, registry, source):
    retained = [v for v in sorted(graph.nodes) if source.draw(registry.entry(v).retention_probability)]
    return induce(graph, retained)
