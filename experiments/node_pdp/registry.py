"""Immutable public preferences: never infer the universe from a private graph."""
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import math
import random

from experiments.node_dp_slots.config import Spec

VERSION = 'node-pdp-budget-v1'


@dataclass(frozen=True)
class Entry:
    identity: int
    epsilon_target: int
    delta_target: float
    retention_probability: float


@dataclass(frozen=True)
class Registry:
    universe: tuple
    budgets: tuple
    source: str
    version: str = VERSION

    def __post_init__(self):
        Spec(self.universe)  # Validate the PUBLIC domain, not actual membership.
        if self.version != VERSION or type(self.source) is not str or not self.source.strip():
            raise ValueError('supported version and external public provenance required')
        if type(self.budgets) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 or
            type(pair[0]) is not int or type(pair[1]) is not int or pair[1] not in (1, 2, 4)
            for pair in self.budgets
        ):
            raise ValueError('immutable stable-ID budget pairs in {1,2,4} required')
        if len(self.budgets) != len(self.universe) or {v for v, _ in self.budgets} != set(self.universe):
            raise ValueError('exactly one public budget for every identity in U required')
        object.__setattr__(self, 'universe', tuple(sorted(self.universe)))
        object.__setattr__(self, 'budgets', tuple(sorted(self.budgets)))

    def entry(self, identity):
        if type(identity) is not int:
            raise KeyError('unknown public stable identity')
        i = bisect_left(self.universe, identity)
        if i == len(self.universe) or self.universe[i] != identity:
            raise KeyError('unknown public stable identity')
        target = self.budgets[i][1]
        p = min(1., math.expm1(target) / math.expm1(2.))
        return Entry(identity, target, 1e-5, p)

    def public(self):
        return dict(version=self.version, source=self.source, universe=list(self.universe),
                    calibration_epsilon=2., entries=[vars(self.entry(v)) for v in self.universe])

    @property
    def fingerprint(self):
        blob = json.dumps(self.public(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def from_public_quotas(cls, universe, *, public_seed, source):
        """Optional public preparation helper: exact quotas on U, never V(G)."""
        Spec(universe)
        if type(public_seed) is not int:
            raise ValueError('public integer seed required')
        ids = tuple(sorted(universe)); n = len(ids)
        weights = (1, 2, 2)
        counts = [n*w//5 for w in weights]
        order = sorted(range(3), key=lambda i: (-(n*weights[i] % 5), i))
        for i in order[:n-sum(counts)]: counts[i] += 1
        labels = [e for e, count in zip((1, 2, 4), counts) for _ in range(count)]
        random.Random(public_seed).shuffle(labels)  # PUBLIC preferences, not secret coins.
        return cls(ids, tuple(zip(ids, labels)), source=source)
