"""Independent immutable configuration; no changes to existing mechanisms."""
from dataclasses import dataclass, field
from experiments.uniform_node_dp.config import Spec as HistoricalSpec

VERSION = 'uniform_v2_matched_architecture_v1'
POSITION = 'POLICY_MATCHED_ARCHITECTURE_CONTROL_UNDER_DESIGN'


@dataclass(frozen=True)
class Spec:
    universe: tuple
    directed: bool = True
    B: int = field(default=16, init=False)
    C: float = field(default=3., init=False)
    K: int = field(default=2, init=False)
    T: int = field(default=100, init=False)
    dimension: int = field(default=2340, init=False)
    s_sum: float = field(default=486., init=False)
    lr: float = field(default=.005, init=False)

    def __post_init__(self):
        HistoricalSpec(self.universe, self.directed)

    def public(self):
        validate(self)
        return dict(HistoricalSpec(self.universe, self.directed).public(),
                    protocol=VERSION, trainable_coordinates=2340,
                    retention_probability=1., position=POSITION)


def validate(spec):
    if type(spec) is not Spec:
        raise TypeError('independent Uniform-v2 Spec required')
    if (spec.B, spec.C, spec.K, spec.T, spec.dimension, spec.s_sum, spec.lr) != (16, 3., 2, 100, 2340, 486., .005):
        raise ValueError('frozen Uniform-v2 constants changed')
    spec.__post_init__()
