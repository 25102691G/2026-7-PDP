"""Immutable public configuration; deliberately not a production PDP Spec."""
from dataclasses import dataclass, field
from experiments.node_dp_slots.config import Spec as OriginalSpec

ALGORITHM = 'matched_pdp_frozen_backbone_v1'


@dataclass(frozen=True)
class Spec:
    universe: tuple
    directed: bool = True
    B: int = field(default=16, init=False)
    C: float = field(default=3., init=False)
    K: int = field(default=2, init=False)
    T: int = field(default=100, init=False)
    dimension: int = field(default=33, init=False)
    s_sum: float = field(default=260., init=False)
    lr: float = field(default=.005, init=False)

    def __post_init__(self):
        OriginalSpec(self.universe, self.directed)  # Public identity validation only.
        if self.universe != tuple(sorted(self.universe)):
            raise ValueError('canonical sorted public U required')

    def public(self):
        result = OriginalSpec(self.universe, self.directed).public()
        # Existing fixed sample/batch streams are public role hashes, not secret coins.
        result.pop('sampling_seed')
        result.pop('batch_seed')
        result.pop('model_seed')
        result.pop('epsilon_target')
        result.update(protocol=ALGORITHM, dimension=33, full_model_dimension=2340,
                      frozen_coordinates=2307, outer_bernoulli='heterogeneous-once',
                      public_slot_streams='node-dp-slots-v1 role/slot hashes',
                      budget_targets=[1, 2, 4], calibration_epsilon=2.,
                      projection='post-update final32 nonnegative', initial_projection=False,
                      publication_allowed=False)
        return result


def validate(spec):
    if type(spec) is not Spec:
        raise TypeError('independent frozen-backbone Spec required')
    spec.__post_init__()
    if (spec.B, spec.C, spec.K, spec.T, spec.dimension, spec.s_sum, spec.lr) != (16,3.,2,100,33,260.,.005):
        raise ValueError('frozen 33D protocol changed')
