"""Frozen G0 Uniform configuration; deliberately not a PDP Spec subtype."""
from dataclasses import dataclass
from experiments.node_dp_slots.config import Spec as PDPSpec


@dataclass(frozen=True)
class Spec:
    universe: tuple
    directed: bool = True
    B=16; C=3.; K=2; T=100; dimension=2340; s_sum=486.; lr=.005

    def __post_init__(self):
        PDPSpec(self.universe,self.directed)  # Reuse only public identity validation.

    def public(self):
        return dict(PDPSpec(self.universe,self.directed).public(),
                    protocol='uniform-node-dp-g0-idealized-v1',s_sum=486.,s_mean=30.375,
                    epsilon_target=1.,delta=1e-5,outer_bernoulli=False,
                    projection='post-update final32 nonnegative',initial_projection=False)


def validate(spec):
    if type(spec) is not Spec:raise TypeError('independent Uniform Spec required')
    if (spec.B,spec.C,spec.K,spec.T,spec.dimension,spec.s_sum,spec.lr)!=(16,3.,2,100,2340,486.,.005):
        raise ValueError('frozen G0 Uniform configuration changed')
    spec.__post_init__()
