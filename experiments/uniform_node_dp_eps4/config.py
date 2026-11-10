"""Independent frozen relaxed Uniform configuration; old Spec is untouched."""
from dataclasses import dataclass
from experiments.uniform_node_dp.config import Spec as StrictSpec

VERSION='uniform_node_dp_eps4_v1'
ROLE='RELAXED_PRIVACY_REFERENCE'
REAL_TRAINING_DISABLED=True


@dataclass(frozen=True)
class Spec:
    universe: tuple
    directed: bool=True
    target_epsilon: float=4.
    target_delta: float=1e-5
    s_sum: float=139.
    p_v: float=1.
    T: int=100
    B: int=16
    C: float=3.
    K: int=2
    sensitivity_sum: float=12.
    dimension: int=2340
    lr: float=.005
    momentum: float=0.
    projection: str='post-update final32 nonnegative'
    initial_projection: bool=False

    def __post_init__(self):validate(self)

    def public(self):
        validate(self)
        return dict(StrictSpec(self.universe,self.directed).public(),protocol=VERSION,
                    target_epsilon=self.target_epsilon,epsilon_target=self.target_epsilon,
                    target_delta=self.target_delta,delta=self.target_delta,s_sum=self.s_sum,s_mean=self.s_sum/16,
                    p_v=1.,sensitivity_sum=12.,trainable_coordinates=2340,role=ROLE)


def validate(spec):
    if type(spec) is not Spec:raise TypeError('independent Uniform epsilon4 Spec required')
    expected=dict(target_epsilon=4.,target_delta=1e-5,s_sum=139.,p_v=1.,T=100,B=16,C=3.,K=2,
                  sensitivity_sum=12.,dimension=2340,lr=.005,momentum=0.,
                  projection='post-update final32 nonnegative',initial_projection=False)
    if any(getattr(spec,key)!=value for key,value in expected.items()):
        raise ValueError('frozen Uniform epsilon4 mechanism changed')
    StrictSpec(spec.universe,spec.directed)  # Same fixed public identity-domain validation.
