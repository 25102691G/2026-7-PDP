"""Candidate-specific1283-dimensional contract; never a2340 Spec subtype."""
from dataclasses import dataclass
from experiments.node_pdp.registry import Registry


@dataclass(frozen=True)
class Spec:
    universe: tuple
    directed: bool = True
    B=16; C=3.; K=2; T=100; dimension=1283; s_sum=260.; lr=.005

    def __post_init__(self):
        if (type(self.universe) is not tuple or any(type(v) is not int or not 0<=v<2**63 for v in self.universe)
            or tuple(sorted(set(self.universe)))!=self.universe or type(self.directed) is not bool):
            raise ValueError('fixed canonical public identity tuple and direction required')

    def public(self):
        return dict(protocol='rowsum-pdp-c2-idealized-v1',universe=list(self.universe),directed=self.directed,
            B=16,C=3.,K=2,T=100,dimension=1283,s_sum=260.,lr=.005,momentum=0.,delta=1e-5,
            features='ones32',loss_lambda=.5,model='GRAT3 legacy_released row-sum parameterization',
            n=40,M_SCS=4,walk_length=200,tau=.3,mu=1.,boundary_divisor=2,Q_per_slot=64,
            partition_seed=0,sampling_seed=0,batch_seed=0,quota=[4]*4,
            projection_coordinates=32,initial_projection=False,publication_allowed=False,
            machine_pdp_certified=False,real_execution_enabled=False)


def validate(registry,spec):
    if type(spec) is not Spec or type(registry) is not Registry:raise TypeError('independent row-sum Spec and fixed Registry required')
    spec.__post_init__()
    if registry.universe!=spec.universe:raise ValueError('fixed identical public U required')
    if (spec.B,spec.C,spec.K,spec.T,spec.dimension,spec.s_sum,spec.lr)!=(16,3.,2,100,1283,260.,.005):
        raise ValueError('frozen row-sum mechanism changed')
