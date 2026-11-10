"""Public, graph-independent immutable protocol."""
from dataclasses import dataclass
import hashlib


def memberships(identity, seed=0):
    return tuple(2*l + (hashlib.sha256(f'privim-slot-partition-v1|{seed}|{l}|{identity}'.encode()).digest()[0]&1)
                 for l in (0,1))


@dataclass(frozen=True)
class Spec:
    universe: tuple
    directed: bool = True
    B=16; C=3.; T=100; K=2; dimension=2340; s_sum=260.; lr=.005

    def __post_init__(self):
        if type(self.universe) is not tuple or len(set(self.universe))!=len(self.universe):
            raise ValueError('public U must be an immutable unique tuple')
        if any(type(v) is not int or not 0<=v<2**63 for v in self.universe) or type(self.directed) is not bool:
            raise ValueError('invalid public ID domain/direction')

    def public(self):
        return dict(protocol='stable-slot-node-dp-v1-idealized',universe=list(self.universe),directed=self.directed,
                    B=16,C=3.,T=100,K=2,dimension=2340,s_sum=260.,s_mean=16.25,lr=.005,momentum=0.,
                    partition_seed=0,model_seed=0,sampling_seed=0,batch_seed=0,quota=[4]*4,
                    model='GRAT3 legacy_released',features='ones32',loss_lambda=.5,n=40,M_SCS=4,
                    walk_length=200,tau=.3,mu=1.,boundary_divisor=2,Q_per_slot=64,
                    epsilon_target=2.,delta=1e-5,machine_dp_certified=False)
