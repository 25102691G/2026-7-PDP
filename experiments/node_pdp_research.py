"""Ordinary scientific v1/v2 training using the same kernel as formal wrappers.

No publication permission, Ledger, ticket or authorization service is created here.
"""
from dataclasses import dataclass
import math
from .node_dp_slots import mechanism as base
from .node_dp_slots.training import execute
from .node_dp_slots.noise import SecretGaussian,TestGaussian
from .node_pdp import sampling,accounting
from .node_pdp_v2_projected import policy,diagnostics

ALGORITHMS=('node_pdp_v1','node_pdp_v2_projected')


@dataclass(frozen=True)
class ResearchResult:
    parameters: tuple
    configuration: dict
    diagnostics: dict


class Session:
    def __init__(self,registry,spec,algorithm,*,model_seed=0):
        self._setup(registry,spec,algorithm,model_seed)
        self.purpose='research';self._sampler=sampling.SecretBernoulli();self._noise=SecretGaussian()

    def _setup(self,registry,spec,algorithm,seed):
        accounting.validate_public(registry,spec)
        if algorithm not in ALGORITHMS or type(seed) is not int or seed<0:raise ValueError('supported algorithm/public model seed required')
        self.registry=registry;self.spec=spec;self.algorithm=algorithm;self.model_seed=seed
        self.model=base.new_model(seed);self.layout=policy.Layout.from_model(self.model);self._used=False

    @classmethod
    def synthetic(cls,registry,spec,algorithm,*,sampling_seed,noise_seed):
        obj=object.__new__(cls);obj._setup(registry,spec,algorithm,0)
        obj.purpose='synthetic';obj._sampler=sampling.TestBernoulli(sampling_seed);obj._noise=TestGaussian(noise_seed)
        return obj

    def run(self,graph):
        if self._used:raise PermissionError('session already executed; create a distinct scientific experiment')
        self._used=True
        if self.purpose=='research' and (type(self._sampler) is not sampling.SecretBernoulli or type(self._noise) is not SecretGaussian):
            raise PermissionError('test random sources forbidden for real research')
        sampling.validate_graph(graph,self.spec)
        induced=sampling.sample_once(graph,self.registry,self._sampler)
        projected=self.algorithm=='node_pdp_v2_projected'
        step=(lambda model,signal,source:policy.step(model,signal,source,self.layout)) if projected else None
        self.model.train()
        values,frames,_=execute(self.spec,induced.graph,self._noise,model=self.model,step=step,
            collector=lambda batch,slots,stats:dict(diagnostics.slot_summary(batch,slots),**stats),
            check_slot_bounds=True,check_noise=True)
        if not all(math.isfinite(v) for v in values):raise RuntimeError('nonfinite scientific model')
        config=dict(algorithm=self.algorithm,purpose=self.purpose,base={**self.spec.public(),'model_seed':self.model_seed},
            registry_fingerprint=self.registry.fingerprint,publication_allowed=False,machine_pdp_certified=False,
            privacy_scope='idealized frozen training kernel; not full research/development/diagnostic publication')
        info=dict(updates=100,noise_calls=self._noise.calls,noise_coordinates=2340*self._noise.calls,outer_sampling_calls=1,
            all_parameters_finite=True,steps=frames,projection_calls=100 if projected else 0,
            all_projected_weights_nonnegative=all(f.get('nonnegative',False) for f in frames) if projected else None,
            actual_zero_gradient_slot_definition=diagnostics.definitions()['actual_zero_gradient_slots'])
        return ResearchResult(values,config,info)
