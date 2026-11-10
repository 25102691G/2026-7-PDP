"""Independent research Session; identical shared core, no authorized real run.

The synthetic API is for trusted artificial fixtures, not graph-provenance
attestation or a Python security sandbox. No dataset/checkpoint loader here.
"""
from dataclasses import dataclass
import math
import torch
from experiments.node_dp_slots import mechanism as base
from experiments.node_dp_slots.training import execute
from experiments.node_pdp_v2_projected import policy
from .config import VERSION,ROLE,validate
from .accounting import account,node_accounts
from .noise import TestGaussian


@dataclass(frozen=True)
class ResearchResult:
    parameters: tuple
    configuration: dict
    diagnostics: dict


class Session:
    def __init__(self,spec,registry,*,model_seed=0):
        self._setup(spec,registry,model_seed,'research')
        self._noise=None  # No real random source or private query is enabled in U2.

    def _setup(self,spec,registry,seed,purpose):
        validate(spec);node_accounts(registry,spec)
        if type(seed) is not int or seed<0:raise ValueError('public nonnegative initialization seed required')
        self._spec,self._registry,self._model_seed=spec,registry,seed
        self._fingerprint=registry.fingerprint;self._purpose=purpose;self._used=False
        self._model=base.new_model(seed);self._layout=policy.Layout.from_model(self._model)
        self._initial=tuple(p.detach().clone() for p in self._model.parameters())  # PUBLIC initial parameters only.

    @classmethod
    def synthetic(cls,spec,registry,*,noise_seed,model_seed=0):
        obj=object.__new__(cls);obj._setup(spec,registry,model_seed,'synthetic')
        obj._noise=TestGaussian(noise_seed);return obj

    @property
    def model(self):return self._model

    @property
    def layout(self):return self._layout

    @property
    def model_seed(self):return self._model_seed

    def run(self,graph):
        if self._used:raise PermissionError('single-use U2 Session; no retry')
        self._used=True
        if self._purpose!='synthetic':raise PermissionError('REAL_TRAINING_DISABLED: U2 has no real-run authorization')
        validate(self._spec);rows=node_accounts(self._registry,self._spec)
        if self._registry.fingerprint!=self._fingerprint:raise ValueError('frozen Registry changed')
        params=list(self.model.parameters())
        if (policy.Layout.from_model(self.model)!=self.layout or sum(p.numel() for p in params if p.requires_grad)!=2340 or
                len(params)!=len(self._initial) or any(not torch.equal(p,q) for p,q in zip(params,self._initial))):
            raise ValueError('public initial model/full trainable layout changed')
        if type(self._noise) is not TestGaussian:raise PermissionError('explicit artificial source required in U2')
        self.model.train()
        values,frames,_=execute(self._spec,graph,self._noise,model=self.model,
            step=lambda m,s,z:policy.step(m,s,z,self.layout),collector=lambda batch,slots,stats:stats,
            check_slot_bounds=True,check_noise=True)
        if not all(math.isfinite(v) for v in values):raise RuntimeError('nonfinite completed artificial model')
        if self._noise.calls!=100 or len(frames)!=100 or not all(f['nonnegative'] for f in frames):
            raise RuntimeError('frozen execution invariants failed')
        configuration=dict(algorithm=VERSION,purpose='synthetic',role=ROLE,
            base={**self._spec.public(),'model_seed':self.model_seed},registry_fingerprint=self._fingerprint,
            account=account(self._spec),node_accounts=rows,projection=self.layout.public(),
            publication_allowed=False,machine_dp_certified=False,execution_gate='REAL_TRAINING_DISABLED')
        diagnostics=dict(updates=100,noise_calls=100,noise_coordinates=234000,projection_calls=100,
                         outer_sampling_calls=0,all_parameters_finite=True,all_projected_weights_nonnegative=True)
        result=ResearchResult(values,configuration,diagnostics);validate_result(result);return result


def validate_result(result):
    if type(result) is not ResearchResult:raise TypeError('historical/other mechanism is not a U2 result')
    if (result.configuration.get('algorithm')!=VERSION or result.configuration.get('purpose')!='synthetic' or
            len(result.parameters)!=2340 or result.configuration.get('publication_allowed') is not False):
        raise ValueError('invalid U2 scientific result provenance')
