"""Uniform research Session: complete input graph, shared100-step mathematical core.

No Bernoulli, production authorization, ledger, release API or benchmark mode.
"""
import math
from experiments.node_dp_slots import mechanism as base
from experiments.node_dp_slots.training import execute
from experiments.node_pdp_v2_projected import policy
from experiments.node_pdp_research import ResearchResult
from .config import validate
from .noise import SecretGaussian,TestGaussian
from .accounting import account


class Session:
    def __init__(self,spec,*,model_seed=0):
        self._setup(spec,model_seed)
        self.purpose='research';self._noise=SecretGaussian()

    def _setup(self,spec,seed):
        validate(spec)
        if type(seed) is not int or seed<0:raise ValueError('public nonnegative model seed required')
        self.spec=spec;self.model_seed=seed;self.model=base.new_model(seed)
        self.layout=policy.Layout.from_model(self.model);self._used=False

    @classmethod
    def synthetic(cls,spec,*,noise_seed,model_seed=0):
        obj=object.__new__(cls);obj._setup(spec,model_seed)
        obj.purpose='synthetic';obj._noise=TestGaussian(noise_seed)
        return obj

    def run(self,graph):
        if self._used:raise PermissionError('single-use research Session; no automatic retry')
        self._used=True;validate(self.spec)
        if self.purpose=='research' and type(self._noise) is not SecretGaussian:
            raise PermissionError('test random sources forbidden for real research')
        if self.purpose=='synthetic' and type(self._noise) is not TestGaussian:
            raise PermissionError('synthetic source required')
        self.model.train()
        values,frames,_=execute(self.spec,graph,self._noise,model=self.model,
            step=lambda m,s,z:policy.step(m,s,z,self.layout),
            collector=lambda batch,slots,stats:stats,check_slot_bounds=True,check_noise=True)
        if not all(math.isfinite(v) for v in values):raise RuntimeError('nonfinite completed model')
        cfg=dict(algorithm='uniform_node_dp',purpose=self.purpose,
            base={**self.spec.public(),'model_seed':self.model_seed},account=account(),
            publication_allowed=False,machine_pdp_certified=False,
            privacy_scope='G0 conditional ideal fixed-domain training; not preprocessing or research diagnostics')
        info=dict(updates=100,noise_calls=self._noise.calls,noise_coordinates=2340*self._noise.calls,
            outer_sampling_calls=0,projection_calls=len(frames),all_parameters_finite=True,
            all_projected_weights_nonnegative=all(f['nonnegative'] for f in frames))
        return ResearchResult(values,cfg,info)
