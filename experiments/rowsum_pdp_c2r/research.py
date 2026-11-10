"""Explicit small artificial graphs and test streams only; old C2 stays quarantined."""
from dataclasses import dataclass
import math
from experiments.node_pdp import sampling
from experiments.rowsum_pdp import accounting,training
from experiments.rowsum_pdp.config import validate
from experiments.rowsum_pdp.noise import TestGaussian
from .model import Model


@dataclass(frozen=True)
class ArtificialResult:
    phi: tuple
    configuration: dict
    diagnostics: dict


class Session:
    def __init__(self,*args,**kwargs):
        raise PermissionError('C2-R permits explicit artificial execution only; real/C3 execution is not authorized')

    @classmethod
    def synthetic(cls,registry,spec,*,model_seed,sampling_seed,noise_seed):
        validate(registry,spec);accounting.node_accounts(registry,spec)
        if len(spec.universe)>256:raise ValueError('C2-R artificial domain limited to256 identities')
        obj=object.__new__(cls)
        obj.registry=registry;obj.spec=spec;obj.model_seed=model_seed;obj.model=Model(model_seed)
        obj._registry_fingerprint=registry.fingerprint;obj._used=False
        obj._sampler=sampling.TestBernoulli(sampling_seed);obj._noise=TestGaussian(noise_seed)
        return obj

    def run(self,graph):
        # Final C2-R acceptance failed: full regressions exceeded the bounded budget.
        raise PermissionError('C2R_FAILED: candidate rejected; artificial execution quarantined')
        if self._used:raise PermissionError('single-use artificial Session; no retry')
        self._used=True
        if type(self._noise) is not TestGaussian or type(self._sampler) is not sampling.TestBernoulli:
            raise PermissionError('explicit test sources required; no research/secret source accepted')
        validate(self.registry,self.spec)
        if self.registry.fingerprint!=self._registry_fingerprint:raise ValueError('fixed Registry changed')
        sampling.validate_graph(graph,self.spec)
        if len(graph.nodes)>128 or len(graph.edges)>4096:raise ValueError('small artificial graphs only')
        induced=sampling.sample_once(graph,self.registry,self._sampler)
        self.model.train();diagnostics=training.execute(self.spec,induced.graph,self.model,self._noise)
        values=tuple(float(x) for x in self.model.vector())
        if not all(math.isfinite(x) for x in values):raise RuntimeError('nonfinite parameters')
        diagnostics['outer_sampling_calls']=1
        base=dict(self.spec.public(),protocol='rowsum-pdp-c2r-piecewise-v1')
        config=dict(algorithm='rowsum_pdp_c2r_piecewise_v1',purpose='synthetic',model_seed=self.model_seed,
                    base=base,registry_fingerprint=self._registry_fingerprint,publication_allowed=False,
                    account=accounting.base_account(self.spec))
        return ArtificialResult(values,config,diagnostics)
