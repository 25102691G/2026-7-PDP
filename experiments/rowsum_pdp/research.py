"""C2 artificial Session only; no real loader, production ledger or release permission."""
from dataclasses import dataclass
import math
from experiments.node_pdp import sampling
from . import accounting
from .config import validate
from .model import Model
from .noise import SecretGaussian,TestGaussian
from .training import execute


@dataclass(frozen=True)
class ArtificialResult:
    phi: tuple
    configuration: dict
    diagnostics: dict


class Session:
    def __init__(self,registry,spec,*,model_seed=0):
        self._setup(registry,spec,model_seed)
        self._purpose='research-disabled-C2';self._sampler=sampling.SecretBernoulli();self._noise=SecretGaussian()

    def _setup(self,registry,spec,seed):
        validate(registry,spec);accounting.node_accounts(registry,spec)
        self.registry=registry;self.spec=spec;self.model_seed=seed;self.model=Model(seed);self._used=False
        self._registry_fingerprint=registry.fingerprint

    @classmethod
    def synthetic(cls,registry,spec,*,model_seed,sampling_seed,noise_seed):
        if len(spec.universe)>256:raise ValueError('C2 artificial domain limited to256 identities')
        obj=object.__new__(cls);obj._setup(registry,spec,model_seed)
        obj._purpose='synthetic';obj._sampler=sampling.TestBernoulli(sampling_seed);obj._noise=TestGaussian(noise_seed)
        return obj

    def run(self,graph):
        # The frozen q1 empty-edge identity is contradicted by the original backend.
        # Preserve the failed candidate; do not silently change its algorithm.
        raise PermissionError('C2 BLOCKED: structural equivalence fails on nonempty edgeless graphs')
        if self._used:raise PermissionError('single-use candidate Session; no retry')
        self._used=True
        if self._purpose!='synthetic':
            if type(self._noise) is not SecretGaussian or type(self._sampler) is not sampling.SecretBernoulli:
                raise PermissionError('test/injected random sources forbidden for research')
            raise PermissionError('C2 permits artificial execution only; C3 not enabled')
        if type(self._noise) is not TestGaussian or type(self._sampler) is not sampling.TestBernoulli:raise PermissionError('explicit artificial sources required')
        validate(self.registry,self.spec)
        if self.registry.fingerprint!=self._registry_fingerprint:raise ValueError('fixed Registry changed')
        sampling.validate_graph(graph,self.spec)
        if len(graph.nodes)>128 or len(graph.edges)>4096:raise ValueError('C2 small artificial graphs only')
        induced=sampling.sample_once(graph,self.registry,self._sampler)
        self.model.train();diagnostics=execute(self.spec,induced.graph,self.model,self._noise)
        values=tuple(float(x) for x in self.model.vector())
        if not all(math.isfinite(x) for x in values):raise RuntimeError('nonfinite candidate parameters')
        diagnostics['outer_sampling_calls']=1
        config=dict(algorithm='rowsum_pdp_c2',purpose='synthetic',model_seed=self.model_seed,base=self.spec.public(),
                    registry_fingerprint=self._registry_fingerprint,publication_allowed=False,account=accounting.base_account(self.spec))
        return ArtificialResult(values,config,diagnostics)
