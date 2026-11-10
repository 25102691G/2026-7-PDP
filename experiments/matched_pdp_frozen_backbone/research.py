"""Independent single-use scientific Session; no graph loader or publication right.

No diagnostics reading private loss, candidates, masks, gradients or RNG state
are returned. Research model outputs still require joint-publication composition.
"""
from dataclasses import dataclass
import math
import threading
import torch
from experiments.node_pdp import sampling
from . import accounting, kernel, layout
from .config import ALGORITHM
from .noise import SecretGaussian


@dataclass(frozen=True)
class ResearchResult:
    parameters: tuple
    configuration: dict
    diagnostics: dict


class Session:
    def __init__(self, registry, spec, *, model_seed=0):
        self._setup(registry,spec,model_seed)
        self.purpose = 'research'
        self._sampler = sampling.SecretBernoulli()
        self._noise = SecretGaussian()

    def _setup(self, registry, spec, seed):
        accounting.node_accounts(registry,spec)
        self.registry, self.spec, self._model_seed = registry, spec, seed
        self.model = layout.new_model(seed)
        self.layout = layout.Layout.from_model(self.model)
        self._initial = {n:p.detach().clone() for n,p in self.model.named_parameters()}
        self._fingerprint = registry.fingerprint
        self._public = spec.public()
        self._used = False
        self._lock = threading.Lock()

    @property
    def model_seed(self):
        """The initialization provenance cannot be relabeled after construction."""
        return self._model_seed

    @classmethod
    def synthetic(cls, registry, spec, *, sampling_seed, noise_seed, model_seed=0):
        from .testing import TestGaussian
        obj = object.__new__(cls)
        obj._setup(registry,spec,model_seed)
        obj.purpose = 'synthetic'
        obj._sampler = sampling.TestBernoulli(sampling_seed)
        obj._noise = TestGaussian(noise_seed)
        return obj

    def run(self, graph):
        from .testing import TestGaussian
        with self._lock:
            if self._used:
                raise PermissionError('single-use research Session; retain failures, no automatic retry')
            self._used = True
            expected = ((sampling.SecretBernoulli,SecretGaussian) if self.purpose == 'research' else
                        (sampling.TestBernoulli,TestGaussian) if self.purpose == 'synthetic' else None)
            if expected is None or (type(self._sampler),type(self._noise)) != expected or self._noise.calls:
                raise PermissionError('test/injected/reused sources forbidden in normal scientific path')
            accounting.node_accounts(self.registry,self.spec)
            if self.registry.fingerprint != self._fingerprint or self.spec.public() != self._public:
                raise ValueError('public protocol changed after initialization')
            self.layout.parameters(self.model)
            if any(not torch.equal(p,self._initial[n]) for n,p in self.model.named_parameters()):
                raise ValueError('model must start at the unmodified public initialization')
        sampling.validate_graph(graph,self.spec)
        induced = sampling.sample_once(graph,self.registry,self._sampler)
        self.model.train()
        values = kernel.execute(self.spec,induced.graph,self._noise,self.model)
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError('nonfinite completed research model')
        cfg = dict(algorithm=ALGORITHM,purpose=self.purpose,base=self.spec.public(),
                   model_seed=self.model_seed,layout=self.layout.public(),
                   registry=self.registry.public(),registry_fingerprint=self._fingerprint,
                   account=accounting.base_account(self.spec),publication_allowed=False,
                   machine_pdp_certified=False,
                   privacy_scope='conditional ideal fixed-domain model only; not development or private diagnostics')
        info = dict(updates=100,noise_calls=self._noise.calls,noise_coordinates=33*self._noise.calls,
                    outer_sampling_calls=1,projection_calls=100,frozen_backbone_unchanged=True,
                    all_projected_weights_nonnegative=True,all_parameters_finite=True)
        return ResearchResult(values,cfg,info)
