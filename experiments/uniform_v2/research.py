"""Independent research Session/result; frozen historical full-dimensional core.

U0 allows synthetic execution only. This is a stage gate, not an adversarial
Python sandbox. Production release is always refused in release.py.
"""
from dataclasses import dataclass
import torch
from experiments.node_pdp_v2_projected.policy import Layout
from experiments.uniform_node_dp.config import Spec as HistoricalSpec
from experiments.uniform_node_dp.research import Session as HistoricalSession
from .config import VERSION, POSITION, validate
from .accounting import account, node_accounts


@dataclass(frozen=True)
class ResearchResult:
    parameters: tuple
    configuration: dict
    diagnostics: dict


class Session:
    def __init__(self, spec, registry, *, model_seed=0):
        self._setup(spec, registry, model_seed, purpose='research')
        self._engine = HistoricalSession(HistoricalSpec(spec.universe, spec.directed), model_seed=model_seed)
        self._remember_initial()

    def _setup(self, spec, registry, seed, *, purpose):
        validate(spec)
        node_accounts(registry, spec)
        if type(seed) is not int or seed < 0:
            raise ValueError('public nonnegative initialization seed required')
        self._spec, self._registry, self._model_seed = spec, registry, seed
        self._fingerprint = registry.fingerprint
        self._purpose, self._used = purpose, False

    @classmethod
    def synthetic(cls, spec, registry, *, noise_seed, model_seed=0):
        obj = object.__new__(cls)
        obj._setup(spec, registry, model_seed, purpose='synthetic')
        obj._engine = HistoricalSession.synthetic(HistoricalSpec(spec.universe, spec.directed),
                                                  noise_seed=noise_seed, model_seed=model_seed)
        obj._remember_initial()
        return obj

    @property
    def model(self):
        return self._engine.model

    @property
    def layout(self):
        return self._engine.layout

    @property
    def model_seed(self):
        return self._model_seed

    def _remember_initial(self):
        # Public initialized model only; never capture random source state.
        self._initial = tuple(p.detach().clone() for p in self.model.parameters())

    def run(self, graph):
        if self._used:
            raise PermissionError('single-use Uniform-v2 Session; no automatic retry')
        self._used = True
        if self._purpose != 'synthetic':
            raise PermissionError('REAL_TRAINING_NOT_AUTHORIZED: U0 synthetic verification only')
        validate(self._spec)
        rows = node_accounts(self._registry, self._spec)
        if self._registry.fingerprint != self._fingerprint:
            raise ValueError('frozen Registry changed')
        params = list(self.model.parameters())
        if (Layout.from_model(self.model) != self.layout or
                sum(p.numel() for p in params if p.requires_grad) != 2340 or
                len(params) != len(self._initial) or
                any(not torch.equal(p, initial) for p, initial in zip(params, self._initial)) or
                self._engine.spec != HistoricalSpec(self._spec.universe, self._spec.directed) or
                self._engine.model_seed != self._model_seed or self._engine.purpose != 'synthetic'):
            raise ValueError('public initialization/full trainable layout changed')
        old = self._engine.run(graph)
        expected = dict(updates=100, noise_calls=100, noise_coordinates=234000,
                        outer_sampling_calls=0, projection_calls=100,
                        all_parameters_finite=True, all_projected_weights_nonnegative=True)
        if old.diagnostics != expected:
            raise RuntimeError('frozen full-dimensional training invariants failed')
        configuration = dict(algorithm=VERSION, position=POSITION, purpose=self._purpose,
                             base={**self._spec.public(), 'model_seed':self._model_seed},
                             registry_fingerprint=self._fingerprint, account=account(), node_accounts=rows,
                             projection=self.layout.public(), publication_allowed=False,
                             machine_dp_certified=False, real_training_authorized=False,
                             privacy_scope='conditional ideal fixed-domain model theorem; excludes diagnostics and preprocessing')
        return ResearchResult(old.parameters, configuration, dict(old.diagnostics))
