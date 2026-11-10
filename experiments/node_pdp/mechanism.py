"""Single-use outer composition; the base model/training mathematics are unchanged."""
from dataclasses import dataclass
import threading

from experiments.node_dp_slots import mechanism as base
from experiments.node_dp_slots.noise import SecretGaussian
from . import accounting, sampling


class ExecutionGap(RuntimeError):
    """No safe failure distribution is claimed; message contains no private fields."""


class _PrechargedTicket:
    """Acknowledge exactly one already charged base model, never reset/refund it."""
    def __init__(self, ledger):
        self._ledger = ledger
        self._consumed = False
        self._lock = threading.Lock()

    def charge(self):
        with self._lock:
            if self._consumed or self._ledger.models != 1:
                raise PermissionError('invalid or reused base reservation')
            self._consumed = True


class _SingleUseGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._used = False

    def reserve(self, engine):
        with self._lock:
            if self._used:
                raise PermissionError('single-model PDP authorization already consumed')
            self._used = True
            # Existing process-wide base ledger is charged BEFORE private access.
            try:
                engine._ledger.charge()
            except Exception:
                raise PermissionError('base model budget unavailable') from None
            engine._ledger = _PrechargedTicket(engine._ledger)


_PROCESS_GUARD = _SingleUseGuard()


class _Receipt:
    def __init__(self):
        self.lock = threading.Lock()
        self.run = None
        self.published = False


@dataclass(frozen=True)
class Run:
    _base_run: object
    _registry: object
    _receipt: object
    _registry_fingerprint: str

    @property
    def testing_diagnostics(self):
        return self._base_run.testing_diagnostics


class Engine:
    def __init__(self, *args, **kwargs):
        raise TypeError('use privacy(registry,spec) or explicit testing constructor')

    @classmethod
    def _construct(cls, registry, spec):
        accounting.node_accounts(registry, spec)
        obj = object.__new__(cls)
        obj._registry = registry
        obj._spec = spec
        obj._fingerprint = registry.fingerprint
        obj._lock = threading.Lock()
        obj._used = False
        return obj

    @classmethod
    def privacy(cls, registry, spec):
        obj = cls._construct(registry, spec)
        obj._mode = 'privacy'
        obj._sampler = sampling.SecretBernoulli()
        obj._base = base.Engine.privacy(spec)  # Default shared ledger, no caller reset API.
        return obj

    @classmethod
    def testing(cls, registry, spec, *, sampling_seed, noise_seed):
        obj = cls._construct(registry, spec)
        obj._mode = 'testing'
        obj._sampler = sampling.TestBernoulli(sampling_seed)
        obj._base = base.Engine.testing(spec, noise_seed=noise_seed)
        return obj

    def run(self, graph):
        with self._lock:
            if self._used:
                raise PermissionError('outer mechanism is single use')
            self._used = True
            if self._mode == 'privacy':
                if (type(self._sampler) is not sampling.SecretBernoulli or
                    type(self._base) is not base.Engine or self._base._mode != 'privacy' or
                    type(self._base._source) is not SecretGaussian):
                    raise PermissionError('testing or injected random source forbidden')
                _PROCESS_GUARD.reserve(self._base)
        try:
            accounting.validate_public(self._registry, self._spec)
            if self._registry.fingerprint != self._fingerprint or self._base.spec != self._spec:
                raise ValueError('public configuration changed')
            sampling.validate_graph(graph, self._spec)
            induced = sampling.sample_once(graph, self._registry, self._sampler)
            completed = self._base.run(induced.graph)
        except Exception:
            # Suppress backend messages, not the unresolved fact/timing of failure.
            raise ExecutionGap('UNRESOLVED execution failure; reservation consumed; no publishable result') from None
        receipt = _Receipt()
        result = Run(completed, self._registry, receipt, self._fingerprint)
        receipt.run = result
        return result
