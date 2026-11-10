"""Independent v2 loop; original query/update primitives, plus post-update projection."""
from dataclasses import dataclass
import threading
import math
import torch
from experiments.node_dp_slots import mechanism as base
from experiments.node_dp_slots.noise import SecretGaussian,TestGaussian
from experiments.node_pdp import mechanism as v1,sampling
from . import policy,accounting,diagnostics


@dataclass(frozen=True)
class Run:
    _parameters: tuple
    _spec: object
    _registry: object
    _layout: policy.Layout
    _mode: str
    _completed: bool
    _receipt: object
    _registry_fingerprint: str
    testing_diagnostics: object=None


class Engine:
    def __init__(self,*args,**kwargs):raise TypeError('use explicit privacy or testing constructor')

    @classmethod
    def _construct(cls,registry,spec):
        accounting.node_accounts(registry,spec)
        obj=object.__new__(cls)
        obj._registry=registry;obj._spec=spec;obj._fingerprint=registry.fingerprint
        obj._model=base.new_model()  # Same public seed0 rule, not projected at construction.
        obj._layout=policy.Layout.from_model(obj._model)
        obj._used=False;obj._lock=threading.Lock()
        return obj

    @classmethod
    def privacy(cls,registry,spec):
        obj=cls._construct(registry,spec);obj._mode='privacy'
        obj._sampler=sampling.SecretBernoulli();obj._carrier=base.Engine.privacy(spec)
        return obj

    @classmethod
    def testing(cls,registry,spec,*,sampling_seed,noise_seed):
        obj=cls._construct(registry,spec);obj._mode='testing'
        obj._sampler=sampling.TestBernoulli(sampling_seed)
        obj._carrier=base.Engine.testing(spec,noise_seed=noise_seed)
        return obj

    def run(self,graph):
        with self._lock:
            if self._used:raise PermissionError('v2 engine is single use')
            self._used=True
            if self._mode=='privacy':
                if (type(self._sampler) is not sampling.SecretBernoulli or
                    self._carrier._mode!='privacy' or type(self._carrier._source) is not SecretGaussian):
                    raise PermissionError('testing/injected sources forbidden')
                # SAME v1 process guard and SAME default base ledger: version is no budget reset.
                v1._PROCESS_GUARD.reserve(self._carrier)
            elif type(self._sampler) is not sampling.TestBernoulli or type(self._carrier._source) is not TestGaussian:
                raise PermissionError('explicit isolated testing sources required')
        try:
            accounting.node_accounts(self._registry,self._spec)
            if self._registry.fingerprint!=self._fingerprint or self._carrier.spec!=self._spec:
                raise ValueError('frozen public configuration changed')
            sampling.validate_graph(graph,self._spec)
            induced=sampling.sample_once(graph,self._registry,self._sampler)
            if self._mode=='privacy':self._carrier._ledger.charge()  # Acknowledge precharge, never charge twice.
            from experiments.node_dp_slots.training import execute
            collect=(lambda batch,slots,stats:dict(diagnostics.slot_summary(batch,slots),**stats)) if self._mode=='testing' else None
            values,steps,_=execute(self._spec,induced.graph,self._carrier._source,model=self._model,
                step=lambda model,signal,source:policy.step(model,signal,source,self._layout),
                collector=collect,check_slot_bounds=True,check_noise=True)
            if not all(math.isfinite(v) for v in values):raise RuntimeError('nonfinite completed parameters')
        except Exception:
            raise v1.ExecutionGap('UNRESOLVED projected execution gap; authorization consumed; no retry') from None
        diagnostic=None
        if self._mode=='testing':
            diagnostic=dict(updates=100,noise_calls=self._carrier._source.calls,noise_coordinates=2340*self._carrier._source.calls,
                projection_calls=len(steps),all_projected_weights_nonnegative=all(s['nonnegative'] for s in steps),
                all_parameters_finite=all(math.isfinite(v) for v in values),steps=steps,
                zero_gradient_slot_definition=diagnostics.definitions()['actual_zero_gradient_slots'])
        receipt=v1._Receipt()
        result=Run(values,self._spec,self._registry,self._layout,self._mode,True,receipt,self._fingerprint,diagnostic)
        receipt.run=result
        return result
