"""PDP-v2-Avg research candidate: read-only mean of all100 processed states.

No training loop, model update, sampling source or publication API is added.
The unchanged ordinary v2 Session supplies the entire actual trajectory.
"""
import copy
from dataclasses import dataclass
import torch
from experiments.node_pdp_research import Session
from experiments.node_pdp_v2_projected import policy
from experiments.pdp_research_observer import TrainingObserver


def task_matrix():
    return [dict(id=f'{d}_seed{s}',dataset=d,model_seed=s,status='NOT_ATTEMPTED')
            for d in ('email','bitcoin','lastfm','hepph','facebook') for s in range(5)]


class ParameterMean:
    """Fixed100 observations, O(d) storage. Does not read initial parameter values."""
    def __init__(self,model):
        self.layout=tuple((n,tuple(p.shape),p.dtype,p.device) for n,p in model.named_parameters())
        if sum(p.numel() for p in model.parameters())!=2340:raise ValueError('fixed2340-coordinate model required')
        self.total=torch.zeros(2340,dtype=torch.float64);self.count=0

    def observe(self,model):
        if self.count>=100:raise RuntimeError('fixed100 post-update observations only')
        if self.layout!=tuple((n,tuple(p.shape),p.dtype,p.device) for n,p in model.named_parameters()):
            raise RuntimeError('parameter layout changed')
        with torch.no_grad():
            state=torch.cat([p.detach().reshape(-1).double() for p in model.parameters()])
            if not bool(torch.isfinite(state).all()):raise RuntimeError('processed parameters must be finite')
            self.total.add_(state);self.count+=1

    def finish(self):
        if self.count!=100:raise RuntimeError('incomplete fixed100 mean refused')
        mean=(self.total/100).to(dtype=self.layout[0][2])
        if not bool(torch.isfinite(mean).all()):raise RuntimeError('nonfinite parameter mean')
        return mean


class MeanObserver(TrainingObserver):
    def __init__(self,accumulator):
        super().__init__();self.accumulator=accumulator;self._step_code=policy.step.__code__

    def _observe(self,frame,event,returned):
        super()._observe(frame,event,returned)
        if event=='return' and frame.f_code is self._step_code:
            if not isinstance(returned,dict) or not returned.get('nonnegative',False):
                raise RuntimeError('processed/projected step did not complete')
            self.accumulator.observe(frame.f_locals['model'])


@dataclass(frozen=True)
class AverageResult:
    last: object
    average_parameters: tuple
    checks: dict
    training_observation: dict


class AverageSession(Session):
    """Independent candidate; never substitutes for the original v2 Session."""
    def _setup(self,registry,spec,algorithm,seed):
        if algorithm!='node_pdp_v2_projected':raise ValueError('only original projected v2 trajectory supported')
        super()._setup(registry,spec,algorithm,seed)

    def run(self,graph):
        accumulator=ParameterMean(self.model);observer=MeanObserver(accumulator)
        with observer:last=super().run(graph)
        average=accumulator.finish()
        before=tuple(float(x) for p in self.model.parameters() for x in p.detach().reshape(-1))
        if before!=last.parameters:raise RuntimeError('original final state changed')
        # Separate model only for deterministic inference; no reinitialization or update.
        self.average_model=copy.deepcopy(self.model)
        with torch.no_grad():
            for p,chunk in zip(self.average_model.parameters(),average.split([p.numel() for p in self.average_model.parameters()])):
                p.copy_(chunk.reshape(p.shape));p.grad=None
        unchanged=before==tuple(float(x) for p in self.model.parameters() for x in p.detach().reshape(-1))
        layout=policy.Layout.from_model(self.average_model)
        nonnegative=bool((layout.target(self.average_model)>=0).all())
        if not unchanged or not nonnegative:raise RuntimeError('mean candidate invariant failed')
        checks=dict(candidate='PDP-v2-Avg',count=100,initial_state_included=False,weights='uniform1/100',
            arithmetic='sequential FP64 sum of post-update/repair/projection states; divide100; original dtype',
            last_unchanged=unchanged,average_head_nonnegative=nonnegative,all_average_parameters_finite=True,
            original_parameters_retained=True,intermediate_checkpoints_saved=0,publication_allowed=False,
            same_trajectory_joint_postprocessing=True,machine_certified=False)
        return AverageResult(last,tuple(float(x) for x in average),checks,observer.summary())
