"""Original noisy update/repair followed by exactly one fixed parameter projection."""
from dataclasses import dataclass
import torch
from experiments.node_dp_slots import mechanism as base

TARGET = 'grat.grat3.fc.weight'


@dataclass(frozen=True)
class Part:
    name: str
    shape: tuple
    start: int
    stop: int


@dataclass(frozen=True)
class Layout:
    parts: tuple
    target_name: str
    shape: tuple
    start: int
    stop: int
    total: int

    @classmethod
    def from_model(cls, model):
        parts=[];offset=0
        for name,value in model.named_parameters():
            parts.append(Part(name,tuple(value.shape),offset,offset+value.numel()))
            offset+=value.numel()
        target=[p for p in parts if p.name==TARGET]
        if offset!=2340 or len(target)!=1 or target[0].shape!=(1,32) or list(model.buffers()):
            raise ValueError('public GRAT parameter layout changed')
        if any(p.dtype!=torch.float32 or p.device.type!='cpu' for p in model.parameters()):
            raise ValueError('frozen CPU/FP32 model required')
        t=target[0]
        return cls(tuple(parts),t.name,t.shape,t.start,t.stop,offset)

    def target(self, model):
        if Layout.from_model(model)!=self:
            raise ValueError('parameter registration/order changed')
        return dict(model.named_parameters())[self.target_name]

    def public(self):
        return dict(target_name=self.target_name,shape=list(self.shape),start=self.start,stop=self.stop,
                    indexing='zero-based, stop exclusive; named_parameters registration order',
                    total_coordinates=self.total,projected_coordinates=self.stop-self.start,
                    initial_projection=False,after_step_range=[1,100],
                    policy='max(0,w3) after original FP32 update and finite repair',
                    parameters=[dict(name=p.name,shape=list(p.shape),start=p.start,stop=p.stop,
                                     coordinates=p.stop-p.start) for p in self.parts])


def project(model, layout):
    """Postprocess model parameters only; no graph, loss, signal or noise access."""
    target=layout.target(model)
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError('original numeric repair must precede projection')
    before=target.detach().clone()
    with torch.no_grad():target.clamp_min_(0.)
    return dict(clamped_coordinates=int((before<0).sum()),zero_coordinates=int((target==0).sum()),
                displacement_l2=float((target.detach().double()-before.double()).norm()),
                nonnegative=bool((target>=0).all()))


def step(model, signal, source, layout):
    layout.target(model)  # Check public layout BEFORE noise/model changes.
    base.update(model.parameters(),signal,source)  # Exactly one original update, including repair.
    return project(model,layout)  # Only these32 processed coordinates change.
