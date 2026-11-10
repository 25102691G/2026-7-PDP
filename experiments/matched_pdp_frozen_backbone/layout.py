"""Public-seeded original model, exact head layout and frozen backbone contract."""
from dataclasses import dataclass
import torch
from experiments.node_dp_slots import mechanism as base

HEAD_NAMES = ('grat.grat3.fc.weight', 'grat.grat3.fc.bias')
EXPECTED = tuple(
    (f'grat.grat{layer}.{part}', shape)
    for layer in (1,2,3)
    for part, shape in (
        ('fc.weight', (32,32) if layer < 3 else (1,32)),
        ('fc.bias', (32,) if layer < 3 else (1,)),
        ('attn_fc.weight', (1,64)), ('attn_fc.bias', (1,)))
)


@dataclass(frozen=True)
class Layout:
    names: tuple = HEAD_NAMES
    shapes: tuple = ((1,32),(1,))
    total: int = 33

    @classmethod
    def from_model(cls, model):
        actual = tuple((n,tuple(p.shape)) for n,p in model.named_parameters())
        if actual != EXPECTED or list(model.buffers()):
            raise ValueError('original GRAT3 parameter/buffer layout changed')
        for n,p in model.named_parameters():
            if p.requires_grad != (n in HEAD_NAMES) or p.dtype != torch.float32 or p.device.type != 'cpu':
                raise ValueError('exact frozen CPU FP32 head layout required')
        return cls()

    def parameters(self, model):
        if Layout.from_model(model) != self:
            raise ValueError('head registration changed')
        named = dict(model.named_parameters())
        return tuple(named[n] for n in self.names)

    def public(self):
        return dict(names=list(self.names), shapes=[list(s) for s in self.shapes],
                    dimension=33, weight_range=[0,32], bias_range=[32,33],
                    frozen_coordinates=2307, full_model_dimension=2340,
                    initial_projection=False, projected_coordinates=32)


def new_model(seed=0):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('public model initialization seed must be a nonnegative int64')
    model = base.new_model(seed)
    for name, p in model.named_parameters():
        p.requires_grad_(name in HEAD_NAMES)
    Layout.from_model(model)
    return model


def head_vector(model):
    return torch.cat([p.detach().reshape(-1).double() for p in Layout.from_model(model).parameters(model)])
