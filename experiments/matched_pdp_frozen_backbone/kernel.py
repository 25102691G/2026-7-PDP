"""Slot-local33D queries and exact100-step schedule, unchanged GRAT3 forward/loss."""
import torch
from experiments.node_dp_slots import mechanism as base
from experiments.node_dp_slots.adapter import Adapter
from experiments.node_dp_slots.aggregation import clip, read_gradient
from .config import validate
from .layout import Layout, head_vector


def position_gradient(model, item):
    parameters = Layout.from_model(model).parameters(model)
    model.zero_grad(set_to_none=True)
    if item is None:
        return torch.zeros(33, dtype=torch.float64)
    _, Loss, get_dgl, get_adj, ig = base.runtime()
    graph = ig.Graph(n=len(item.nodes), edges=item.edges, directed=item.directed)
    # Keep the original DGL empty-edge semantics; do not invent self loops.
    message = get_dgl(graph, 'ones', 32, item.directed, False)
    if not item.directed:
        graph = ig.Graph(n=len(item.nodes), edges=list(item.edges)+[(v,u) for u,v in item.edges], directed=True)
    out = model(message, message.ndata['feat']).squeeze(1)
    if not bool(torch.isfinite(out).all()):
        return torch.zeros(33, dtype=torch.float64)
    loss = Loss(.5)(out, get_adj(graph), 50)
    if not bool(torch.isfinite(loss)):
        return torch.zeros(33, dtype=torch.float64)
    loss.backward()
    result = read_gradient(parameters)
    model.zero_grad(set_to_none=True)
    return result


def double_clip(clipped_positions):
    if tuple(clipped_positions.shape) != (4,4,33):
        raise ValueError('fixed4x4x33 query required')
    slots = torch.stack([clip(clipped_positions[j].double().sum(0)) for j in range(4)])
    return slots.sum(0), slots


def center(model, batch):
    if len(batch) != 4 or any(len(row) != 4 for row in batch):
        raise ValueError('four slots of four positions required')
    positions = torch.zeros(4,4,33,dtype=torch.float64)
    for j in range(4):
        for i,item in enumerate(batch[j]):
            positions[j,i] = clip(position_gradient(model,item))
    return double_clip(positions)


def step(model, signal, source, layout):
    parameters = layout.parameters(model)
    if tuple(signal.shape) != (33,):
        raise ValueError('33D signal required')
    noise = source.draw()  # Always all33 coordinates, even if signal is zero.
    if tuple(noise.shape) != (33,):
        raise ValueError('33D noise required')
    next_state = (head_vector(model)-.005*(signal.double()+noise.double())/16).float()
    next_state = torch.where(torch.isfinite(next_state),next_state,torch.zeros_like(next_state))
    with torch.no_grad():
        for parameter, block in zip(parameters,next_state.split((32,1))):
            parameter.copy_(block.reshape(parameter.shape))
        parameters[0].clamp_min_(0.)
    return dict(nonnegative=bool((parameters[0]>=0).all()))


def execute(spec, graph, source, model):
    validate(spec)
    layout = Layout.from_model(model)
    adapter = Adapter(spec,graph)
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    for _ in range(spec.T):
        signal, slots = center(model,adapter.batch())
        if any(float(b.norm()) > 3. for b in slots):
            raise RuntimeError('slot Clip3 bound violated')
        calls = source.calls
        step(model,signal,source,layout)
        if source.calls != calls+1:
            raise RuntimeError('exactly one full33D Gaussian vector per step required')
        if any(not torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen):
            raise RuntimeError('frozen backbone changed')
    return tuple(float(v) for v in torch.cat([p.detach().reshape(-1) for p in model.parameters()]))
