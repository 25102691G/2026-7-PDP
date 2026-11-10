"""Independent1283 numerical kernel; original slot adapter/Clip3/loss reused unchanged."""
import torch
from experiments.node_dp_slots.aggregation import clip,read_gradient
from experiments.node_dp_slots.adapter import Adapter
from experiments.node_dp_slots.mechanism import runtime
from .model import Layout


def double_clip(positions):
    if tuple(positions.shape)!=(4,4,1283):raise ValueError('fixed4x4x1283 layout required')
    slots=torch.stack([clip(positions[j].double().sum(0)) for j in range(4)])
    return slots.sum(0),slots


def center(model,batch):
    Layout.from_model(model)
    positions=torch.zeros(4,4,1283,dtype=torch.float64)
    _,Loss,get_dgl,get_adj,ig=runtime();loss_function=Loss(.5)
    if len(batch)!=4 or any(len(items)!=4 for items in batch):raise ValueError('four slots with four positions required')
    parameters=list(model.parameters())
    for j in range(4):
        for i,item in enumerate(batch[j]):
            model.zero_grad(set_to_none=True)
            if item is None:continue
            graph=ig.Graph(n=len(item.nodes),edges=item.edges,directed=item.directed)
            message=get_dgl(graph,'ones',32,item.directed,False)
            if not item.directed:graph=ig.Graph(n=len(item.nodes),edges=list(item.edges)+[(v,u) for u,v in item.edges],directed=True)
            adjacency=get_adj(graph)
            out=model(message,message.ndata['feat']).squeeze(1)
            if not bool(torch.isfinite(out).all()):continue
            loss=loss_function(out,adjacency,50)
            if not bool(torch.isfinite(loss)):continue
            loss.backward();positions[j,i]=clip(read_gradient(parameters))
    model.zero_grad(set_to_none=True)
    return double_clip(positions)


def step(model,signal,source,layout):
    if Layout.from_model(model)!=layout or tuple(signal.shape)!=(1283,):raise ValueError('independent phi update layout required')
    before=source.calls;noise=source.draw()
    if source.calls!=before+1 or tuple(noise.shape)!=(1283,):raise RuntimeError('one complete1283 Gaussian per step required')
    parameters=list(model.parameters());old=model.vector().double()
    updated=(old-.005*(signal.double()+noise.double())/16).float()
    updated=torch.where(torch.isfinite(updated),updated,torch.zeros_like(updated))
    with torch.no_grad():
        for p,value in zip(parameters,updated.split([p.numel() for p in parameters])):p.copy_(value.reshape(p.shape))
    return layout.project(model)


def execute(spec,graph,model,noise):
    adapter=Adapter(spec,graph);layout=Layout.from_model(model);nonzero=0
    for _ in range(100):
        signal,slots=center(model,adapter.batch())
        if any(float(s.norm())>3 for s in slots):raise RuntimeError('phi slot bound violated')
        nonzero+=bool(signal.count_nonzero())
        step(model,signal,noise,layout)
    return dict(updates=100,noise_calls=noise.calls,noise_coordinates=1283*noise.calls,projection_calls=100,
                all_projected_weights_nonnegative=bool((model.layer3.fc.weight>=0).all()),nonzero_signal_steps=nonzero)
