"""New research orchestration; existing losses, clipping/update and policy unchanged.

Only public slot-local sampler parameters and loop length vary. No production
Session, release API, ledger, mutable global patch or accountant replacement.
"""
import json
from pathlib import Path
import torch
import igraph
from experiments.node_dp_slots.adapter import Adapter, SlotCache, Item, stream
from experiments.node_dp_slots.config import memberships
from experiments.node_dp_slots import mechanism as base
from experiments.node_pdp import sampling
from experiments.node_pdp_v2_projected import policy
from experiments.sampling import sample_privim_star
from .plan import noise_plan, validate_task


class CandidateCache(SlotCache):
    def __init__(self, ids, edges, directed, j, *, n, M):
        self.ids=tuple(ids)
        self.graph=igraph.Graph(n=len(ids),edges=edges,directed=directed)
        self.sample_rng=stream('sample',j);self.batch_rng=stream('batch',j)
        result=sample_privim_star(self.graph,size=n,boundary_divisor=2,
            sample_rate=min(1.,64/max(1,len(ids))),restart_probability=.3,
            walk_length=200,threshold=M,decay=1.,include_boundary_stage=True,rng=self.sample_rng)
        self.subgraphs=[];self.frequencies=result.frequencies
        for selected in result.node_sets:
            local={old:new for new,old in enumerate(selected)}
            child=tuple(sorted((local[u],local[v]) for u,v in self.graph.get_edgelist() if u in local and v in local))
            self.subgraphs.append(Item(tuple(ids[i] for i in selected),child,directed))


class CandidateAdapter(Adapter):
    def __init__(self,spec,graph,*,n,M):
        if n not in (40,50,60,70,80) or M not in (2,3,4,5,6,7): raise ValueError('unsupported public sampler parameters')
        sampling.validate_graph(graph,spec)
        ids=tuple(sorted(graph.nodes))
        edges=graph.edges if graph.directed else tuple(tuple(sorted(e)) for e in graph.edges)
        self.caches=[]
        for j in range(4):
            selected=[v for v in ids if j in memberships(v)]
            local={v:i for i,v in enumerate(selected)}
            induced=sorted((local[u],local[v]) for u,v in edges if u in local and v in local)
            self.caches.append(CandidateCache(selected,induced,spec.directed,j,n=n,M=M))


class _ScaledSource:
    def __init__(self,source,factor):self.source=source;self.factor=factor
    @property
    def calls(self):return self.source.calls
    def draw(self):return self.source.draw()*self.factor


def _source(method,seed=None):
    if method=='pdp_v2':from experiments.node_dp_slots import noise
    elif method=='uniform1':from experiments.uniform_node_dp import noise
    elif method=='uniform4':from experiments.uniform_node_dp_eps4 import noise
    else:raise ValueError('private method required')
    return noise.SecretGaussian() if seed is None else noise.TestGaussian(seed)


def _loop(model,spec,graph,*,n,M,steps,source,factor,progress=None):
    adapter=CandidateAdapter(spec,graph,n=n,M=M)
    scaled=_ScaledSource(source,factor);layout=policy.Layout.from_model(model)
    counts=dict(updates=0,gaussian_vectors=0,gaussian_coordinates=0,projections=0,
                slot_caches=4,slot_container_sizes=[len(c.subgraphs) for c in adapter.caches])
    model.train()
    for i in range(steps):
        signal,slots=base.center(model,adapter.batch())
        if any(float(s.norm())>3. for s in slots):raise RuntimeError('slot clipping bound violated')
        previous=scaled.calls
        stats=policy.step(model,signal,scaled,layout)
        if scaled.calls!=previous+1 or not stats['nonnegative']:raise RuntimeError('Gaussian/projection contract violated')
        counts.update(updates=i+1,gaussian_vectors=scaled.calls,gaussian_coordinates=scaled.calls*2340,projections=i+1)
        if progress is not None and ((i+1)%10==0 or i+1==steps):progress(dict(counts))
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise RuntimeError('nonfinite completed parameters')
    return counts


def run_private_synthetic(model,spec,graph,*,n,M,steps,method,noise_seed,reference_steps=None):
    """Seeded artificial-fixture entry only; no dataset loading or secret source."""
    if type(noise_seed) is not int or noise_seed<0:raise ValueError('explicit nonnegative synthetic seed required')
    reference_steps=steps if reference_steps is None else reference_steps
    plan=noise_plan(method,reference_steps)
    if not 0<steps<=reference_steps:raise ValueError('synthetic update range')
    return _loop(model,spec,graph,n=n,M=M,steps=steps,source=_source(method,noise_seed),factor=plan['temporal_factor'])


def fit_private(task,registry,spec,graph,directory,progress):
    """A once-only claimed manual-screen task; never accepts an injected source."""
    validate_task(task);directory=Path(directory)
    claim=json.loads((directory/'claim.json').read_text())
    if claim['task']!=task:raise PermissionError('task/claim mismatch')
    if registry.universe!=spec.universe:raise ValueError('frozen registry/domain mismatch')
    with (directory/'fit_started.json').open('x') as f:json.dump(dict(task_id=task['id'],attempts=1),f)
    model=base.new_model(task['seed'])
    if task['method']=='pdp_v2':
        # One COMPLETE-training amplification. No retained IDs/mask are persisted.
        graph=sampling.sample_once(graph,registry,sampling.SecretBernoulli()).graph
    plan=noise_plan(task['method'],task['T'])
    counts=_loop(model,spec,graph,n=task['n'],M=task['M'],steps=task['T'],
        source=_source(task['method']),factor=plan['temporal_factor'],progress=progress)
    counts['outer_sampling_calls']=int(task['method']=='pdp_v2')
    return model,counts
