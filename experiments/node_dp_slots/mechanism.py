"""Fixed100-step full path; idealized claim only, resource failures explicit."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import sys
import torch
from .config import Spec
from .adapter import Adapter
from .aggregation import clip,read_gradient,double_clip
from .noise import SecretGaussian,TestGaussian
from .accounting import Ledger

_PROCESS_LEDGER=Ledger()


class ExecutionGap(RuntimeError):pass


@lru_cache(maxsize=1)
def runtime():
    root=Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from model.GRAT import GRAT3
    from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
    from experiments.graph_util import get_rev_dgl,get_adj_mat
    import igraph
    return GRAT3,KSetMaxCoverAdjLoss,get_rev_dgl,get_adj_mat,igraph


def new_model(seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=runtime()[0](32,32,32,variant='legacy_released',initialization='released_default')
    if sum(p.numel() for p in model.parameters())!=2340:raise RuntimeError('public model layout changed')
    return model


def center(model,batch):
    parameters=[p for p in model.parameters() if p.requires_grad]
    a=torch.zeros(4,4,2340,dtype=torch.float64)
    _,Loss,get_dgl,get_adj,ig=runtime();loss_function=Loss(.5)
    for j in range(4):
        for i,item in enumerate(batch[j]):
            model.zero_grad(set_to_none=True)
            if item is None:continue
            graph=ig.Graph(n=len(item.nodes),edges=item.edges,directed=item.directed)
            message=get_dgl(graph,'ones',32,item.directed,False)
            if not item.directed:
                graph=ig.Graph(n=len(item.nodes),edges=list(item.edges)+[(v,u) for u,v in item.edges],directed=True)
            adjacency=get_adj(graph)
            out=model(message,message.ndata['feat']).squeeze(1)
            if not bool(torch.isfinite(out).all()):continue
            loss=loss_function(out,adjacency,50)
            if not bool(torch.isfinite(loss)):continue
            loss.backward()
            a[j,i]=clip(read_gradient(parameters))
    model.zero_grad(set_to_none=True)
    return double_clip(a)


def update(parameters,signal,source):
    parameters=list(parameters)
    if signal.shape!=(2340,):raise RuntimeError('public signal layout changed')
    z=source.draw()  # one vector AFTER bounded slot sum, never one draw per slot
    if z.shape!=(2340,):raise RuntimeError('public noise layout changed')
    old=torch.cat([p.detach().reshape(-1).double() for p in parameters])
    next_state=(old-.005*(signal.double()+z.double())/16).float()
    next_state=torch.where(torch.isfinite(next_state),next_state,torch.zeros_like(next_state))
    with torch.no_grad():
        for p,b in zip(parameters,next_state.split([p.numel() for p in parameters])):p.copy_(b.reshape(p.shape))


@dataclass(frozen=True)
class Run:
    _parameters: tuple
    _spec: Spec
    _mode: str
    _completed: bool
    testing_diagnostics: object=None


class Engine:
    def __init__(self,*args,**kwargs):raise TypeError('use privacy(spec) or testing(spec,noise_seed=...)')

    @classmethod
    def privacy(cls,spec,*,ledger=None):
        if type(spec) is not Spec:raise TypeError('frozen Spec required before graph')
        obj=object.__new__(cls);obj.spec=spec;obj._mode='privacy';obj._source=SecretGaussian()
        obj._ledger=ledger if ledger is not None else _PROCESS_LEDGER
        return obj

    @classmethod
    def testing(cls,spec,*,noise_seed):
        if type(spec) is not Spec:raise TypeError('frozen Spec required')
        obj=object.__new__(cls);obj.spec=spec;obj._mode='testing';obj._source=TestGaussian(noise_seed);obj._ledger=None
        return obj

    def run(self,graph):
        if self._mode=='privacy':
            if type(self._source) is not SecretGaussian:raise TypeError('test/injected noise forbidden')
            self._ledger.charge()  # before graph access; failures do not refund
        elif self._source.calls:raise ValueError('testing engine single use')
        try:
            from .training import execute
            values,_,_=execute(self.spec,graph,self._source)
            updates=100
        except (MemoryError,RuntimeError,OSError) as error:
            raise ExecutionGap('backend/resource failure; no certified completion or safe failure guarantee') from error
        diagnostics=None
        if self._mode=='testing':
            diagnostics=dict(updates=updates,noise_calls=self._source.calls,coordinates=2340,
                             finite_parameters=all(math.isfinite(x) for x in values))
        return Run(values,self.spec,self._mode,True,diagnostics)
