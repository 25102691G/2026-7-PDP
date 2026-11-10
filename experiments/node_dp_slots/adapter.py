"""Slot-local inputs, one-time RWR/SCS/BES caches and separate batch streams."""
from dataclasses import dataclass
import hashlib
import random
import igraph
from experiments.sampling import sample_privim_star
from .config import memberships


@dataclass(frozen=True)
class Graph:
    nodes: tuple
    edges: tuple
    directed: bool

    def __post_init__(self):
        if type(self.nodes) is not tuple or type(self.edges) is not tuple or type(self.directed) is not bool:
            raise ValueError('immutable typed graph required')
        if any(type(v) is not int for v in self.nodes) or any(type(e) is not tuple or len(e)!=2 or any(type(v) is not int for v in e) for e in self.edges):
            raise ValueError('integer stable identities required')

    def delete(self,v):
        return Graph(tuple(x for x in self.nodes if x!=v),tuple(e for e in self.edges if v not in e),self.directed)


@dataclass(frozen=True)
class Item:
    nodes: tuple
    edges: tuple
    directed: bool


def choose_positions(container,rng):
    return rng.sample(container,min(4,len(container)))+[None]*max(0,4-len(container))


def stream(role,j):
    seed=int.from_bytes(hashlib.sha256(f'node-dp-slots-v1|{role}|{j}'.encode()).digest(),'big')
    return random.Random(seed)


class SlotCache:
    def __init__(self,ids,edges,directed,j):
        self.ids=tuple(ids);self.graph=igraph.Graph(n=len(ids),edges=edges,directed=directed)
        self.sample_rng=stream('sample',j);self.batch_rng=stream('batch',j)
        result=sample_privim_star(self.graph,size=40,boundary_divisor=2,
                                 sample_rate=min(1.,64/max(1,len(ids))),restart_probability=.3,
                                 walk_length=200,threshold=4,decay=1.,include_boundary_stage=True,rng=self.sample_rng)
        self.subgraphs=[];self.frequencies=result.frequencies
        for selected in result.node_sets:
            local={old:new for new,old in enumerate(selected)}
            child=tuple(sorted((local[u],local[v]) for u,v in self.graph.get_edgelist() if u in local and v in local))
            self.subgraphs.append(Item(tuple(ids[i] for i in selected),child,directed))

    def fingerprint(self):
        return (self.ids,tuple(self.graph.get_edgelist()),tuple(self.subgraphs),tuple(self.frequencies),
                self.sample_rng.getstate(),self.batch_rng.getstate())


class Adapter:
    def __init__(self,spec,graph):
        ids=tuple(sorted(graph.nodes))
        if len(set(ids))!=len(ids) or not set(ids)<=set(spec.universe) or graph.directed!=spec.directed:
            raise ValueError('graph outside public domain')
        edges=tuple(tuple(e) for e in graph.edges)
        if any(len(e)!=2 or e[0]==e[1] or not set(e)<=set(ids) for e in edges):raise ValueError('invalid simple edges')
        if not graph.directed:edges=tuple(tuple(sorted(e)) for e in edges)
        if len(set(edges))!=len(edges):raise ValueError('duplicate edges outside domain')
        self.caches=[]
        for j in range(4):
            selected=[v for v in ids if j in memberships(v)]
            local={v:i for i,v in enumerate(selected)}
            induced=sorted((local[u],local[v]) for u,v in edges if u in local and v in local)
            self.caches.append(SlotCache(selected,induced,spec.directed,j))

    def batch(self):return [choose_positions(c.subgraphs,c.batch_rng) for c in self.caches]

    def fingerprints(self):return tuple(c.fingerprint() for c in self.caches)
