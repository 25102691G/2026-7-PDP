"""ones32 function-family parameterization. No prepared loader or private buffers."""
from collections import OrderedDict
import copy
from dataclasses import dataclass
import torch
from torch import nn
from experiments.node_dp_slots.mechanism import new_model


class Model(nn.Module):
    def __init__(self,seed=0):
        super().__init__()
        if type(seed) is not int or seed<0:raise ValueError('public nonnegative model initialization required')
        original=new_model(seed);first=original.grat.grat1
        self.a=nn.Parameter(first.fc.weight.detach().sum(dim=1).clone())
        self.b1=nn.Parameter(first.fc.bias.detach().clone())
        self.layer2=copy.deepcopy(original.grat.grat2);self.layer3=copy.deepcopy(original.grat.grat3)
        self.register_buffer('w10',first.fc.weight.detach().clone())
        self.register_buffer('a0',self.a.detach().clone())
        self.register_buffer('attn_w0',first.attn_fc.weight.detach().clone())
        self.register_buffer('attn_b0',first.attn_fc.bias.detach().clone())
        Layout.from_model(self)

    @staticmethod
    def q1(graph,features):
        if tuple(features.shape)!=(graph.num_nodes(),32) or features.dtype!=torch.float32 or features.device.type!='cpu' or not bool((features==1).all()):
            raise ValueError('frozen CPU FP32 ones32 feature domain required')
        source,target=graph.edges()
        if bool((source==target).any()) or len(set(zip(source.tolist(),target.tolist())))!=graph.num_edges():
            raise ValueError('simple loop-free model input graph required')
        q=features.new_zeros(graph.num_nodes())
        if source.numel():q.index_add_(0,target,1./graph.out_degrees()[source].to(dtype=features.dtype))
        return q

    def first_hidden(self,graph,features):
        q=self.q1(graph,features)
        return torch.relu(q[:,None]*self.a[None,:]+self.b1[None,:])

    def logits(self,graph,features):
        h=self.first_hidden(graph,features)
        if graph.num_nodes()==0:return h.new_empty((0,1))
        h=torch.relu(self.layer2(graph,h))
        return self.layer3(graph,h)

    def forward(self,graph,features):return torch.sigmoid(self.logits(graph,features))

    def vector(self):return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def reconstruct(self):
        # Public deterministic map; retain original state_dict ordering and FP32 initial coordinates.
        result=OrderedDict()
        result['grat.grat1.fc.weight']=self.w10+((self.a-self.a0)/32.)[:,None]
        result['grat.grat1.fc.bias']=self.b1
        result['grat.grat1.attn_fc.weight']=self.attn_w0
        result['grat.grat1.attn_fc.bias']=self.attn_b0
        for name,layer in [('grat.grat2',self.layer2),('grat.grat3',self.layer3)]:
            for key,value in layer.state_dict(keep_vars=True).items():result[name+'.'+key]=value
        return result


@dataclass(frozen=True)
class Layout:
    parts: tuple
    total: int=1283
    start: int=1185
    stop: int=1217

    @classmethod
    def from_model(cls,model):
        expected=[('a',(32,)),('b1',(32,))]
        for layer,out in [('layer2',32),('layer3',1)]:
            expected.extend([(layer+'.fc.weight',(out,32)),(layer+'.fc.bias',(out,)),
                             (layer+'.attn_fc.weight',(1,64)),(layer+'.attn_fc.bias',(1,))])
        actual=[(name,tuple(p.shape)) for name,p in model.named_parameters()]
        if actual!=expected or sum(p.numel() for p in model.parameters())!=1283:
            raise ValueError('independent1283 parameter registration/layout changed')
        if any(p.dtype!=torch.float32 or p.device.type!='cpu' or not p.requires_grad for p in model.parameters()):
            raise ValueError('frozen CPU FP32 trainable phi required')
        expected_buffers=[('w10',(32,32)),('a0',(32,)),('attn_w0',(1,64)),('attn_b0',(1,))]
        if [(n,tuple(p.shape)) for n,p in model.named_buffers()]!=expected_buffers:
            raise ValueError('public initialization buffers changed')
        if any(p.requires_grad or p.dtype!=torch.float32 or p.device.type!='cpu' for p in model.buffers()):
            raise ValueError('frozen public buffer layout changed')
        offset=0;parts=[]
        for name,p in model.named_parameters():parts.append((name,tuple(p.shape),offset,offset+p.numel()));offset+=p.numel()
        target=next(p for p in parts if p[0]=='layer3.fc.weight')
        if target[2:]!=(1185,1217):raise ValueError('head32 offset changed')
        return cls(tuple(parts))

    def project(self,model):
        if Layout.from_model(model)!=self:raise ValueError('phi layout changed')
        target=model.layer3.fc.weight
        if not bool(torch.isfinite(target).all()):raise RuntimeError('finite repair must precede projection')
        with torch.no_grad():target.clamp_min_(0.)
        return dict(nonnegative=bool((target>=0).all()))
