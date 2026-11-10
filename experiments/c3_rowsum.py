"""Bounded C3 research orchestration; frozen math cores and no production export.

The C2/C2R Sessions remain quarantined. This independently validated C3 interface
never changes them. Private H/items/schedule have no serialization API.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import math
import torch
from experiments.node_dp_slots import mechanism as base
from experiments.node_dp_slots.adapter import Adapter
from experiments.node_dp_slots.noise import SecretGaussian as OldSecret,TestGaussian as OldTest
from experiments.node_dp_slots.config import Spec as OldSpec
from experiments.node_pdp.accounting import validate_public
from experiments.node_pdp.sampling import SecretBernoulli,TestBernoulli,sample_once,validate_graph
from experiments.node_pdp_v2_projected import policy
from experiments.rowsum_pdp_c2r.model import Model
from experiments.rowsum_pdp.model import Layout as PhiLayout
from experiments.rowsum_pdp.config import Spec as PhiSpec,validate as validate_phi
from experiments.rowsum_pdp.noise import SecretGaussian as PhiSecret,TestGaussian as PhiTest
from experiments.rowsum_pdp import training as phi_core

ROOT=Path(__file__).resolve().parents[1]
VERSION='C3-rowsum-piecewise-controlled-v1'
ORDER=(('pdp_v2',0),('pdp_v2',1),('rowsum_c2r',0),('rowsum_c2r',1))


def signature():
    """Only public code/config bytes, never H, caches, masks or random states."""
    protected=json.loads((ROOT/'docs/pdp_research/stage_c3_rowsum/preservation_before.json').read_text())
    files={name for name in protected if name.endswith('.py') and name.startswith(('experiments/','model/','pdp/','baselines/'))}
    files.update(('experiments/c3_rowsum.py','tests/test_c3_rowsum.py',
        'docs/pdp_research/stage_c3_rowsum/frozen_plan.json',
        'docs/pdp_research/stage_c3_rowsum/EXECUTION_CONTRACT.md',
        'docs/pdp_research/stage_c3_rowsum/worker.py',
        'docs/pdp_research/stage_c3_rowsum/run.py'))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(files)}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def verify_preflight(seal):
    if seal.get('status')!='C3_PREFLIGHT_PASS' or seal.get('signature')!=signature():
        raise PermissionError('C3 preflight/code/config seal required; no real sampling or training')


class Group:
    """One hidden H, immutable original cached schedule, four independent fits."""
    def __init__(self,registry,spec,graph,seed,*,synthetic,preflight=None,coin_seed=None,noise_seeds=None):
        if type(seed) is not int or seed not in range(5):raise ValueError('public initialization0..4 only')
        if type(synthetic) is not bool:raise TypeError('explicit research/synthetic boundary required')
        if not synthetic:
            verify_preflight(preflight or {})
            if coin_seed is not None or noise_seeds is not None:raise PermissionError('test randomness forbidden in research')
        validate_public(registry,spec);validate_graph(graph,spec)
        self.phi_spec=PhiSpec(spec.universe,spec.directed);validate_phi(registry,self.phi_spec)
        self.seed=seed;self.spec=spec;self.synthetic=synthetic;self.position=0;self.failed=False;self.last_completed_updates=0
        coins=TestBernoulli(coin_seed) if synthetic else SecretBernoulli()
        held=sample_once(graph,registry,coins).graph
        adapter=Adapter(spec,held)
        self._schedule=tuple(tuple(tuple(slot) for slot in adapter.batch()) for _ in range(100))
        if synthetic:
            if type(noise_seeds) is not tuple or len(noise_seeds)!=4:raise ValueError('four explicit artificial streams required')
            self._sources=tuple((OldTest if method=='pdp_v2' else PhiTest)(s) for (method,_),s in zip(ORDER,noise_seeds))
        else:self._sources=(OldSecret(),OldSecret(),PhiSecret(),PhiSecret())
        assert len({id(x) for x in self._sources})==4
        self.witness=dict(same_once_sampled_H=True,same_immutable_cached_items=True,same_100_step_batch_table=True,
                          independent_model_state=True,distinct_gaussian_sources=True,no_saved_H_mask_cache_noise=True)
        # Public parameters only: no model inference on any graph.
        original=base.new_model(seed);candidate=Model(seed)
        assert all(torch.equal(value,candidate.reconstruct()[name]) for name,value in original.state_dict().items())
        self.initial_layouts=dict(pdp_v2=policy.Layout.from_model(original).public(),rowsum_c2r=asdict(PhiLayout.from_model(candidate)))

    def train_next(self,method,repeat):
        if self.failed or self.position>=4 or (method,repeat)!=ORDER[self.position]:
            raise PermissionError('fixed single-attempt group order; no retries')
        source=self._sources[self.position]
        expected=(OldTest if self.synthetic else OldSecret) if method=='pdp_v2' else (PhiTest if self.synthetic else PhiSecret)
        if type(source) is not expected or source.calls!=0:
            raise PermissionError('fresh independent source of correct mode required')
        self.position+=1;self.last_completed_updates=0
        try:
            model=base.new_model(self.seed) if method=='pdp_v2' else Model(self.seed)
            layout=policy.Layout.from_model(model) if method=='pdp_v2' else PhiLayout.from_model(model)
            for step_index in range(100):
                batch=[list(slot) for slot in self._schedule[step_index]]
                signal,slots=(base.center if method=='pdp_v2' else phi_core.center)(model,batch)
                if any(float(slot.norm())>3 for slot in slots):raise RuntimeError('frozen slot bound violation')
                before=source.calls
                (policy.step if method=='pdp_v2' else phi_core.step)(model,signal,source,layout)
                if source.calls!=before+1:raise RuntimeError('exactly one complete Gaussian per update required')
                self.last_completed_updates=step_index+1
            head=layout.target(model) if method=='pdp_v2' else model.layer3.fc.weight
            if source.calls!=100 or not bool((head>=0).all()):raise RuntimeError('100 updates/head projection required')
            dimension=2340 if method=='pdp_v2' else 1283
            return model,dict(updates=100,gaussian_calls=100,gaussian_coordinates=100*dimension,dimension=dimension,
                              sigma_sum=260,projection_calls=100,head_nonnegative=True,publication_allowed=False,algorithm_version=VERSION)
        except BaseException:
            self.failed=True
            raise


def evaluate(model,graph,message,reference,k=50):
    """Original ranking/ties and sparse one-step IC, with no extra diagnostics."""
    from experiments import node_sampling_utility as u
    from baselines.heuristics import get_influence
    if not math.isfinite(reference) or reference<=0:raise ValueError('frozen positive CELF required')
    rt=u.runtime();model.eval()
    with torch.no_grad():scores=rt['evaluate'].ranking_scores(model,message,message.ndata['feat']).reshape(-1)
    if not bool(torch.isfinite(scores).all()):raise RuntimeError('nonfinite logits')
    selected=rt['evaluate'].select_topk_nodes(scores,k).tolist()
    if len(selected)!=k or len(set(selected))!=k:raise RuntimeError('invalid fixed-k selection')
    spread=get_influence(graph,selected)
    return dict(spread=spread,coverage=spread/reference*100,reference_spread=reference,k=k,ic_p=1,steps=1)


def export(*args,**kwargs):
    raise PermissionError('C3 restricted development only; production export not authorized')


def trajectory_record(task,result,mechanism,**metadata):
    if mechanism['updates']!=100 or mechanism['gaussian_calls']!=100:raise RuntimeError('incomplete trajectory cannot be COMPLETED')
    return dict(task,**result,status='COMPLETED',updates=100,mechanism_checks=mechanism,**metadata)
