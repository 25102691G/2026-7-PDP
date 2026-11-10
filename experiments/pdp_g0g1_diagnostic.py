"""Bounded historical v2 diagnosis: no training, sampling or privacy-ledger API.

The parent reserves one new output and enforces a 600-second worker wall limit.
The child charges every model-logits attempt before execution, maximum 30.
Only original test graphs, saved final weights and public initializations are used.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'outputs/research/five-dataset-three-methods-20261102-v1/restricted'
DATASETS=('bitcoin','lastfm','facebook')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    path.chmod(0o600)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closed_neighborhoods(n,edges,directed):
    result=[{i} for i in range(n)]
    for u,v in edges:
        result[u].add(v)
        if not directed:result[v].add(u)
    return result


def coverage_stats(neighborhoods,selected):
    if len(set(selected))!=len(selected):raise ValueError('seed IDs must be distinct')
    covered=set();increments=[];counts=Counter()
    for v in selected:
        area=neighborhoods[v];increments.append(len(area-covered));covered.update(area);counts.update(area)
    A=sum(len(neighborhoods[v]) for v in selected)
    return dict(spread=len(covered),A=A,R=1-len(covered)/A if A else None,
        mean_selected_closed_reach=A/len(selected) if selected else None,
        mean_selected_outdegree=A/len(selected)-1 if selected else None,
        ordered_marginal_gains=increments,
        leave_one_out_unique=[sum(counts[u]==1 for u in neighborhoods[v]) for v in selected])


def decompose(A0,R0,A,R):
    return dict(reach_term=(A-A0)*(1-(R+R0)/2),redundancy_term=-(R-R0)*(A+A0)/2)


def _ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);r=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in order[i:j]:r[k]=(i+j-1)/2
        i=j
    return r


def spearman(x,y):
    if len(x)!=len(y) or len(x)<2:return None
    a=_ranks(x);b=_ranks(y);m=(len(x)-1)/2
    va=sum((v-m)**2 for v in a);vb=sum((v-m)**2 for v in b)
    return sum((v-m)*(w-m) for v,w in zip(a,b))/math.sqrt(va*vb) if va and vb else None


class Budget:
    def __init__(self,max_calls=30,seconds=600,*,clock=time.perf_counter,receipt=None):
        self.max_calls=max_calls;self.seconds=seconds;self.clock=clock;self.start=clock()
        self.attempts=0;self.completed=0;self.receipt=receipt

    def record(self):
        result=dict(attempted_forwards=self.attempts,completed_forwards=self.completed,
                    forward_limit=self.max_calls,elapsed_seconds=self.clock()-self.start,seconds_limit=self.seconds)
        if self.receipt:write(self.receipt,result)
        return result

    def call(self,function):
        if self.clock()-self.start>=self.seconds:raise TimeoutError('diagnostic elapsed-time budget exhausted')
        if self.attempts>=self.max_calls:raise RuntimeError('diagnostic forward budget exhausted')
        self.attempts+=1;self.record()
        result=function();self.completed+=1;self.record();return result


def selected_tasks():
    tasks=read(HISTORY/'all_tasks.json')['tasks']
    selected=[t for t in tasks if t['dataset'] in DATASETS and t['algorithm']=='node_pdp_v2_projected']
    selected.sort(key=lambda t:(DATASETS.index(t['dataset']),t['model_seed']))
    if len(selected)!=15 or any(sorted(t['model_seed'] for t in selected if t['dataset']==d)!=list(range(5)) for d in DATASETS):
        raise ValueError('complete five-seed historical v2 cases required')
    return selected


def protected_files(tasks):
    paths={ROOT/'AGENTS.md',HISTORY/'all_tasks.json',HISTORY/'tasks_frozen.json'}
    for folder in ('model','experiments/node_dp_slots','experiments/node_pdp','experiments/node_pdp_v2_projected'):
        paths.update((ROOT/folder).rglob('*.py'))
    paths.update(ROOT/name for name in ('experiments/run_pdp_research.py','experiments/pdp_research_evaluation.py',
        'experiments/node_pdp_research.py','experiments/graph_util.py','experiments/load_graph.py','experiments/igraph_loader.py'))
    for task in tasks:
        paths.update(Path(task['metrics']['source']).glob('*.json'))
    for dataset in DATASETS:
        directory=ROOT/'dataset/prepared'/dataset
        paths.update(directory/name for name in ('metadata.json','test_nodes.csv','node_mapping.csv',f'{dataset}_test_edges.txt'))
    return sorted(paths)


def _state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _runtime(restricted):
    os.environ.setdefault('DGLBACKEND','pytorch')
    os.environ.setdefault('DGLDEFAULTDIR',str(restricted/'dgl-cache'))
    os.environ.setdefault('MPLCONFIGDIR',str(restricted/'mpl-cache'))
    for path in (ROOT,ROOT/'experiments'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    # Legacy imports initialize log files; contain them in this new restricted run.
    previous=Path.cwd();os.chdir(restricted)
    try:
        import torch
        import dgl
        from model.GRAT import GRAT3
        from experiments.graph_util import get_rev_dgl
        from experiments.load_graph import connect_node_to_graph
        from experiments.igraph_loader import load_prepared_graph
        from experiments.evaluate_models import select_topk_nodes
    finally:os.chdir(previous)
    torch.set_num_threads(1)
    forbidden=('experiments.node_dp_slots.mechanism','experiments.node_pdp.mechanism',
               'experiments.node_dp_slots.accounting','experiments.node_pdp_v2_projected.mechanism')
    if any(name in sys.modules for name in forbidden):raise RuntimeError('privacy engine/ledger module imported')
    return torch,dgl,GRAT3,get_rev_dgl,connect_node_to_graph,load_prepared_graph,select_topk_nodes


def _tensor_summary(x):
    x=x.detach().double()
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std(unbiased=False)),
                zero_fraction=float((x==0).double().mean()),negative_fraction=float((x<0).double().mean()),
                zero_rows=int((x==0).all(1).sum()) if x.ndim==2 else None)


def _parameter_norms(model):
    result={name:float(p.detach().double().norm()) for name,p in model.named_parameters()}
    result['joint']=math.sqrt(sum(v*v for v in result.values()));return result


def _infer(model,message,neighborhoods,budget,selector,torch):
    probe={};hooks=[]
    for j in (1,2,3):
        layer=getattr(model.grat,f'grat{j}')
        hooks.append(layer.fc.register_forward_pre_hook(lambda module,args,j=j:probe.__setitem__(f'm{j}',args[0].detach().clone())))
        hooks.append(layer.register_forward_hook(lambda module,args,out,j=j:probe.__setitem__(f'pre{j}',out.detach().clone())))
    model.eval();before=_state_hash(model)
    try:
        with torch.inference_mode():
            scores=budget.call(lambda:model.logits(message,message.ndata['feat']).reshape(-1)).detach()
    finally:
        for hook in hooks:hook.remove()
    if _state_hash(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('model mutated or gradient produced')
    if not bool(torch.isfinite(scores).all()):raise RuntimeError('nonfinite scores')
    ids=selector(scores,50).tolist();values=scores.tolist();degree=[len(s)-1 for s in neighborhoods]
    c=coverage_stats(neighborhoods,ids);ordered=torch.sort(scores,descending=True).values
    bias=float(model.grat.grat3.fc.bias.detach().reshape(()));margin=scores.double()-bias
    probabilities=torch.sigmoid(scores)
    result=dict(c,topk_local_ids=ids,score_outdegree_spearman=spearman(values,degree),
        selected_mean_score=float(scores[ids].double().mean()),logit_summary=_tensor_summary(scores),
        sigmoid_saturation_fraction=float(((probabilities<=1e-4)|(probabilities>=1-1e-4)).double().mean()),
        positive_margin_nodes=int((margin>0).sum()),negative_margin_nodes=int((margin<0).sum()),
        bias_tie_nodes=int((margin==0).sum()),topk_boundary_gap=float(ordered[49].double()-ordered[50].double()),
        boundary_tie_count=int((scores==ordered[49]).sum()),selected_zero_outdegree=sum(degree[i]==0 for i in ids),
        parameter_norms=_parameter_norms(model),activations={key:_tensor_summary(value) for key,value in probe.items()})
    for j in (1,2):result['activations'][f'relu{j}']=_tensor_summary(probe[f'pre{j}'].clamp_min(0))
    return result,values


def _trajectory(path):
    frames=read(path/'training_diagnostics.json')['steps']
    observation=path/'sampling_diagnostics.json'
    observed=read(observation) if observation.exists() else None
    result=dict(zero_gradient_slots=sum(f['zero_gradient_slots'] for f in frames),
        padded_positions=sum(16-f['effective_positions'] for f in frames),
        projection_trigger_steps=sum(f['clamped_coordinates']>0 for f in frames),
        cumulative_projection_displacement=sum(f['displacement_l2'] for f in frames),
        sampling_candidates=observed['sampling_slots'] if observed else None,
        unrecorded=['actual Gaussian vectors','outer retained identities','per-step checkpoints','unclipped gradient directions'])
    if observed:
        phases=[]
        for name,lo,hi in [('early',0,20),('middle',40,60),('late',80,100)]:
            slots=[s for step in observed['steps'][lo:hi] for s in step['slots']]
            values=[s['slot_preclip_norm'] for s in slots if s['slot_preclip_norm'] is not None]
            phases.append(dict(phase=name,mean_preclip_norm=sum(values)/len(values) if values else None,
                slot_clip_triggers=sum(bool(s['slot_norm_above_C']) for s in slots),slot_positions=len(slots)))
        result['phases']=phases
    return result


def worker(output):
    restricted=output/'restricted';budget=Budget(receipt=restricted/'forward_budget.json')
    tasks=selected_tasks();manifest=read(restricted/'manifest.json');expected=manifest['protected_sha256']
    records=[]
    try:
        torch,dgl,GRAT3,get_rev,connect,load_test,selector=_runtime(restricted)
        initial_hashes=read(HISTORY/'tasks_frozen.json')['initial_model_hashes']
        for dataset in DATASETS:
            _,graph,directed=load_test(dataset,'test',ROOT/'dataset/prepared')
            neighborhoods=closed_neighborhoods(graph.vcount(),graph.get_edgelist(),directed)
            message=get_rev(graph,'ones',32,directed,False)
            coverage,message=connect(list(range(graph.vcount())),message)
            if message.ndata[dgl.NID].tolist()!=list(range(graph.vcount())):raise RuntimeError('message node mapping changed')
            if any(set(coverage.neighbors(v,mode='out'))|{v}!=neighborhoods[v] for v in range(graph.vcount())):
                raise RuntimeError('original coverage direction mismatch')
            with (ROOT/'dataset/prepared'/dataset/'test_nodes.csv').open() as f:mapping={int(r['local_id']):int(r['original_id']) for r in csv.DictReader(f)}
            for task in (t for t in tasks if t['dataset']==dataset):
                seed=task['model_seed'];path=Path(task['metrics']['source']);complete=read(path/'completion.json')
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed);model=GRAT3(32,32,32,variant='legacy_released',initialization='released_default')
                if _state_hash(model)!=initial_hashes[str(seed)] or _state_hash(model)!=complete['initial_model_sha256']:
                    raise RuntimeError('public initial-state reconstruction hash mismatch')
                initial_parameters=[p.detach().clone() for p in model.parameters()]
                initial,initial_scores=_infer(model,message,neighborhoods,budget,selector,torch)
                old_initial=read(path/'initial_evaluation.json')
                if initial['topk_local_ids']!=old_initial['topk_local_ids'] or initial['spread']!=old_initial['spread']:
                    raise RuntimeError('historical initial ranking/spread mismatch')
                values=read(path/'model.json')['parameters'];flat=torch.tensor(values,dtype=torch.float32)
                if flat.numel()!=2340:raise RuntimeError('historical model dimensions mismatch')
                with torch.no_grad():
                    offset=0
                    for p in model.parameters():p.copy_(flat[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
                if _state_hash(model)!=complete['final_model_sha256']:raise RuntimeError('saved final parameter hash mismatch')
                final,final_scores=_infer(model,message,neighborhoods,budget,selector,torch)
                old_final=read(path/'final_evaluation.json')
                if final['topk_local_ids']!=old_final['topk_local_ids'] or final['spread']!=old_final['spread']:
                    raise RuntimeError('historical final ranking/spread mismatch')
                displacements={name:float((p.detach().double()-q.double()).norm()) for (name,p),q in zip(model.named_parameters(),initial_parameters)}
                displacements['joint']=math.sqrt(sum(v*v for v in displacements.values()))
                result=dict(dataset=dataset,model_seed=seed,source=str(path),reference_spread=old_final['reference_spread'],
                    initial=initial,final=final,parameter_displacement=displacements,
                    initial_state_source='public seeded reconstruction; bitwise state hash matches historical receipt, not a saved initial checkpoint',
                    initial_final_top50_overlap=len(set(initial['topk_local_ids'])&set(final['topk_local_ids']))/50,
                    trajectory=_trajectory(path),initial_scores=initial_scores,final_scores=final_scores,
                    test_original_ids=[mapping[i] for i in range(graph.vcount())])
                records.append(result);write(restricted/f'{dataset}_seed{seed}.json',result)
                print(dataset,seed,'reproduced',final['spread'],'A',final['A'],'R',round(final['R'],4),'rho',final['score_outdegree_spearman'],flush=True)
        changed=[p for p,h in expected.items() if digest(p)!=h]
        if changed:raise RuntimeError('protected evidence changed')
        with (restricted/'per_seed.csv').open('x',newline='') as stream:
            fields=['dataset','model_seed','initial_coverage','final_coverage','spread','A','R','mean_selected_outdegree',
                    'score_outdegree_spearman','sigmoid_saturation_fraction','topk_boundary_gap','bias_tie_nodes','zero_gradient_slots',
                    'initial_final_top50_overlap','parameter_displacement','reach_delta_vs_seed0','redundancy_delta_vs_seed0']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in records:
                base=next(b for b in records if b['dataset']==r['dataset'] and b['model_seed']==0)['final'];f=r['final']
                terms=decompose(base['A'],base['R'],f['A'],f['R'])
                if abs(terms['reach_term']+terms['redundancy_term']-(f['spread']-base['spread']))>1e-8:raise RuntimeError('decomposition mismatch')
                writer.writerow(dict(dataset=r['dataset'],model_seed=r['model_seed'],initial_coverage=100*r['initial']['spread']/r['reference_spread'],
                    final_coverage=100*f['spread']/r['reference_spread'],**{k:f[k] for k in ['spread','A','R','mean_selected_outdegree','score_outdegree_spearman','sigmoid_saturation_fraction','topk_boundary_gap','bias_tie_nodes']},
                    zero_gradient_slots=r['trajectory']['zero_gradient_slots'],initial_final_top50_overlap=r['initial_final_top50_overlap'],parameter_displacement=r['parameter_displacement']['joint'],
                    reach_delta_vs_seed0=terms['reach_term'],redundancy_delta_vs_seed0=terms['redundancy_term']))
        (restricted/'per_seed.csv').chmod(0o600)
        write(restricted/'completion.json',dict(status='completed',**budget.record(),cases=len(records),
            historical_rankings_and_spreads_match=True,all_model_parameters_unchanged=True,protected_changes=changed,
            training_calls=0,backward_calls=0,private_sampling_calls=0,privacy_ledger_modules_imported=False,
            scope='internal diagnostic; not DP output; G2-G5 not authorized'))
    except BaseException as error:
        write(restricted/'failure.json',dict(status='stopped-no-retry',exception_type=type(error).__name__,message=str(error),**budget.record()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args();os.umask(0o077)
    output=args.output.resolve()
    if args.worker:
        worker(output);return
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('new output must be under outputs')
    output.mkdir(parents=True,exist_ok=False,mode=0o700);restricted=output/'restricted';restricted.mkdir(mode=0o700)
    tasks=selected_tasks();protected={str(p):digest(p) for p in protected_files(tasks)}
    write(restricted/'manifest.json',dict(scope='G0/G1 only',datasets=list(DATASETS),model_seeds=list(range(5)),
        max_forwards=30,worker_wall_limit_seconds=600,no_training=True,no_new_private_randomness=True,
        protected_sha256=protected,script_sha256=digest(__file__),no_automatic_retry=True))
    command=[sys.executable,'-m','experiments.pdp_g0g1_diagnostic','--worker','--output',str(output)]
    start=time.perf_counter()
    with (restricted/'worker.log').open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();code=-9
            write(restricted/'resource_stop.json',dict(reason='600-second cumulative worker wall limit; no retry',returncode=code))
    (restricted/'worker.log').chmod(0o600)
    write(restricted/'supervisor.json',dict(returncode=code,worker_wall_seconds=time.perf_counter()-start,
        protected_changes=[p for p,h in protected.items() if digest(p)!=h]))
    raise SystemExit(0 if code==0 else 1)


if __name__=='__main__':main()
