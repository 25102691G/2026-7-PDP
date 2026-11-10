"""Offline Noise-Only counterfactual and structural references; ZERO training."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import random
import statistics
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments import node_sampling_h2 as h
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

u=h.u;c=h.c
H2=ROOT/'outputs/utility_sampling/email-eps2-clip3-h2-seed0-20261030-v1'
RANDOM_SEEDS=tuple(range(20))
LR=.005;B=16;K=50;DENOMINATOR=432


@contextmanager
def inference_guards(rt=None):
    def forbidden(*args,**kwargs):raise RuntimeError('training/gradient/subgraph operation forbidden in this diagnostic')
    with ExitStack() as stack:
        for owner,name in ((torch.Tensor,'backward'),(torch.autograd,'backward'),
                           (torch.autograd,'grad'),(torch.optim.SGD,'step'),
                           (u,'prepare_container'),(u,'load_saved_case'),(h,'train_case')):
            stack.enter_context(patch.object(owner,name,side_effect=forbidden))
        if rt is not None:stack.enter_context(patch.object(rt['train'],'train_single',side_effect=forbidden))
        with torch.inference_mode():yield


def sgd_defaults():
    signature=inspect.signature(torch.optim.SGD.__init__)
    return {key:value.default for key,value in signature.parameters.items() if value.default is not inspect.Parameter.empty}


def assert_plain_sgd(configuration,defaults):
    if (configuration.get('learning_rate')!=LR or configuration.get('batch_size')!=B
        or configuration.get('momentum')!=0 or configuration.get('lr_decay')!=1
        or configuration.get('weight_decay',0)!=0 or configuration.get('ema',False)):
        raise ValueError('closed Noise-Only formula not equivalent to configured update')
    for key,expected in dict(weight_decay=0,dampening=0,nesterov=False,maximize=False,differentiable=False).items():
        if defaults.get(key)!=expected:raise ValueError(f'nonstandard native SGD default: {key}')


def verify_saved_step(packet,noises):
    arrays=[packet[key] for key in ('before','signal_sum','used_gradient','after')]
    if len({len(part) for part in arrays+[noises]})!=1:raise ValueError('parameter list mismatch')
    for before,signal,used,after,noise in zip(*arrays,noises):
        if not all(t.shape==before.shape and bool(torch.isfinite(t).all()) for t in (signal,used,after,noise)):
            raise ValueError('invalid saved tensors')
        expected=signal.clone().add_(noise).mul_(1/B)
        if not torch.equal(used,expected) or not torch.equal(after,before.add(expected,alpha=-LR)):
            raise ValueError('actual optimizer not equal to plain SGD; no approximation permitted')


def reconstruct_noise_only(initial,frames):
    """Closed real-arithmetic formula, plus original FP32 addition-order check."""
    with torch.no_grad():
        summed=[torch.zeros_like(p,dtype=torch.float64) for p in initial]
        sequential=[p.detach().clone() for p in initial]
        for frame in frames:
            if len(frame)!=len(initial):raise ValueError('noise parameter count mismatch')
            for i,(p,z) in enumerate(zip(initial,frame)):
                if p.shape!=z.shape or not bool(torch.isfinite(z).all()):raise ValueError('invalid actual noise')
                summed[i].add_(z.detach().double())
                sequential[i].add_(z.detach().clone().mul_(1/B),alpha=-LR)
        closed=[p.detach().double()-(LR/B)*z for p,z in zip(initial,summed)]
        return dict(closed64=closed,closed32=[p.float() for p in closed],
                    sequential32=sequential,noise_sum64=summed)


def degree_topk(graph,k=K):
    degrees=graph.outdegree()
    return sorted(range(graph.vcount()),key=lambda node:(-degrees[node],node))[:k]


def random_sets(n,k=K):
    return {seed:random.Random(seed).sample(range(n),k) for seed in RANDOM_SEEDS}


def selection_metrics(graph,selected):
    covered=set(selected)
    for node in selected:covered.update(graph.successors(node))
    degrees=graph.outdegree(selected)
    return dict(influence_spread=len(covered),coverage_ratio_percent=100*len(covered)/DENOMINATOR,
        mean_selected_outdegree=statistics.mean(degrees),selected_zero_outdegree_count=degrees.count(0))


def logit_diagnostics(scores,degrees,k=K):
    values=scores.detach().cpu().double().reshape(-1).numpy()
    if not np.isfinite(values).all():raise ValueError('non-finite inference')
    unique,counts=np.unique(values,return_counts=True)
    boundary=np.sort(values)[-k]
    varying=len(unique)>1 and len(set(degrees))>1
    return dict(logit_outdegree_spearman=(float(spearmanr(values,degrees).statistic) if varying else None),
        logit_outdegree_pearson=(float(pearsonr(values,degrees).statistic) if varying else None),
        unique_logits=len(unique),max_logit_tie_count=int(counts.max()),
        boundary_tie_count=int((values==boundary).sum()),
        boundary_tie_crosses_k=bool((values>boundary).sum()+(values==boundary).sum()>k))


def model_with_parameters(parameters):
    model=u.new_model(0)
    if list(model.buffers()):raise ValueError('unexpected model buffers; formula covers parameters only')
    if len(list(model.parameters()))!=len(parameters):raise ValueError('unexpected GRAT parameter count')
    with torch.no_grad():
        for p,value in zip(model.parameters(),parameters):
            if p.shape!=value.shape:raise ValueError('parameter shape mismatch')
            p.copy_(value);p.requires_grad_(False)
    model.eval()
    return model


def loaded_model(path):
    model=u.new_model(0);model.load_state_dict(torch.load(path,map_location='cpu'))
    model.requires_grad_(False);model.eval()
    return model


def params(model):return [p.detach() for p in model.parameters()]
def distance(a,b):return c.joint_norm([x.double()-y.double() for x,y in zip(a,b)])


def audit_history(manifest,internal):
    assert_plain_sgd(manifest['configuration'],sgd_defaults())
    if manifest['configuration']!=h.CONFIG or manifest['celf_spread']!=DENOMINATOR:
        raise ValueError('historical H2 protocol differs')
    seeds=json.loads((H2/'internal/noise_seeds.json').read_text())
    sources={};checks=[];shared_initial=None
    for arm in u.ARMS:
        cell=H2/'internal'/f'seed0_{arm}'
        packets=torch.load(cell/'update_audit.pt',map_location='cpu')
        records=json.loads((cell/'gradients.json').read_text())
        diag=json.loads((cell/'diagnostics.json').read_text())
        if len(packets)!=len(records) or len(records)!=100:raise ValueError('incomplete saved trajectory')
        initial=packets[0]['before']
        if shared_initial is None:shared_initial=initial
        elif any(not torch.equal(a,b) for a,b in zip(initial,shared_initial)):raise ValueError('initial weights differ across arms')
        generator=torch.Generator().manual_seed(seeds[arm])
        previous=initial;frames=[];signals=[torch.zeros_like(p,dtype=torch.float64) for p in initial]
        for index,(packet,record) in enumerate(zip(packets,records)):
            if any(not torch.equal(a,b) for a,b in zip(previous,packet['before'])):raise ValueError('hidden between-step update')
            noise=[torch.randn(p.shape,dtype=p.dtype,generator=generator).mul_(103.7109375) for p in packet['before']]
            if h.tensor_hash(noise)!=record['noise_sha256'] or record['update']!=index+1:
                raise ValueError('noise replay differs from actual saved noise')
            verify_saved_step(packet,noise)
            for i,g in enumerate(packet['signal_sum']):signals[i].add_(g.double())
            frames.append(noise);previous=packet['after']
        actual=loaded_model(cell/'model.pt')
        if any(not torch.equal(p,a) for p,a in zip(params(actual),previous)):
            raise ValueError('checkpoint contains a hidden post-training update/EMA')
        if u.state_hash(actual)!=diag['final_state_sha256']:raise ValueError('H2 checkpoint hash mismatch')
        initial_model=model_with_parameters(initial)
        if u.state_hash(initial_model)!=diag['initialization']['initial_state_sha256']:
            raise ValueError('saved initial model hash mismatch')
        reconstructed=reconstruct_noise_only(initial,frames)
        residual=distance(params(actual),[p.double()-(LR/B)*(z+g) for p,z,g in zip(initial,reconstructed['noise_sum64'],signals)])
        if residual>1e-4:raise ValueError('closed full-trajectory decomposition is inconsistent')
        noise_model=model_with_parameters(reconstructed['closed32'])
        sequential_model=model_with_parameters(reconstructed['sequential32'])
        torch.save(noise_model.state_dict(),internal/f'noise_only_{arm}.pt')
        torch.save(sequential_model.state_dict(),internal/f'noise_only_fp32_check_{arm}.pt')
        discrepancy=distance(reconstructed['closed32'],reconstructed['sequential32'])
        max_abs=max(float((a-b).abs().max()) for a,b in zip(reconstructed['closed32'],reconstructed['sequential32']))
        checks.append(dict(arm=arm,actual_steps_verified=100,actual_noise_coordinates_verified=234000,
            actual_noise_hashes_match=True,actual_plain_sgd_bitwise_match=True,no_hidden_checkpoint_update=True,
            full_closed_form_roundoff_residual=residual,closed_vs_fp32_norm=discrepancy,
            closed_vs_fp32_max_abs=max_abs,h2_noise_only_parameter_distance=distance(params(actual),params(noise_model)),
            gradient_displacement_norm=(LR/B)*c.joint_norm(signals),
            gradient_decomposition_error=distance([a.double()-n.double() for a,n in zip(params(actual),params(noise_model))],
                [-(LR/B)*g for g in signals])))
        sources[arm]=dict(h2=actual,noise_only=noise_model,noise_only_fp32=sequential_model,
            zero=loaded_model(h.BASELINE/'internal'/f'seed0_{arm}'/'model.pt'),initial=initial_model)
    return sources,checks,shared_initial


def run(output):
    output=Path(output).resolve();internal=u.create_output(output)
    snapshot=u.protected_snapshot(h.STRUCTURE)
    for directory in (H2,h.BASELINE,ROOT/'outputs/utility_sampling/email-eps2-nonoise-20261030-v1'):
        snapshot.update({str(p.resolve()):u.file_hash(p) for p in directory.rglob('*') if p.is_file()})
    for name in ('node_sampling_utility','node_sampling_clipping','node_sampling_h2'):
        for p in (ROOT/'experiments'/f'{name}.py',ROOT/'tests'/f'test_{name}.py'):
            snapshot[str(p)]=u.file_hash(p)
    u.write_json(internal/'protected_before.json',snapshot)
    history=json.loads((H2/'manifest.json').read_text())
    if history['status']!='completed' or history['records']!=3:raise ValueError('H2 history incomplete')
    u.verify_snapshot(json.loads((H2/'internal/protected_before.json').read_text()))
    for path,digest in {**history['input_sha256'],**history['source_sha256']}.items():
        if u.file_hash(ROOT/path)!=digest:raise ValueError('historical input/source changed')
    rt=u.runtime()
    import igraph
    environment=dict(python=sys.version,torch=torch.__version__,dgl=rt['dgl'].__version__,
        numpy=np.__version__,igraph=igraph.__version__,threads=torch.get_num_threads())
    if environment!=history['environment']:raise ValueError('inference runtime differs from saved H2 runtime')
    config=dict(dataset='email',source_model_seed=0,random_top50_seeds=list(RANDOM_SEEDS),
        diffusion_steps=1,ic_probability=1,k=50,celf_spread=432,
        noise_only_primary='float64 closed formula cast to float32 for original GRAT',
        real_training_performed=False,gradient_computation_performed=False,privacy_claim=False)
    manifest=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),configuration=config,
        environment=environment,source_h2=str(H2),source_h2_manifest_sha256=u.file_hash(H2/'manifest.json'),
        source_sha256={str(p.relative_to(ROOT)):u.file_hash(p) for p in (Path(__file__),Path(__file__).with_name('test_diagnostic.py'))})
    u.write_json(output/'manifest.json',manifest)
    with inference_guards(rt):
        sources,checks,initial=audit_history(history,internal)
        _,graph,directed=rt['loader'].load_prepared_graph('email','test',h.PREPARED)
        if not directed or graph.vcount()!=458:raise ValueError('unexpected test graph direction/size')
        dg=rt['graph_util'].get_rev_dgl(graph,'ones',32,directed,False)
        if not bool((dg.ndata['feat']==1).all()) or tuple(dg.ndata['feat'].shape)!=(458,32):raise ValueError('feature mismatch')
        a,b=dg.edges()
        if sorted(zip(a.tolist(),b.tolist()))!=sorted((v,u) for u,v in graph.get_edgelist()):raise ValueError('message edge direction mismatch')
        mapping={int(r['local_id']):int(r['original_id']) for r in u.read_csv(h.PREPARED/'email/test_nodes.csv')}
        if sorted(mapping)!=list(range(458)) or len(set(mapping.values()))!=458:raise ValueError('test ID mapping invalid')
        degrees=graph.outdegree();rows=[];all_ids={};parameter_metrics=[]
        def logits(model):
            if any(p.requires_grad for p in model.parameters()):raise ValueError('inference model not frozen')
            values=rt['evaluate'].ranking_scores(model,dg,dg.ndata['feat']).reshape(-1)
            if not bool(torch.isfinite(values).all()):raise ValueError('non-finite logits')
            return values
        def top(values):return rt['evaluate'].select_topk_nodes(values,50).tolist()
        references={arm:top(logits(sources[arm]['h2'])) for arm in u.ARMS}
        old_h2={(int(r['seed']),r['arm']):r for r in u.read_csv(H2/'results.csv')}
        old_zero={(int(r['seed']),r['arm']):r for r in u.read_csv(h.BASELINE/'results.csv')}
        def add_result(method,arm,selected,model=None,values=None,random_seed=None):
            metrics=selection_metrics(graph,selected)
            native=rt['evaluate'].get_influence_d(graph,selected,1)
            if native!=metrics['influence_spread']:raise ValueError('outgoing coverage disagrees with original evaluator')
            row=dict(method=method,arm=arm,random_seed=random_seed,**metrics)
            row.update({f'overlap_h2_{key}':len(set(selected)&set(ref))/50 for key,ref in references.items()})
            row.update({key:None for key in ('logit_outdegree_spearman','logit_outdegree_pearson','unique_logits',
                'max_logit_tie_count','boundary_tie_count','boundary_tie_crosses_k')})
            if model is not None:
                row.update(logit_diagnostics(values,degrees))
                parameter_metrics.append(dict(method=method,arm=arm,parameter_norm=c.joint_norm(params(model)),
                    displacement_from_initial=distance(params(model),initial)))
                original=rt['evaluate'].evaluate_model(model,1,'GRAT3',0,'email_test_edges.txt',graph,dg,[50],{50:432},repeat=1)[0]
                if original['n_covered']!=native:raise ValueError('original neural evaluation mismatch')
            identifier=f'{method}:{arm}:{random_seed}'
            all_ids[identifier]=dict(local_ids=selected,original_ids=[mapping[i] for i in selected])
            rows.append(row);return row
        initial_model=sources['original']['initial'];initial_values=logits(initial_model)
        initial_row=add_result('initial','shared',top(initial_values),initial_model,initial_values)
        for arm in u.ARMS:
            for method,key in (('noise_only','noise_only'),('clipped_zero_noise','zero'),('h2','h2')):
                model=sources[arm][key];values=logits(model)
                row=add_result(method,arm,top(values),model,values)
                expected=(old_h2[(0,arm)] if method=='h2' else old_zero[(0,arm)] if method=='clipped_zero_noise' else None)
                if expected is not None and (row['influence_spread']!=float(expected['influence_spread'])
                        or not math.isclose(row['coverage_ratio_percent'],float(expected['coverage_ratio_percent']))):
                    raise ValueError('historical spread mismatch; stop before conclusions')
            primary=next(r for r in rows if r['method']=='noise_only' and r['arm']==arm)
            seq=top(logits(sources[arm]['noise_only_fp32']))
            primary_ids=all_ids[f'noise_only:{arm}:None']['local_ids']
            check=next(check for check in checks if check['arm']==arm)
            check.update(closed_vs_fp32_top50_identical=seq==primary_ids,
                closed_vs_fp32_top50_overlap=len(set(seq)&set(primary_ids))/50,
                fp32_check_spread=selection_metrics(graph,seq)['influence_spread'],
                closed_spread=primary['influence_spread'],h2_noise_only_top50_overlap=primary[f'overlap_h2_{arm}'])
        add_result('out_degree','shared',degree_topk(graph))
        for seed,selected in random_sets(graph.vcount()).items():add_result('random','shared',selected,random_seed=seed)
        if len(rows)!=31:raise ValueError('unexpected evaluation scope')
        if initial_row['influence_spread']!=323:raise ValueError('initial historical spread mismatch')
        u.write_csv(output/'results.csv',rows)
        u.write_json(internal/'top50_ids.json',all_ids)
        u.write_csv(internal/'parameter_metrics.csv',parameter_metrics)
        u.write_json(internal/'reconstruction_checks.json',checks)
        torch.save({str(i):p for i,p in enumerate(initial)},internal/'initial_parameters.pt')
    random_rows=[r for r in rows if r['method']=='random']
    summary=dict(seeds=list(RANDOM_SEEDS),runs=20,
        influence_spread_mean=statistics.mean(r['influence_spread'] for r in random_rows),
        influence_spread_sample_std=statistics.stdev(r['influence_spread'] for r in random_rows),
        coverage_percent_mean=statistics.mean(r['coverage_ratio_percent'] for r in random_rows),
        coverage_percent_sample_std=statistics.stdev(r['coverage_ratio_percent'] for r in random_rows),
        min_spread=min(r['influence_spread'] for r in random_rows),max_spread=max(r['influence_spread'] for r in random_rows))
    u.write_json(output/'random_summary.json',summary)
    for p in output.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    u.verify_snapshot(snapshot)
    u.write_json(output/'verification.json',dict(passed=True,historical_models_matched=True,
        actual_sgd_steps_replayed=300,actual_noise_coordinates_replayed=702000,
        no_new_training=True,no_backward=True,no_subgraph_generation=True,
        original_sources_and_outputs_unchanged=True,closed_vs_fp32_top50_all_equal=all(r['closed_vs_fp32_top50_identical'] for r in checks),
        native_evaluation_and_direction_verified=True,private_artifacts_restricted=True))
    manifest.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),evaluation_rows=len(rows))
    u.write_json(output/'manifest.json',manifest)
    for row in rows:
        if row['method']!='random':print(f"{row['method']} {row['arm']}: spread={row['influence_spread']}, coverage={row['coverage_ratio_percent']:.6f}%",flush=True)
    print(f"Random(20): {summary['coverage_percent_mean']:.6f}% +/- {summary['coverage_percent_sample_std']:.6f}% sample SD",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
