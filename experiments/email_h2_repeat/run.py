"""Final Email H2 replication: exactly six seed=1,2 fits, no seed=0 fit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import secrets
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments import node_sampling_h2 as h
from experiments.noise_only_diagnostic import run as offline
import torch
import numpy as np

u=h.u;c=h.c
H2_ZERO=offline.H2
NO_ZERO=ROOT/'outputs/noise_only/email-h2-seed0-20261030-v1'
CASES=tuple((seed,arm) for seed in (1,2) for arm in u.ARMS)
CONFIG=dict(h.CONFIG,seeds=[1,2])
METRICS=('h2_spread','h2_coverage','noise_only_spread','noise_only_coverage','zero_coverage',
    'h2_minus_noise_only_pp','h2_minus_zero_pp','h2_noise_only_top50_overlap',
    'h2_selected_zero_outdegree','noise_only_selected_zero_outdegree','zero_selected_zero_outdegree',
    'h2_logit_outdegree_spearman','noise_only_logit_outdegree_spearman','zero_logit_outdegree_spearman',
    'initial_coverage','training_seconds')


def validate_scope(config,cases):
    if config!=dict(h.approved_config(),seeds=[1,2]) or tuple(cases)!=CASES:
        raise ValueError('only ordered seeds 1,2 x three arms at fixed H2 are authorized')


def make_noise_seeds(previous):
    values=[secrets.randbits(63) for _ in CASES]
    if len(set(values))!=6 or set(values)&set(previous):
        raise ValueError('noise seed collision: abort without redraw or training')
    return {f'{seed}_{arm}':value for (seed,arm),value in zip(CASES,values)}


def dispatch(loader,sink=None,gate_sink=None):
    validate_scope(CONFIG,CASES)
    gate=h.synthetic_preflight()
    if not gate.get('passed'):raise ValueError('synthetic preflight failed')
    if gate_sink:gate_sink(gate)
    rows=[]
    for case in CASES:
        rows.append(train_case(loader(case)))
        if sink:sink(rows)
    return rows


def rebuild_noise_only(initial,noise_frames):
    # This function has no gradient input and never calls an optimizer.
    return offline.reconstruct_noise_only(initial,noise_frames)


def from_parameters(seed,values):
    model=u.new_model(seed)
    if list(model.buffers()) or len(list(model.parameters()))!=len(values):raise ValueError('unexpected GRAT state')
    with torch.no_grad():
        for p,value in zip(model.parameters(),values):
            if p.shape!=value.shape:raise ValueError('shape mismatch')
            p.copy_(value)
    return model.requires_grad_(False).eval()


def evaluate(model,ctx):
    rt=u.runtime();dg=ctx['test_dgl'];graph=ctx['test_graph']
    model.eval()
    with torch.no_grad():
        scores=rt['evaluate'].ranking_scores(model,dg,dg.ndata['feat']).reshape(-1)
        probs=model(dg,dg.ndata['feat'])
        if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(probs).all()):raise ValueError('non-finite outputs')
        selected=rt['evaluate'].select_topk_nodes(scores,50).tolist()
        result=offline.selection_metrics(graph,selected)
        original=rt['evaluate'].evaluate_model(model,1,'GRAT3',ctx['seed'],'email_test_edges.txt',graph,dg,[50],{50:432},repeat=1)[0]
        if result['influence_spread']!=original['n_covered']:raise ValueError('original evaluator mismatch')
        result.update(offline.logit_diagnostics(scores,graph.outdegree()))
    return result,selected


def offline_case(ctx,model,optimizer):
    seed,arm=ctx['seed'],ctx['arm'];rt=u.runtime()
    with offline.inference_guards(rt):
        model.requires_grad_(False)
        audit=optimizer.audit_steps;records=optimizer.records
        if len(audit)!=len(records) or len(records)!=100:raise ValueError('incomplete real trajectory')
        initial=audit[0]['before'];previous=initial
        generator=torch.Generator().manual_seed(ctx['noise_seed'])
        frames=[];signal_sum=[torch.zeros_like(p,dtype=torch.float64) for p in initial]
        for i,(packet,record) in enumerate(zip(audit,records)):
            if any(not torch.equal(a,b) for a,b in zip(previous,packet['before'])):raise ValueError('parameter discontinuity')
            noise=[torch.randn(p.shape,dtype=p.dtype,generator=generator).mul_(103.7109375) for p in packet['before']]
            if h.tensor_hash(noise)!=record['noise_sha256'] or record['update']!=i+1:raise ValueError('noise replay mismatch')
            offline.verify_saved_step(packet,noise)
            for j,g in enumerate(packet['signal_sum']):signal_sum[j].add_(g.double())
            frames.append(noise);previous=packet['after']
        if any(not torch.equal(p.detach(),a) for p,a in zip(model.parameters(),previous)):
            raise ValueError('hidden final state update')
        reconstruction=rebuild_noise_only(initial,frames)
        noise_model=from_parameters(seed,reconstruction['closed32'])
        fp32_model=from_parameters(seed,reconstruction['sequential32'])
        noise_metric,noise_selected=evaluate(noise_model,ctx)
        fp32_metric,fp32_selected=evaluate(fp32_model,ctx)
        h2_metric,h2_selected=evaluate(model,ctx)
        zero=u.new_model(seed)
        zero.load_state_dict(torch.load(h.BASELINE/'internal'/f'seed{seed}_{arm}'/'model.pt',map_location='cpu'))
        zero.requires_grad_(False)
        old_diag=json.loads((h.BASELINE/'internal'/f'seed{seed}_{arm}'/'diagnostics.json').read_text())
        if u.state_hash(zero)!=old_diag['final_state_sha256']:raise ValueError('zero checkpoint mismatch')
        zero_metric,zero_selected=evaluate(zero,ctx)
        if zero_metric['influence_spread']!=float(ctx['baseline']['influence_spread']):raise ValueError('historical spread mismatch')
        residual=offline.distance([p.detach() for p in model.parameters()],
            [p.double()-(.005/16)*(z+s) for p,z,s in zip(initial,reconstruction['noise_sum64'],signal_sum)])
        if residual>1e-4:raise ValueError('SGD decomposition mismatch')
        cell=ctx['cell']
        torch.save(noise_model.state_dict(),cell/'noise_only.pt')
        torch.save(fp32_model.state_dict(),cell/'noise_only_fp32_check.pt')
        mapping=ctx['test_id_map']
        u.write_json(cell/'top50_ids.json',{name:dict(local_ids=ids,original_ids=[mapping[i] for i in ids])
            for name,ids in (('h2',h2_selected),('noise_only',noise_selected),('zero',zero_selected))})
        checks=dict(actual_updates_replayed=100,noise_coordinates_replayed=234000,
            all_coordinate_and_parameter_updates_bitwise_match=True,
            closed_vs_fp32_top50_identical=noise_selected==fp32_selected,
            closed_vs_fp32_top50_overlap=len(set(noise_selected)&set(fp32_selected))/50,
            fp32_check_spread=fp32_metric['influence_spread'],closed_spread=noise_metric['influence_spread'],
            closed_vs_fp32_max_abs=max(float((a-b).abs().max()) for a,b in zip(reconstruction['closed32'],reconstruction['sequential32'])),
            full_sgd_decomposition_residual=residual,
            h2_noise_only_parameter_distance=offline.distance(list(model.parameters()),list(noise_model.parameters())),
            gradient_displacement_norm=.005/16*c.joint_norm(signal_sum))
        u.write_json(cell/'reconstruction_checks.json',checks)
        nonzero=[r['noise_to_signal_ratio'] for r in records if r['noise_to_signal_ratio'] is not None]
        dominance=dict(arm=arm,seed=seed,updates=100,
            noise_dominates_steps=sum(r['noise_norm_mean']>r['signal_mean_norm'] for r in records),
            exactly_zero_signal_steps=sum(r['signal_mean_norm']==0 for r in records),
            mean_noise_norm_sum=statistics.mean(r['noise_norm_sum'] for r in records),
            mean_parameter_update_norm=statistics.mean(r['parameter_update_norm'] for r in records),
            median_noise_signal_ratio_nonzero=statistics.median(nonzero) if nonzero else None,
            cumulative_signal_parameter_norm=.005/16*c.joint_norm(signal_sum),
            cumulative_noise_parameter_norm=.005/16*c.joint_norm(reconstruction['noise_sum64']),
            trajectory_decomposition_residual=residual,all_actual_updates_replayed=True)
        u.write_json(cell/'dominance.json',dominance)
        u.write_json(cell/'inference_metrics.json',dict(h2=h2_metric,noise_only=noise_metric,zero=zero_metric))
        return dict(h2=h2_metric,noise=noise_metric,zero=zero_metric,
                    overlap=len(set(noise_selected)&set(h2_selected))/50,dominance=dominance,checks=checks)


def train_case(ctx):
    seed,arm=ctx['seed'],ctx['arm']
    if (seed,arm) not in CASES:raise ValueError('seed0/other training forbidden')
    model=ctx['model'];rt=u.runtime()
    if u.state_hash(model)!=ctx['baseline']['initial_state_sha256']:raise ValueError('W0 changed before training')
    initial=c.audit_initialization(model,ctx['test_graph'],ctx['test_dgl'],432)
    optimizer=h.FixedH2SGD(model.parameters(),noise_seed=ctx['noise_seed'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=1.)
    loss_fn=rt['Loss'](.5);losses=[];cell=ctx['cell'];cell.mkdir(mode=0o700)
    def save(status,error=None):
        torch.save(model.state_dict(),cell/'model.pt');torch.save(optimizer.audit_steps,cell/'update_audit.pt')
        u.write_json(cell/'gradients.json',optimizer.records);u.write_csv(cell/'losses.csv',losses)
        u.write_json(cell/'status.json',dict(status=status,updates=len(optimizer.records),error=error))
        for p in cell.iterdir():p.chmod(0o600)
    start=time.perf_counter()
    try:
        for step,indices in enumerate(ctx['batches']):
            batch=[ctx['containers'][i] for i in indices]
            values=rt['train'].train_single(model,optimizer,scheduler,loss_fn,
                [b[2] for b in batch],[b[1] for b in batch],[b[0] for b in batch],
                [None]*16,[None]*16,[None]*16,50,step,clip_gradients=True)
            if not all(math.isfinite(v) for v in values) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('nonfinite training: abort, do not retry/tune')
            losses.append(dict(update=step+1,mean_batch_loss=statistics.mean(values)))
    except Exception as error:
        save('failed',str(error));raise
    elapsed=time.perf_counter()-start
    if len(optimizer.records)!=100:raise ValueError('wrong update count')
    save('trained')
    u.write_json(cell/'diagnostics.json',dict(seed=seed,arm=arm,configuration=CONFIG,
        initial_state_sha256=initial['initial_state_sha256'],initialization=initial,
        batch_indices=ctx['batches'],final_state_sha256=u.state_hash(model)))
    result=offline_case(ctx,model,optimizer)
    hm,nm,zm=result['h2'],result['noise'],result['zero']
    row=dict(seed=seed,arm=arm,h2_spread=hm['influence_spread'],h2_coverage=hm['coverage_ratio_percent'],
        noise_only_spread=nm['influence_spread'],noise_only_coverage=nm['coverage_ratio_percent'],
        zero_coverage=zm['coverage_ratio_percent'],h2_minus_noise_only_pp=hm['coverage_ratio_percent']-nm['coverage_ratio_percent'],
        h2_minus_zero_pp=hm['coverage_ratio_percent']-zm['coverage_ratio_percent'],
        h2_noise_only_top50_overlap=result['overlap'],h2_selected_zero_outdegree=hm['selected_zero_outdegree_count'],
        noise_only_selected_zero_outdegree=nm['selected_zero_outdegree_count'],zero_selected_zero_outdegree=zm['selected_zero_outdegree_count'],
        h2_logit_outdegree_spearman=hm['logit_outdegree_spearman'],noise_only_logit_outdegree_spearman=nm['logit_outdegree_spearman'],
        zero_logit_outdegree_spearman=zm['logit_outdegree_spearman'],initial_coverage=initial['coverage_ratio_percent'],
        training_seconds=elapsed,updates=100,all_noise_updates_verified=True,all_joint_clipping_verified=True,
        finite_parameters_outputs_losses=True,parameters_changed=u.state_hash(model)!=initial['initial_state_sha256'],
        subgraph_container_size=len(ctx['containers']),source_run=str(ctx['output']))
    u.write_json(cell/'result.json',row)
    u.write_json(cell/'status.json',dict(status='completed',updates=100))
    for p in cell.iterdir():p.chmod(0o600)
    print(f"Email H2 seed={seed} {arm}: 100 updates; H2={row['h2_coverage']:.6f}%, Noise-Only={row['noise_only_coverage']:.6f}%, paired delta={row['h2_minus_noise_only_pp']:+.6f}pp",flush=True)
    return row


def read_seed0():
    old={(int(r['seed']),r['arm']):r for r in u.read_csv(H2_ZERO/'results.csv')}
    zero={(int(r['seed']),r['arm']):r for r in u.read_csv(h.BASELINE/'results.csv')}
    no={(r['method'],r['arm']):r for r in u.read_csv(NO_ZERO/'results.csv') if r['method'] in ('noise_only','h2','clipped_zero_noise')}
    if not json.loads((H2_ZERO/'verification.json').read_text())['passed'] or not json.loads((NO_ZERO/'independent_verification.json').read_text())['passed']:
        raise ValueError('seed0 historical verification not passed')
    rows=[]
    for arm in u.ARMS:
        h2,nm,zm=no[('h2',arm)],no[('noise_only',arm)],no[('clipped_zero_noise',arm)];original=old[(0,arm)]
        if (float(h2['influence_spread'])!=float(original['influence_spread'])
            or not math.isclose(float(h2['coverage_ratio_percent']),float(original['coverage_ratio_percent']))
            or float(zm['influence_spread'])!=float(zero[(0,arm)]['influence_spread'])
            or not math.isclose(float(zm['coverage_ratio_percent']),float(zero[(0,arm)]['coverage_ratio_percent']))):
            raise ValueError('seed0 historical spread/coverage mismatch; stop without recomputing')
        rows.append(dict(seed=0,arm=arm,h2_spread=int(h2['influence_spread']),h2_coverage=float(h2['coverage_ratio_percent']),
            noise_only_spread=int(nm['influence_spread']),noise_only_coverage=float(nm['coverage_ratio_percent']),
            zero_coverage=float(zm['coverage_ratio_percent']),
            h2_minus_noise_only_pp=float(h2['coverage_ratio_percent'])-float(nm['coverage_ratio_percent']),
            h2_minus_zero_pp=float(h2['coverage_ratio_percent'])-float(zm['coverage_ratio_percent']),
            h2_noise_only_top50_overlap=float(nm[f'overlap_h2_{arm}']),
            h2_selected_zero_outdegree=int(h2['selected_zero_outdegree_count']),noise_only_selected_zero_outdegree=int(nm['selected_zero_outdegree_count']),
            zero_selected_zero_outdegree=int(zm['selected_zero_outdegree_count']),
            h2_logit_outdegree_spearman=float(h2['logit_outdegree_spearman']) if h2['logit_outdegree_spearman'] else None,
            noise_only_logit_outdegree_spearman=float(nm['logit_outdegree_spearman']) if nm['logit_outdegree_spearman'] else None,
            zero_logit_outdegree_spearman=float(zm['logit_outdegree_spearman']) if zm['logit_outdegree_spearman'] else None,
            initial_coverage=float(original['initial_coverage_ratio_percent']),training_seconds=float(original['training_seconds']),
            updates=100,all_noise_updates_verified=True,all_joint_clipping_verified=True,finite_parameters_outputs_losses=True,
            parameters_changed=True,subgraph_container_size=int(original['subgraph_container_size']),source_run=str(H2_ZERO)))
    return rows,json.loads((H2_ZERO/'internal/dominance_summary.json').read_text())


def summarize(rows,metrics=METRICS):
    result=[]
    for arm in u.ARMS:
        group=[row for row in rows if row['arm']==arm]
        if not group:continue
        record=dict(arm=arm,runs=len(group))
        for metric in metrics:
            values=[row[metric] for row in group if row.get(metric) is not None and math.isfinite(row[metric])]
            record[metric+'_mean']=statistics.mean(values) if values else None
            record[metric+'_std']=statistics.stdev(values) if len(values)>1 else None
            record[metric+'_n']=len(values)
        result.append(record)
    return result


def run(output):
    validate_scope(CONFIG,CASES)
    output=Path(output).resolve();internal=u.create_output(output)
    baseline=h.read_manifest(h.BASELINE)
    if baseline['configuration']!=c.CONFIG:raise ValueError('zero baseline protocol mismatch')
    snapshot=u.protected_snapshot(h.STRUCTURE)
    for directory in (h.BASELINE,H2_ZERO,NO_ZERO,ROOT/'outputs/utility_sampling/email-eps2-nonoise-20261030-v1'):
        snapshot.update({str(p.resolve()):u.file_hash(p) for p in directory.rglob('*') if p.is_file()})
    for name in ('node_sampling_structure','node_sampling_utility','node_sampling_clipping','node_sampling_h2'):
        for p in (ROOT/'experiments'/f'{name}.py',ROOT/'tests'/f'test_{name}.py'):snapshot[str(p)]=u.file_hash(p)
    for p in (ROOT/'experiments/noise_only_diagnostic').rglob('*.py'):snapshot[str(p)]=u.file_hash(p)
    u.write_json(internal/'protected_before.json',snapshot)
    u.verify_snapshot(json.loads((H2_ZERO/'internal/protected_before.json').read_text()))
    u.verify_snapshot(json.loads((NO_ZERO/'internal/protected_before.json').read_text()))
    for history in (H2_ZERO,NO_ZERO):
        m=json.loads((history/'manifest.json').read_text())
        if m['status']!='completed':raise ValueError('historical run incomplete')
        for path,digest in m['source_sha256'].items():
            if u.file_hash(ROOT/path)!=digest:raise ValueError('historical source changed')
    offline.assert_plain_sgd(CONFIG,offline.sgd_defaults())
    rows0,dom0=read_seed0()
    previous_seeds=set(json.loads((H2_ZERO/'internal/noise_seeds.json').read_text()).values())
    noise_seeds=make_noise_seeds(previous_seeds);u.write_json(internal/'noise_seeds.json',noise_seeds)
    rt=u.runtime()
    import igraph
    environment=dict(python=sys.version,torch=torch.__version__,dgl=rt['dgl'].__version__,numpy=np.__version__,
        igraph=igraph.__version__,threads=torch.get_num_threads())
    if environment!=baseline['environment']:raise ValueError('paired runtime mismatch')
    _,test_graph,directed=rt['loader'].load_prepared_graph('email','test',h.PREPARED)
    if not directed or test_graph.vcount()!=458:raise ValueError('test identity mismatch')
    test_dgl=rt['graph_util'].get_rev_dgl(test_graph,'ones',32,directed,False)
    src,dst=test_dgl.edges()
    if sorted(zip(src.tolist(),dst.tolist()))!=sorted((v,u) for u,v in test_graph.get_edgelist()):raise ValueError('direction mismatch')
    if tuple(test_dgl.ndata['feat'].shape)!=(458,32) or not bool((test_dgl.ndata['feat']==1).all()):raise ValueError('features mismatch')
    test_paths=[h.PREPARED/'email'/name for name in ('email_test_edges.txt','test_nodes.csv','metadata.json')]
    test_hash=hashlib.sha256(''.join(u.file_hash(p) for p in test_paths).encode()).hexdigest()
    id_rows=u.read_csv(h.PREPARED/'email/test_nodes.csv')
    id_map={int(r['local_id']):int(r['original_id']) for r in id_rows};test_ids=set(id_map.values())
    zero_rows={(int(r['seed']),r['arm']):r for r in u.read_csv(h.BASELINE/'results.csv')}
    contexts={};inputs={str(p.resolve()):u.file_hash(p) for p in test_paths}
    unclipped=Path(baseline['baseline_run'])
    old_inputs=json.loads((unclipped/'manifest.json').read_text())['input_sha256']
    for path,digest in inputs.items():
        if old_inputs.get(path)!=digest:raise ValueError('test file changed')
    for seed,arm in CASES:
        case=u.load_saved_case(h.STRUCTURE,h.PREPARED,arm,seed);zero=zero_rows[(seed,arm)]
        if test_hash!=zero['test_graph_sha256'] or test_ids.intersection(case['original_ids']):raise ValueError('test/train isolation mismatch')
        for path,digest in case['input_sha256'].items():
            if baseline['input_sha256'].get(path)!=digest:raise ValueError('saved decisions/mappings/containers changed')
        inputs.update(case['input_sha256'])
        diag=json.loads((h.BASELINE/'internal'/f'seed{seed}_{arm}'/'diagnostics.json').read_text())
        batches=diag['batch_indices']
        if batches!=json.loads((unclipped/'internal'/f'seed{seed}_{arm}'/'diagnostics.json').read_text())['batch_indices']:
            raise ValueError('paired batch sequence mismatch')
        containers=u.prepare_container(case)
        if len(batches)!=100 or any(len(b)!=len(set(b)) or len(b)!=16 or any(i<0 or i>=len(containers) for i in b) for b in batches):
            raise ValueError('invalid batches')
        model=u.new_model(seed)
        if u.state_hash(model)!=zero['initial_state_sha256'] or {n:list(v.shape) for n,v in model.state_dict().items()}!=json.loads(zero['parameter_shapes']):
            raise ValueError('initialization/shape mismatch before real training')
        contexts[(seed,arm)]=dict(seed=seed,arm=arm,model=model,baseline=zero,containers=containers,batches=batches,
            test_graph=test_graph,test_dgl=test_dgl,test_id_map=id_map,noise_seed=noise_seeds[f'{seed}_{arm}'],
            cell=internal/f'seed{seed}_{arm}',output=output)
    manifest=dict(status='preflight_pending',started_at=datetime.now(timezone.utc).isoformat(),configuration=CONFIG,
        classification='exploratory_uncertified',privacy_claim=False,allowed_real_training_cases=6,records=0,
        environment=environment,input_sha256=inputs,seed0_retrained=False,seed0_reinferred=False,
        source_sha256={str(p.relative_to(ROOT)):u.file_hash(p) for p in (Path(__file__),Path(__file__).with_name('test_repeat.py'))})
    u.write_json(output/'manifest.json',manifest)
    def save(rows):
        u.write_csv(output/'new_results.csv',rows);(output/'new_results.csv').chmod(0o600)
        manifest['records']=len(rows);u.write_json(output/'manifest.json',manifest)
    def gate_sink(gate):
        u.write_json(output/'synthetic_preflight.json',gate);manifest['status']='preflight_passed'
        u.write_json(output/'manifest.json',manifest)
        print('PREFLIGHT PASSED: exactly six seed1/2 cases; no seed0 fit/inference.',flush=True)
    try:new=dispatch(contexts.__getitem__,save,gate_sink)
    except Exception as error:
        manifest.update(status='failed',error=str(error));u.write_json(output/'manifest.json',manifest);raise
    merged=sorted(rows0+new,key=lambda r:(r['seed'],u.ARMS.index(r['arm'])))
    if len(merged)!=9 or {(r['seed'],r['arm']) for r in merged}!={(s,a) for s in (0,1,2) for a in u.ARMS}:
        raise ValueError('merged identities mismatch')
    u.write_csv(output/'merged_results.csv',merged);u.write_csv(output/'summary.csv',summarize(merged))
    dominance=dom0+[json.loads((internal/f'seed{s}_{a}'/'dominance.json').read_text()) for s,a in CASES]
    u.write_csv(internal/'merged_dominance.csv',dominance)
    dm=('noise_dominates_steps','exactly_zero_signal_steps','mean_noise_norm_sum','mean_parameter_update_norm',
        'median_noise_signal_ratio_nonzero','cumulative_signal_parameter_norm','cumulative_noise_parameter_norm')
    u.write_csv(internal/'dominance_summary.csv',summarize(dominance,dm))
    comparisons=[]
    for seed in (0,1,2):
        by={r['arm']:r for r in merged if r['seed']==seed}
        comparisons.append(dict(seed=seed,personalized_minus_original_h2_pp=by['personalized']['h2_coverage']-by['original']['h2_coverage'],
            personalized_minus_uniform_h2_pp=by['personalized']['h2_coverage']-by['uniform_matched']['h2_coverage']))
    u.write_csv(output/'paired_group_differences.csv',comparisons)
    for p in output.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    u.verify_snapshot(snapshot)
    manifest.update(status='completed',records=6,actual_new_training_updates=600,merged_records=9,
        completed_at=datetime.now(timezone.utc).isoformat(),historical_files_unchanged=True,
        email_empirical_expansion_finished=True)
    u.write_json(output/'manifest.json',manifest)
    return merged


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
