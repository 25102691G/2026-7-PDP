"""Approved Email/H2/seed=0 three-arm exploration; NOT certified DP/PDP.

Only --output is configurable. Native PrivateSGD and all existing research
modules remain unchanged. Synthetic checks must pass before any GNN update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments import node_sampling_clipping as c
from experiments.privacy import PrivateSGD
import numpy as np
import torch

u = c.u
STRUCTURE = ROOT/'outputs/structure_sampling/personalized-node-20261030-v2'
BASELINE = ROOT/'outputs/utility_sampling/email-eps2-clip3-zero-20261030-v1'
PREPARED = ROOT/'dataset/prepared'


def approved_config():
    return dict(c.CONFIG, seeds=[0], noise_level='H2', dp_noise=True,
        gaussian_std_on_sum=103.7109375, gaussian_std_on_mean=6.48193359375,
        optimizer_noise_multiplier=34.5703125, noise_stream='independent',
        privacy_claim=False, accounting_performed=False,
        classification='exploratory_uncertified')


CONFIG = approved_config()


def validate_scope(configuration):
    if configuration != approved_config():
        raise ValueError('only Email H2 seed=0, the fixed three arms and 100 updates are authorized')


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()


class FixedH2SGD(c.ObservedPrivateSGD):
    """Reuse original joint clipping; observe native nonzero-noise SGD step.

    The zero-noise observer's step is intentionally not called. A separate
    generator replays the expected draws without consuming the actual stream.
    """
    def __init__(self, params, *, noise_seed):
        PrivateSGD.__init__(self, params, l2_norm_clip=3., noise_multiplier=34.5703125,
            minibatch_size=16, microbatch_size=1, noise_seed=noise_seed,
            noise_stream='independent', lr=.005, momentum=0.)
        self.current_samples = []
        self.records = []
        self.audit_steps = []

    def step(self, closure=None):
        if len(self.current_samples) != 16:
            raise ValueError('exactly 16 jointly clipped subgraphs required')
        if self.l2_norm_clip != 3 or self.noise_multiplier != 34.5703125 or self.minibatch_size != 16:
            raise ValueError('fixed H2 scale changed')
        params = self.parameters()
        before = [p.detach().clone() for p in params]
        sums = [g.clone() for g in self.accumulated()]
        rng_before = torch.random.get_rng_state().clone()
        generators = {}
        noises = []
        for p in params:
            key = str(p.device)
            if key not in generators:
                gen = torch.Generator(device=p.device)
                actual = self._noise_generators.get(key)
                if actual is None: gen.manual_seed(self.noise_seed)
                else: gen.set_state(actual.get_state())
                generators[key] = gen
            noises.append(torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                      generator=generators[key]).mul_(103.7109375))
        expected_gradients = [g.clone().add_(z).mul_(1/16) for g,z in zip(sums,noises)]
        result = PrivateSGD.step(self, closure)
        after = [p.detach().clone() for p in params]
        used = [p.grad.detach().clone() for p in params]
        noise_ok = all(torch.equal(actual, expected) for actual, expected in zip(used,expected_gradients))
        noise_ok &= all(torch.equal(gen.get_state(),self._noise_generators[key].get_state()) for key,gen in generators.items())
        update_ok = all(torch.equal(new, old.add(g, alpha=-.005)) for new,old,g in zip(after,before,expected_gradients))
        unchanged = torch.equal(rng_before, torch.random.get_rng_state())
        finite = all(bool(torch.isfinite(t).all()) for t in after+used+sums+noises)
        bound_ok = (all(s['bound_ok'] for s in self.current_samples)
                    and c.joint_norm(sums)/16 <= 3+1e-5)
        if not noise_ok or not update_ok or not unchanged or not finite or not bound_ok:
            raise ValueError('H2 coordinate replay, update, RNG, finite value or clipping check failed')
        noise_norm = c.joint_norm(noises)
        if noise_norm <= 0 or not math.isclose(noise_norm,self.last_noise_norm,rel_tol=1e-5):
            raise ValueError('noise failed to enter native optimizer')
        signal_mean = c.joint_norm(sums)/16
        self.records.append(dict(update=len(self.records)+1, samples=list(self.current_samples),
            clip_fraction=self.last_clip_fraction, gradient_sum_norm=c.joint_norm(sums),
            signal_mean_norm=signal_mean, noise_norm_sum=noise_norm, noise_norm_mean=noise_norm/16,
            used_gradient_norm=c.joint_norm(used), parameter_update_norm=c.joint_norm([a-b for a,b in zip(after,before)]),
            noise_to_signal_ratio=(noise_norm/16/signal_mean if signal_mean else None),
            noise_coordinate_count=sum(p.numel() for p in params), noise_sha256=tensor_hash(noises),
            noise_verified=bool(noise_ok), update_verified=bool(update_ok),
            global_rng_unchanged=bool(unchanged), finite=True, joint_bound_ok=bound_ok))
        self.audit_steps.append(dict(before=before, signal_sum=sums, used_gradient=used, after=after))
        return result


def synthetic_parameters():
    row = next(r for r in u.read_csv(BASELINE/'results.csv') if r['arm']=='original' and int(r['seed'])==0)
    shapes = json.loads(row['parameter_shapes'])
    params = [torch.nn.Parameter(torch.zeros(shape)) for shape in shapes.values()]
    if sum(p.numel() for p in params) != 2340: raise ValueError('unexpected GRAT parameter count')
    return params


def synthetic_preflight(draws=512):
    """Many synthetic draws through the actual optimizer; never train a GNN."""
    if draws < 128: raise ValueError('multiple draws, not one noise vector, are required')
    torch_before = torch.random.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    streams, summaries = [], []
    for seed in (17001,17002,17003):
        params = synthetic_parameters()
        optimizer = PrivateSGD(params, l2_norm_clip=3, noise_multiplier=34.5703125,
            minibatch_size=16, microbatch_size=1, noise_seed=seed, noise_stream='independent',lr=.005,momentum=0.)
        gen = torch.Generator().manual_seed(seed)
        samples=[]
        for _ in range(draws):
            optimizer.zero_accum_grad()
            optimizer.step()
            actual = torch.cat([p.grad.flatten() for p in params])*16
            expected = torch.cat([torch.randn(p.shape,generator=gen).mul_(103.7109375).flatten() for p in params])
            if not torch.equal(actual,expected): raise ValueError('synthetic coordinate scale/order mismatch')
            samples.append(actual)
        matrix=torch.stack(samples).double()/103.7109375
        average=float(matrix.mean()); sd=float(matrix.std(unbiased=True))
        coordinate_variance=matrix.var(dim=0,unbiased=True)
        if (abs(average)>.01 or abs(sd-1)>.01 or not bool(((coordinate_variance>.3)&(coordinate_variance<2)).all())):
            raise ValueError('multi-draw Gaussian scale/coordinate variance check failed')
        summaries.append(dict(draws=draws,coordinates=2340,normalized_mean=average,normalized_std=sd,
            min_coordinate_variance=float(coordinate_variance.min()),max_coordinate_variance=float(coordinate_variance.max()),
            every_coordinate_matches_native_draws=True))
        streams.append(matrix.flatten())
    correlations=[]
    for i in range(3):
        for j in range(i):
            correlation=float(torch.corrcoef(torch.stack([streams[i],streams[j]]))[0,1])
            if abs(correlation)>.02 or torch.equal(streams[i],streams[j]):
                raise ValueError('synthetic noise streams are not independent in the implementation check')
            correlations.append(correlation)
    numpy_after=np.random.get_state()
    rng_ok=(torch.equal(torch_before,torch.random.get_rng_state()) and python_before==random.getstate()
        and numpy_before[0]==numpy_after[0] and np.array_equal(numpy_before[1],numpy_after[1])
        and numpy_before[2:]==numpy_after[2:])
    if not rng_ok:raise ValueError('noise modified global initialization/node/batch RNG state')
    # Deterministic joint-gradient test, independent of the statistical checks.
    a=torch.nn.Parameter(torch.tensor([1.]));b=torch.nn.Parameter(torch.tensor([2.]))
    optimizer=FixedH2SGD([{'params':[a]},{'params':[b]}],noise_seed=17004)
    optimizer.zero_accum_grad()
    for _ in range(16):
        optimizer.zero_microbatch_grad();a.grad=torch.tensor([3.]);b.grad=torch.tensor([4.])
        optimizer.microbatch_step()
    if any(not math.isclose(s['pre_norm'],5.) or s['post_norm']>3+1e-5 for s in optimizer.current_samples):
        raise ValueError('joint multi-parameter clipping incorrect')
    optimizer.step()
    return dict(passed=True,draws_per_stream=draws,coordinates=2340,streams=summaries,
        cross_stream_correlations=correlations,global_rng_unchanged=True,all_streams_distinct=True,
        deterministic_joint_clip_and_sgd_verified=True,gaussian_std_sum=103.7109375,
        gaussian_std_mean=6.48193359375,optimizer_noise_multiplier=34.5703125,
        actual_native_step_coordinate_replay=True,gnn_training_performed=False)


def run_approved_cases(context_loader, preflight_sink=None, result_sink=None):
    validate_scope(CONFIG)
    preflight=synthetic_preflight()
    if not preflight.get('passed'):raise ValueError('synthetic preflight did not pass')
    if preflight_sink is not None:preflight_sink(preflight)
    rows=[]
    for arm in u.ARMS:
        rows.append(train_case(context_loader(arm)))
        if result_sink is not None:result_sink(rows)
    return preflight,rows


def selected_nodes(model, graph):
    rt=u.runtime()
    with torch.no_grad():
        logits=rt['evaluate'].ranking_scores(model,graph,graph.ndata['feat']).reshape(-1)
    if not bool(torch.isfinite(logits).all()):raise ValueError('non-finite ranking logits')
    return rt['evaluate'].select_topk_nodes(logits,50).tolist()


def train_case(context):
    rt=u.runtime();arm=context['arm'];case=context['case'];baseline=context['baseline']
    containers=context['containers'];batches=context['batches'];test_graph=context['test_graph'];dg=context['test_dgl']
    model=u.new_model(0)
    initial=c.audit_initialization(model,test_graph,dg,context['denominator'])
    if initial['initial_state_sha256']!=baseline['initial_state_sha256']:raise ValueError('initial weights differ')
    shapes={name:list(t.shape) for name,t in model.state_dict().items()}
    if shapes!=json.loads(baseline['parameter_shapes']) or sum(p.numel() for p in model.parameters())!=2340:
        raise ValueError('GRAT parameter shapes/count changed')
    zero_model=u.new_model(0)
    zero_model.load_state_dict(torch.load(BASELINE/'internal'/f'seed0_{arm}'/'model.pt',map_location='cpu'))
    zero_diagnostics=json.loads((BASELINE/'internal'/f'seed0_{arm}'/'diagnostics.json').read_text())
    if u.state_hash(zero_model)!=zero_diagnostics['final_state_sha256']:
        raise ValueError('saved zero-noise model hash differs from its recorded state')
    zero_selected=selected_nodes(zero_model,dg)
    zero_covered=set(zero_selected)
    for node in zero_selected:zero_covered.update(test_graph.successors(node))
    if len(zero_covered)!=float(baseline['influence_spread']) or not math.isclose(
            100*len(zero_covered)/context['denominator'],float(baseline['coverage_ratio_percent'])):
        raise ValueError('saved zero-noise model does not reproduce the comparison metric')
    optimizer=FixedH2SGD(model.parameters(),noise_seed=context['noise_seed'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=1.)
    loss_fn=rt['Loss'](.5);losses=[]
    case_dir=context['internal']/f'seed0_{arm}'
    case_dir.mkdir(mode=0o700)
    def save_training_state(status, error=None):
        torch.save(model.state_dict(),case_dir/'model.pt')
        torch.save(optimizer.audit_steps,case_dir/'update_audit.pt')
        u.write_json(case_dir/'gradients.json',optimizer.records)
        u.write_csv(case_dir/'losses.csv',losses)
        u.write_json(case_dir/'status.json',dict(status=status,updates=len(optimizer.records),error=error))
        for p in case_dir.iterdir():p.chmod(0o600)
    start=time.perf_counter()
    try:
        for step,indices in enumerate(batches):
            batch=[containers[i] for i in indices]
            values=rt['train'].train_single(model,optimizer,scheduler,loss_fn,
                [b[2] for b in batch],[b[1] for b in batch],[b[0] for b in batch],
                [None]*16,[None]*16,[None]*16,50,step,clip_gradients=True)
            if not all(math.isfinite(v) for v in values) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('non-finite real training; abort without retry or parameter adjustment')
            losses.append(dict(update=step+1,mean_batch_loss=statistics.mean(values)))
    except Exception as error:
        save_training_state('failed',str(error))
        raise
    seconds=time.perf_counter()-start
    if len(optimizer.records)!=100:raise ValueError('wrong update count')
    evaluation=u.evaluate(model,test_graph,dg,context['denominator'],0,arm)
    selected=selected_nodes(model,dg)
    overlap=len(set(selected)&set(zero_selected))
    changed=u.state_hash(model)!=initial['initial_state_sha256']
    if not changed:raise ValueError('parameters did not change')
    rt_diagnostics=rt['evaluate'].ranking_diagnostics(
        rt['evaluate'].ranking_scores(model,dg,dg.ndata['feat']),50)
    save_training_state('completed')
    u.write_json(case_dir/'diagnostics.json',dict(configuration=CONFIG,initialization=initial,
        batch_indices=batches,final_state_sha256=u.state_hash(model),parameter_shapes=shapes,
        final_selected_local_ids=selected,zero_selected_local_ids=zero_selected,ranking=rt_diagnostics))
    for p in case_dir.iterdir():p.chmod(0o600)
    row=dict(seed=0,arm=arm,updates=100,parameter_count=2340,initial_state_sha256=initial['initial_state_sha256'],
        initial_coverage_ratio_percent=initial['coverage_ratio_percent'],**evaluation,
        zero_noise_influence_spread=float(baseline['influence_spread']),
        zero_noise_coverage_ratio_percent=float(baseline['coverage_ratio_percent']),
        delta_vs_zero_noise_pp=evaluation['coverage_ratio_percent']-float(baseline['coverage_ratio_percent']),
        top50_overlap_count=overlap,top50_overlap_fraction=overlap/50,top50_jaccard=overlap/(100-overlap),
        model_parameters_changed=changed,finite_parameters_and_outputs=True,training_seconds=seconds,
        subgraph_container_size=len(containers),gaussian_std_on_sum=103.7109375,
        gaussian_std_on_mean=6.48193359375,noise_multiplier=34.5703125,
        all_noise_coordinates_verified=all(r['noise_verified'] and r['noise_coordinate_count']==2340 for r in optimizer.records),
        all_updates_verified=all(r['update_verified'] for r in optimizer.records),
        noise_stream_global_rng_unchanged=all(r['global_rng_unchanged'] for r in optimizer.records),
        test_graph_sha256=context['test_hash'])
    print(f"Email H2 seed=0 {arm}: updates=100, spread={evaluation['influence_spread']}, coverage={evaluation['coverage_ratio_percent']:.3f}%, delta={row['delta_vs_zero_noise_pp']:+.3f}pp",flush=True)
    return row


def read_manifest(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['records']!=9:raise ValueError('baseline incomplete')
    for key in ('source_sha256','input_sha256'):
        for name,digest in manifest[key].items():
            if u.file_hash(ROOT/name)!=digest:raise ValueError(f'baseline provenance changed: {name}')
    return manifest


def run_experiment(output):
    validate_scope(CONFIG)
    output=Path(output).resolve();internal=u.create_output(output)
    baseline_manifest=read_manifest(BASELINE)
    if baseline_manifest['configuration']!=c.CONFIG:raise ValueError('clipped baseline config mismatch')
    baseline_rows={(int(r['seed']),r['arm']):r for r in u.read_csv(BASELINE/'results.csv')}
    snapshot=u.protected_snapshot(STRUCTURE)
    for directory in (BASELINE,ROOT/'outputs/utility_sampling/email-eps2-nonoise-20261030-v1'):
        snapshot.update({str(p.resolve()):u.file_hash(p) for p in directory.rglob('*') if p.is_file()})
    for p in (ROOT/'experiments/node_sampling_utility.py',ROOT/'tests/test_node_sampling_utility.py',
              ROOT/'experiments/node_sampling_clipping.py',ROOT/'tests/test_node_sampling_clipping.py'):
        snapshot[str(p)]=u.file_hash(p)
    u.write_json(internal/'protected_before.json',snapshot)
    rt=u.runtime()
    import igraph
    env=dict(python=sys.version,torch=torch.__version__,dgl=rt['dgl'].__version__,numpy=np.__version__,
             igraph=igraph.__version__,threads=torch.get_num_threads())
    if env!=baseline_manifest['environment']:raise ValueError('paired runtime differs')
    _,test_graph,directed=rt['loader'].load_prepared_graph('email','test',PREPARED)
    dg=rt['graph_util'].get_rev_dgl(test_graph,'ones',32,directed,False)
    denominator=rt['evaluate'].calculate_celf_coverages(test_graph,1,[50])[50]
    if denominator!=baseline_manifest['celf_spread']:raise ValueError('test denominator differs')
    test_paths=[PREPARED/'email'/name for name in ('email_test_edges.txt','test_nodes.csv','metadata.json')]
    test_hash=hashlib.sha256(''.join(u.file_hash(p) for p in test_paths).encode()).hexdigest()
    test_ids={int(r['original_id']) for r in u.read_csv(PREPARED/'email/test_nodes.csv')}
    contexts={}
    noise_seeds={arm:secrets.randbits(63) for arm in u.ARMS}
    if len(set(noise_seeds.values()))!=3:raise ValueError('noise seed collision; abort before training')
    u.write_json(internal/'noise_seeds.json',noise_seeds)
    inputs={str(p.resolve()):u.file_hash(p) for p in test_paths}
    for arm in u.ARMS:
        case=u.load_saved_case(STRUCTURE,PREPARED,arm,0);baseline=baseline_rows[(0,arm)]
        if test_ids.intersection(case['original_ids']) or test_hash!=baseline['test_graph_sha256']:
            raise ValueError('test isolation or identity failed')
        for path,digest in case['input_sha256'].items():
            if baseline_manifest['input_sha256'].get(path)!=digest:
                raise ValueError('decisions/mappings/containers differ from clipped baseline')
        inputs.update(case['input_sha256'])
        batches=json.loads((BASELINE/'internal'/f'seed0_{arm}'/'diagnostics.json').read_text())['batch_indices']
        unclipped=Path(baseline_manifest['baseline_run'])
        old_batches=json.loads((unclipped/'internal'/f'seed0_{arm}'/'diagnostics.json').read_text())['batch_indices']
        if batches!=old_batches:raise ValueError('clipped baseline batch sequence differs from original paired run')
        containers=u.prepare_container(case)
        if len(batches)!=100 or any(len(b)!=len(set(b)) or len(b)!=16 or any(i<0 or i>=len(containers) for i in b) for b in batches):
            raise ValueError('saved batch sequence invalid')
        contexts[arm]=dict(arm=arm,case=case,baseline=baseline,batches=batches,containers=containers,
            test_graph=test_graph,test_dgl=dg,test_hash=test_hash,denominator=denominator,
            noise_seed=noise_seeds[arm],internal=internal)
    manifest=dict(status='preflight_pending',started_at=datetime.now(timezone.utc).isoformat(),
        configuration=CONFIG,environment=env,classification='exploratory_uncertified',privacy_claim=False,
        allowed_real_training_cases=3,records=0,celf_spread=denominator,input_sha256=inputs,
        baseline_run=str(BASELINE),baseline_manifest_sha256=u.file_hash(BASELINE/'manifest.json'),
        source_sha256={str(p.relative_to(ROOT)):u.file_hash(p) for p in (Path(__file__),ROOT/'tests/test_node_sampling_h2.py')})
    u.write_json(output/'manifest.json',manifest)
    def save_gate(result):
        u.write_json(output/'synthetic_preflight.json',result)
        manifest['status']='preflight_passed';u.write_json(output/'manifest.json',manifest)
        print('SYNTHETIC PREFLIGHT PASSED; permitting exactly Email/H2/seed0 three arms.',flush=True)
    def save_results(rows):
        u.write_csv(output/'results.csv',rows);(output/'results.csv').chmod(0o600)
        manifest['records']=len(rows);u.write_json(output/'manifest.json',manifest)
    try:
        _,rows=run_approved_cases(contexts.__getitem__,save_gate,save_results)
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        u.write_json(output/'manifest.json',manifest)
        raise
    if len(rows)!=3:raise ValueError('unauthorized real case count')
    u.write_csv(output/'results.csv',rows);(output/'results.csv').chmod(0o600)
    u.verify_snapshot(snapshot)
    manifest.update(status='completed',records=3,actual_optimizer_updates=300,
        completed_at=datetime.now(timezone.utc).isoformat(),existing_files_unchanged=True)
    u.write_json(output/'manifest.json',manifest)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run_experiment(parser.parse_args().output)


if __name__=='__main__':main()
