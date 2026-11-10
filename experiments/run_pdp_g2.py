"""Fixed24-cell restricted G2 entrypoint. Never a production release API."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'outputs/research/five-dataset-three-methods-20261102-v1/restricted'
G1=ROOT/'outputs/research/g0-g1-historical-audit-20261102-v1/restricted'
REFERENCES={'bitcoin':1684,'lastfm':1269,'facebook':3157}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.temporary')
    with temporary.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.chmod(0o600);temporary.replace(path)


def protected():
    paths={Path(p) for p in json.loads((G1/'manifest.json').read_text())['protected_sha256']}
    paths|={p for p in G1.parent.rglob('*') if p.is_file()}
    paths|={ROOT/'AGENTS.md',ROOT/'docs/pdp_research/G0_THEORY_BOUNDARY_REVIEW.md'}
    for d in ('email','bitcoin','lastfm','hepph','facebook'):
        paths|={p for p in (ROOT/'dataset/prepared'/d).iterdir() if p.is_file()}
    for directory in ('experiments/node_dp_slots','experiments/node_pdp','experiments/node_pdp_v2_projected'):
        paths|=set((ROOT/directory).glob('*.py'))
    return {str(p):sha(p) for p in sorted(paths)}


def evaluate_once(model,graph,message,reference):
    from experiments.pdp_research_evaluation import evaluate
    from experiments.pdp_g0g1_diagnostic import closed_neighborhoods,coverage_stats,spearman
    probe={}
    def capture(module,args,result):probe['logits']=result.detach().double().reshape(-1).tolist()
    handle=model.grat.register_forward_hook(capture)
    try:metrics=evaluate(model,graph,message,reference)
    finally:handle.remove()
    logits=probe['logits'];degrees=graph.outdegree();active=[i for i,d in enumerate(degrees) if d>0]
    areas=closed_neighborhoods(graph.vcount(),graph.get_edgelist(),graph.is_directed())
    reach=coverage_stats(areas,metrics['topk_local_ids'])
    if reach['spread']!=metrics['spread']:raise RuntimeError('independent G2 reach/spread mismatch')
    metrics.update(reach)
    bands={'1-4':(1,4),'5-15':(5,15),'16-63':(16,63),'64+':(64,float('inf'))}
    strata={}
    for label,(lo,hi) in bands.items():
        members=[i for i,d in enumerate(degrees) if lo<=d<=hi]
        strata[label]=dict(n=len(members),rho=spearman([logits[i] for i in members],[degrees[i] for i in members]))
    metrics.update(spearman_degree_strata=strata,spearman_all=spearman(logits,degrees),
        spearman_nonzero_degree=spearman([logits[i] for i in active],[degrees[i] for i in active]))
    return metrics


def worker(restricted,tasks,frozen,expected):
    from experiments.pdp_g2_controlled import ControlledPair
    from experiments.run_pdp_research import load_inputs
    from experiments.node_pdp.accounting import base_account,amplify
    from experiments import node_sampling_utility as utility
    utility.runtime()
    outer_draws=0
    try:
        for dataset in ('bitcoin','lastfm','facebook'):
            registry_path=frozen['registries'][dataset]
            registry,spec,train,test,message,test_ids,inputs,evidence=load_inputs(dataset,registry_path,REFERENCES[dataset])
            for seed in (1,2):
                for outer in (0,1):
                    group=[t for t in tasks if t['dataset']==dataset and t['model_seed']==seed and t['outer_index']==outer]
                    for task in group:task['status']='OUTER_SAMPLING'
                    write(restricted/'tasks.json',tasks)
                    pair=ControlledPair(registry,spec,train,model_seed=seed);outer_draws+=1
                    for task in group:
                        run=restricted/task['id'];run.mkdir(mode=0o700)
                        task.update(status='STARTED',started_unix=time.time());write(restricted/'tasks.json',tasks)
                        config=spec.public();config['model_seed']=seed
                        write(run/'configuration.json',dict(algorithm='node_pdp_v2_projected',configuration=config,
                            registry_fingerprint=registry.fingerprint,prepared_domain_size=len(registry.universe),
                            initialization_seed=seed,outer_index=outer,gaussian_replicate=task['gaussian_replicate'],
                            factor_labels_are_not_random_seeds=True,registry_path=registry_path,
                            labels=['restricted','exploratory','idealized-only','machine-uncertified'],
                            publication_allowed=False,shared_outer_pair_joint_release_not_single_model_PDP=True,
                            evidence=evidence,inputs=inputs,ideal_base_account=base_account(spec),
                            marginal_account_by_target={str(e):amplify(base_account(spec)['epsilon'],1e-5,
                                min(1.,__import__('math').expm1(e)/__import__('math').expm1(2))) for e in (1,2,4)}))
                        started=time.perf_counter();result=pair.train_next();training_seconds=time.perf_counter()-started
                        if result.initial_hash!=frozen['initial_model_hashes'][str(seed)]:raise RuntimeError('frozen public initialization hash mismatch')
                        write(run/'model_parameters.json',dict(parameters=result.parameters,initial_state_hash=result.initial_hash))
                        write(run/'diagnostics.json',dict(training=result.diagnostics,sampling=result.sampling_diagnostics))
                        start_eval=time.perf_counter();metrics=evaluate_once(result.model,test,message,REFERENCES[dataset])
                        record=dict(task={k:v for k,v in task.items() if k!='status'},metrics=metrics,
                            training_seconds=training_seconds,evaluation_seconds=time.perf_counter()-start_eval,
                            updates=result.diagnostics['updates'],gaussian_calls=result.diagnostics['noise_calls'],
                            gaussian_coordinates=result.diagnostics['noise_coordinates'],
                            zero_gradient_slots=sum(s['zero_gradient_slots'] for s in result.diagnostics['steps']),
                            projection_trigger_steps=sum(s['clamped_coordinates']>0 for s in result.diagnostics['steps']),
                            projected_coordinate_events=sum(s['clamped_coordinates'] for s in result.diagnostics['steps']),
                            projection_displacement_sum=sum(s['displacement_l2'] for s in result.diagnostics['steps']),
                            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            initialization_hash=result.initial_hash,n=1,std=None)
                        write(run/'result.json',record)
                        task.update(status='COMPLETED',result=str(run/'result.json'),finished_unix=time.time())
                        write(restricted/'tasks.json',tasks)
                        write(restricted/'progress.json',dict(completed=sum(t['status']=='COMPLETED' for t in tasks),outer_draws=outer_draws,
                                                            last_completed=task['id'],last_coverage=metrics['coverage']))
                        del result
                    del pair
        return True
    except BaseException as error:
        for task in tasks:
            if task['status'] in ('STARTED','OUTER_SAMPLING'):
                task.update(status='FAILED' if task['status']=='STARTED' else 'OUTER_FAILED_NO_TRAINING',error_type=type(error).__name__)
        write(restricted/'tasks.json',tasks)
        # No exception values, frame locals, private masks or source states.
        write(restricted/'failure.json',dict(error_type=type(error).__name__,automatic_retry=False,
            completed=sum(t['status']=='COMPLETED' for t in tasks),outer_draws=outer_draws))
        return False
    finally:
        mismatches=[p for p,value in expected.items() if not Path(p).is_file() or sha(p)!=value]
        write(restricted/'integrity_after.json',dict(protected_files=len(expected),changed=mismatches,all_unchanged=not mismatches))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args()
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from experiments.pdp_g2_controlled import task_matrix
    preflight=json.loads(args.preflight.read_text())
    if not preflight['new_tests_success'] or not preflight['regression_success']:raise RuntimeError('synthetic/regression preflight not passed')
    for name,value in preflight['source_sha256'].items():
        if sha(ROOT/name)!=value:raise RuntimeError('implementation changed after preflight')
    original=json.loads((G1/'manifest.json').read_text())['protected_sha256']
    if any(sha(p)!=value for p,value in original.items()):raise RuntimeError('historical G0/G1 protected source/input mismatch')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    restricted=args.output/'restricted';restricted.mkdir(mode=0o700)
    expected=protected();tasks=task_matrix();frozen=json.loads((HISTORY/'tasks_frozen.json').read_text())
    write(restricted/'preflight.json',preflight);write(restricted/'tasks.json',tasks)
    write(restricted/'manifest.json',dict(maximum_models=24,maximum_outer_graphs=12,task_matrix=tasks,
        registry_paths={d:frozen['registries'][d] for d in REFERENCES},references=REFERENCES,
        source_sha256={name:sha(ROOT/name) for name in preflight['source_sha256']},
        protected_sha256=expected,automatic_retry=False,strict_export=False,
        gaussian_source='unchanged SecretGaussian, new instance per trajectory; no recorded/replayed source state',
        grouping='outer graphs nested within initialization; exactly two Gaussian trajectories per graph',
        protocol_sha256=sha(ROOT/'docs/pdp_research/G2_CONTROLLED_PROTOCOL.md')))
    logfile=restricted/'runtime.log'
    with logfile.open('x') as log:
        logfile.chmod(0o600)
        with redirect_stdout(log),redirect_stderr(log):success=worker(restricted,tasks,frozen,expected)
    print(json.dumps(dict(success=success,completed=sum(t['status']=='COMPLETED' for t in tasks),output=str(restricted))))
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
