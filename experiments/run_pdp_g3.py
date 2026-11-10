"""Fixed25 fresh trajectories,50 same-trajectory evaluations; no release API."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'outputs/research/five-dataset-three-methods-20261102-v1/restricted'
G2=ROOT/'outputs/research/g2-controlled-attribution-20261102-v1/restricted'
REFERENCES={'email':432,'bitcoin':1684,'lastfm':1269,'hepph':1741,'facebook':3157}


def historical_diff(expected):
    """Only Finder display metadata is not a scientific-input integrity gate."""
    import hashlib
    result=dict(research_mismatches=[],platform_metadata_mismatches=[])
    for name,value in expected.items():
        p=Path(name);actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
        if actual!=value:
            category='platform_metadata_mismatches' if p.name=='.DS_Store' else 'research_mismatches'
            result[category].append(dict(path=str(p),expected=value,actual=actual))
    return result


def protected():
    from experiments.run_pdp_g2 import sha
    previous=json.loads((G2/'manifest.json').read_text())
    paths={Path(p) for p in previous['protected_sha256']}
    paths|={ROOT/p for p in previous['source_sha256']}
    paths|={p for p in G2.parent.rglob('*') if p.is_file()}
    return {str(p):sha(p) for p in sorted(paths) if p.name!='.DS_Store'}


def worker(restricted,tasks,frozen,expected):
    from experiments.pdp_g3_average import AverageSession
    from experiments.run_pdp_g2 import evaluate_once,write,sha
    from experiments.run_pdp_research import load_inputs
    from experiments.pdp_g0g1_diagnostic import _state_hash
    from experiments.node_pdp.accounting import base_account,amplify
    from experiments import node_sampling_utility as utility
    import math
    utility.runtime()
    try:
        for dataset,reference in REFERENCES.items():
            registry,spec,train,test,message,test_ids,inputs,evidence=load_inputs(dataset,frozen['registries'][dataset],reference)
            for seed in range(5):
                task=next(t for t in tasks if t['dataset']==dataset and t['model_seed']==seed)
                run=restricted/task['id'];run.mkdir(mode=0o700)
                task.update(status='STARTED',started_unix=time.time());write(restricted/'tasks.json',tasks)
                session=AverageSession(registry,spec,'node_pdp_v2_projected',model_seed=seed)
                initial_hash=_state_hash(session.model)
                if initial_hash!=frozen['initial_model_hashes'][str(seed)]:raise RuntimeError('public initialization mismatch')
                account=base_account(spec)
                write(run/'configuration.json',dict(training_algorithm='node_pdp_v2_projected',candidate='PDP-v2-Avg',
                    configuration={**spec.public(),'model_seed':seed},registry_fingerprint=registry.fingerprint,
                    prepared_domain_size=len(registry.universe),registry_path=frozen['registries'][dataset],
                    initial_state_hash=initial_hash,average=dict(steps=100,first_step=1,last_step=100,weights='uniform1/100',
                        accumulation_dtype='FP64',output_dtype='FP32',initial_state_included=False),
                    inputs=inputs,evidence=evidence,ideal_base_account=account,
                    marginal_account_by_target={str(e):amplify(account['epsilon'],1e-5,min(1.,math.expm1(e)/math.expm1(2))) for e in (1,2,4)},
                    labels=['restricted','exploratory','idealized-only','machine-uncertified'],
                    publication_allowed=False,joint_last_and_mean_single_transcript=True))
                started=time.perf_counter();result=session.run(train);training_seconds=time.perf_counter()-started
                last_hash=_state_hash(session.model)
                write(run/'last_parameters.json',dict(parameters=result.last.parameters,parameter_state_hash=last_hash))
                write(run/'average_parameters.json',dict(parameters=result.average_parameters,parameter_state_hash=_state_hash(session.average_model)))
                write(run/'diagnostics.json',dict(original=result.last.diagnostics,average_checks=result.checks,
                                                  observation=result.training_observation))
                evaluated={};elapsed={}
                for label,model in (('v2_last',session.model),('PDP-v2-Avg',session.average_model)):
                    begin=time.perf_counter();m=evaluate_once(model,test,message,reference);elapsed[label]=time.perf_counter()-begin
                    m['topk_stable_ids']=[test_ids[i] for i in m['topk_local_ids']];evaluated[label]=m
                if _state_hash(session.model)!=last_hash:raise RuntimeError('evaluation changed original parameters')
                write(run/'result.json',dict(task={k:v for k,v in task.items() if k!='status'},metrics=evaluated,
                    paired_coverage_difference=evaluated['PDP-v2-Avg']['coverage']-evaluated['v2_last']['coverage'],
                    training_seconds=training_seconds,evaluation_seconds=elapsed,initialization_hash=initial_hash,
                    updates=result.last.diagnostics['updates'],bernoulli_sampling_calls=result.last.diagnostics['outer_sampling_calls'],
                    gaussian_calls=result.last.diagnostics['noise_calls'],gaussian_coordinates=result.last.diagnostics['noise_coordinates'],
                    actual_zero_gradient_slots=sum(s['zero_gradient_slots'] for s in result.last.diagnostics['steps']),
                    projection_trigger_steps=sum(s['clamped_coordinates']>0 for s in result.last.diagnostics['steps']),
                    max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,n=1,std=None))
                task.update(status='COMPLETED',result=str(run/'result.json'),finished_unix=time.time())
                write(restricted/'tasks.json',tasks)
                write(restricted/'progress.json',dict(completed=sum(t['status']=='COMPLETED' for t in tasks),
                    last_completed=task['id'],last_coverage=evaluated['v2_last']['coverage'],
                    average_coverage=evaluated['PDP-v2-Avg']['coverage']))
                del result,session
        return True
    except BaseException as error:
        for task in tasks:
            if task['status']=='STARTED':task.update(status='FAILED',error_type=type(error).__name__)
        write(restricted/'tasks.json',tasks)
        write(restricted/'failure.json',dict(error_type=type(error).__name__,automatic_retry=False,
            completed=sum(t['status']=='COMPLETED' for t in tasks)))
        return False
    finally:
        changed=[p for p,value in expected.items() if not Path(p).is_file() or sha(p)!=value]
        write(restricted/'integrity_after.json',dict(protected_files=len(expected),changed=changed,all_unchanged=not changed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args()
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from experiments.pdp_g3_average import task_matrix
    from experiments.run_pdp_g2 import sha,write
    preflight=json.loads(args.preflight.read_text())
    if not preflight['tests_passed']:raise RuntimeError('synthetic/regression preflight failed')
    if any(sha(ROOT/p)!=value for p,value in preflight['source_sha256'].items()):raise RuntimeError('G3 changed after tests')
    previous=json.loads((G2/'manifest.json').read_text())
    historical=historical_diff(previous['protected_sha256'])
    if historical['research_mismatches']:raise RuntimeError('historical research/source/data protected mismatch')
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False);restricted=args.output/'restricted';restricted.mkdir(mode=0o700)
    tasks=task_matrix();frozen=json.loads((HISTORY/'tasks_frozen.json').read_text());expected=protected()
    write(restricted/'preflight.json',preflight);write(restricted/'tasks.json',tasks)
    write(restricted/'manifest.json',dict(maximum_trajectories=25,maximum_final_evaluations=50,tasks=tasks,
        references=REFERENCES,registries=frozen['registries'],source_sha256=preflight['source_sha256'],
        protected_sha256=expected,automatic_retry=False,strict_export=False,
        preflight_platform_metadata_differences=historical['platform_metadata_mismatches'],
        platform_metadata_policy='only.DS_Store excluded; never dataset metadata.json or any scientific artifact',
        average='FP64 sequential sum(theta1..theta100)/100 thenFP32; theta0 excluded; no feedback',
        protocol_sha256=sha(ROOT/'docs/pdp_research/G3_AVERAGE_PROTOCOL.md'),
        labels=['exploratory','idealized-only','machine-uncertified'],g4_g5_authorized=False))
    for source,target in ((ROOT/'docs/pdp_research/G3_AVERAGE_PROTOCOL.md',restricted/'PROTOCOL_FROZEN.md'),
                          (ROOT/'docs/pdp_research/G3_AVERAGE_THEORY.md',restricted/'THEORY_FROZEN.md')):
        with target.open('xb') as f:f.write(source.read_bytes())
        target.chmod(0o600)
    logfile=restricted/'runtime.log'
    with logfile.open('x') as log:
        logfile.chmod(0o600)
        with redirect_stdout(log),redirect_stderr(log):success=worker(restricted,tasks,frozen,expected)
    print(json.dumps(dict(success=success,completed=sum(t['status']=='COMPLETED' for t in tasks),output=str(restricted))))
    return 0 if success else 1


if __name__=='__main__':raise SystemExit(main())
