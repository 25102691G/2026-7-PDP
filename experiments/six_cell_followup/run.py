"""Manual launch controller. plan/preflight/status never train or infer.

Run each frozen task once in an isolated subprocess. Stop on any failure;
resume only unclaimed tasks after successfully completed tasks. No retries.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from .plan import (ROOT,DOC,DEFAULT_OUTPUT,DATASETS,METHODS,REFERENCES,REGISTRIES,
                   matrix,noise_plan,summarize,validate_task)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    data=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x') as f:f.write(data)
    else:
        descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
        temp=Path(name)
        try:
            with os.fdopen(descriptor,'w') as f:f.write(data)
            os.replace(temp,path)
        finally:
            if temp.exists():temp.unlink()
    path.chmod(0o600)


def output_root(path):
    p=Path(path).resolve()
    if not p.is_relative_to((ROOT/'outputs/research').resolve()) or p==ROOT/'outputs/research':
        raise ValueError('new dedicated study directory must be under outputs/research')
    return p


def protected_hashes():
    paths=[]
    for folder in ('experiments','model','baselines'):
        paths.extend((ROOT/folder).rglob('*.py'))
    paths += [ROOT/'AGENTS.md',ROOT/'tests/test_dataset_parameter_screen.py',ROOT/'tests/test_six_cell_followup.py',DOC/'PROPOSAL.md',DOC/'CANDIDATES_DRAFT.json']
    for d in DATASETS:
        paths.extend(p for p in (ROOT/'dataset/prepared'/d).iterdir() if p.is_file())
        paths.append(ROOT/REGISTRIES[d])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def freeze(root):
    root=output_root(root)
    if root.exists():raise FileExistsError('study already exists; never overwrite a plan')
    hashes=protected_hashes()
    manifest=dict(version='six-cell-followup-v1',tasks=matrix(),methods=list(METHODS),datasets=list(DATASETS),
        seeds=[0,1,2],planned_fits=144,planned_updates=31950,planned_final_evaluations=144,
        registries=REGISTRIES,references=REFERENCES,source_and_input_sha256=hashes,
        public_constants=dict(model='GRAT3 legacy_released/released_default',features='ones32',
            loss_lambda=.5,lr=.005,momentum=0.,B=16,C=3.,K=2,L=200,tau=.3,mu=1.,
            sampling_seed=0,batch_seed=0,partition_seed=0,boundary_divisor=2,
            private_quota=[4,4,4,4],Q_per_slot=64,nonprivate_Q=256,
            projection='private only: final32 weights after each update; bias/initialization unchanged'),
        noise_plans={m:{str(t):noise_plan(m,t) for t in sorted({x['T'] for x in matrix() if x['method']==m})} for m in METHODS},
        limits=dict(total_worker_wall_seconds=14400,task_wall_seconds=3600,peak_rss_bytes=4*1024**3,minimum_free_disk_bytes=2*1024**3),
        selection='complete batch only; per-cell mean gate then lowest sample SD, higher minimum, fewer updates, candidate order; no automatic table edits',
        privacy_scope='conditional ideal single-model bound only; research selection and joint publication NOT covered',
        created_unix_seconds=time.time())
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    save(root/'manifest.json',manifest,exclusive=True)
    save(root/'manifest_seal.json',dict(sha256=sha(root/'manifest.json')),exclusive=True)
    print(json.dumps(dict(status='PLAN_FROZEN_NO_REAL_EXECUTION',output=str(root),fits=144,updates=31950)))
    return manifest


def checked_manifest(root):
    if sha(root/'manifest.json')!=read(root/'manifest_seal.json')['sha256']:raise ValueError('manifest seal mismatch')
    manifest=read(root/'manifest.json')
    if manifest['version']!='six-cell-followup-v1' or manifest['tasks']!=matrix():
        raise ValueError('frozen design differs from reviewed source')
    if manifest['references']!=REFERENCES or manifest['registries']!=REGISTRIES:raise ValueError('input bindings changed')
    if manifest['planned_fits']!=144 or manifest['planned_updates']!=31950:raise ValueError('execution cap mismatch')
    if manifest['limits']!=dict(total_worker_wall_seconds=14400,task_wall_seconds=3600,peak_rss_bytes=4*1024**3,minimum_free_disk_bytes=2*1024**3):raise ValueError('reviewed resource limits changed')
    for name,h in manifest['source_and_input_sha256'].items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=h:raise RuntimeError(f'frozen input/source changed: {name}')
    return manifest


def preflight(root):
    manifest=checked_manifest(root)
    # Numeric accounting only; no prepared-graph loading, model creation or RNG calls.
    from experiments.uniform_node_dp_eps4.accounting import gaussian_bound
    numerical=[]
    for method in METHODS[1:]:
        sigma100=manifest['noise_plans'][method]['100']['sigma_sum']
        original=gaussian_bound(sigma_sum=sigma100,steps=100)['epsilon_numeric']
        for t in sorted({x['T'] for x in matrix() if x['method']==method}):
            p=manifest['noise_plans'][method][str(t)]
            if p!=noise_plan(method,t):raise ValueError('noise plan modified')
            bound=gaussian_bound(sigma_sum=p['sigma_sum'],steps=t)['epsilon_numeric']
            if bound>original+1e-12:raise ValueError('longer-T privacy accounting exceeded original')
            numerical.append(dict(method=method,T=t,sigma_sum=p['sigma_sum'],base_epsilon_numeric=bound))
    for d in DATASETS:
        meta=read(ROOT/'dataset/prepared'/d/'metadata.json')
        if meta['train_ratio']!=.5 or meta['seed']!=0 or meta['isolated_policy']!='active':raise ValueError('prepared split changed')
    versions={name:importlib.metadata.version(name) for name in ('torch','dgl','igraph','numpy','scipy','psutil')}
    if shutil.disk_usage(root).free<manifest['limits']['minimum_free_disk_bytes']:raise RuntimeError('less than2GiB free disk')
    result=dict(status='PREFLIGHT_PASS_NO_REAL_EXECUTION',versions=versions,hashes_checked=len(manifest['source_and_input_sha256']),
        accounting=numerical,new_training=0,new_inference=0,models_claimed=len(list((root/'models').glob('*/claim.json'))) if (root/'models').exists() else 0)
    save(root/'preflight.json',result)
    print(json.dumps(result))
    return manifest


def claim(root,task_id,task=None):
    if Path(task_id).name!=task_id:raise ValueError('invalid task identity')
    p=root/'models'/task_id;p.mkdir(parents=True,exist_ok=False,mode=0o700)
    save(p/'claim.json',dict(task=task,attempts=1,status='CONSUMED_NO_RETRY'),exclusive=True)
    return p


def pending_tasks(root,tasks):
    pending=[]
    for t in tasks:
        p=root/'models'/t['id']
        if not p.exists():pending.append(t);continue
        if (p/'failure.json').exists() or not (p/'completion.json').exists():
            raise RuntimeError(f'failed/interrupted/inflight task cannot be retried: {t["id"]}')
        c=read(p/'completion.json')
        if c.get('status')!='COMPLETED' or c.get('task')!=t:raise RuntimeError('invalid completion marker')
    return pending


def nonprivate_command(task,directory):
    validate_task(task)
    argv=[sys.executable,str(ROOT/'experiments/train_models.py'),'--dataset',task['dataset'],
        '--settings','PrivIM+','--no-dp','--model_name','GRAT3',
        '--grat-variant','legacy_released','--grat-initialization','released_default',
        '--feature-protocol','ones','--loss-penalty','0.5','--n_iteration',str(task['T']),
        '--subgraph_size',str(task['n']),'--batchsize','16','--lr','0.005','--optimizer-momentum','0',
        '--k_train','50','--d','1','--lr-decay','1','--lr-decay-steps','5',
        '--score_upper_bound',str(task['M']),'--walk-length','200','--restart-probability','0.3',
        '--frequency-decay','1','--boundary-divisor','2','--num_subgraph','256',
        '--sampling-protocol','paper','--r','3','--seed',str(task['seed']),'--model-seed-mode','fixed',
        '--round','1','--sampling-seed','0','--sampling-seed-mode','fixed','--batch-seed','0',
        '--batch-seed-mode','fixed','--protocol-profile','strict','--training-protocol','fixed-iterations',
        '--parameter-averaging','none','--device','cpu','--output-root',str(directory/'original_trainer')]
    return argv


def invoke_original(command):
    """Execute the unchanged CLI in this fresh, already thread-limited worker."""
    import runpy
    previous=sys.argv
    try:
        sys.argv=command[1:]
        runpy.run_path(command[1],run_name='__main__')
    finally:sys.argv=previous


def fit_nonprivate(task,directory):
    import torch
    from experiments.node_dp_slots import mechanism as base
    command=nonprivate_command(task,directory)
    save(directory/'training_command.json',command,exclusive=True)
    save(directory/'fit_started.json',dict(task_id=task['id'],attempts=1),exclusive=True)
    invoke_original(command)
    metas=list((directory/'original_trainer').glob('**/model/*.json'))
    if len(metas)!=1:raise RuntimeError('expected one original checkpoint/metadata')
    meta=read(metas[0])
    expected=dict(with_dp=False,non_private_clipping=False,actual_optimizer_steps=task['T'],subgraph_size=task['n'],
        loss_lambda=.5,learning_rate=.005,batch_size=16,model_seed=task['seed'],sampling_seed=0,batch_seed=0,
        gradient_reduction='batch_mean',parameter_averaging='none',feature_protocol='ones')
    if any(meta.get(k)!=v for k,v in expected.items()) or meta['sampling']['frequency_threshold']!=task['M']:
        raise RuntimeError('original Non-Private metadata mismatch')
    model=base.new_model(task['seed'])
    model.load_state_dict(torch.load(metas[0].with_suffix('.pt'),map_location='cpu',weights_only=True),strict=True)
    return model,dict(updates=meta['actual_optimizer_steps'],gaussian_vectors=0,projections=0,
                      outer_sampling_calls=0,subgraph_container_size=meta['subgraph_container_size'])


def worker(root,task_id):
    """Internal entry: the parent must consume a once-only task first."""
    if os.getpid()!=os.getpgrp():raise PermissionError('worker requires its own isolated process group')
    os.umask(0o077);manifest=checked_manifest(root)
    task=next(t for t in manifest['tasks'] if t['id']==task_id)
    directory=root/'models'/task_id
    if read(directory/'claim.json')['task']!=task:raise PermissionError('missing matching once-only claim')
    save(directory/'worker_entered.json',dict(pid=os.getpid()),exclusive=True)
    started=time.perf_counter()
    try:
        import threading
        import psutil
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        process=psutil.Process();peak=[0]
        def peak_rss():
            total=0
            for child in [process]+process.children(recursive=True):
                try:total+=child.memory_info().rss
                except psutil.NoSuchProcess:pass
            peak[0]=max(peak[0],total)
            return peak[0]
        stop=threading.Event()
        def monitor():
            while not stop.wait(.5):
                if peak_rss()>manifest['limits']['peak_rss_bytes']:
                    save(directory/'failure.json',dict(status='FAILED_NO_RETRY',reason='RSS_LIMIT'),exclusive=True)
                    os.killpg(os.getpgrp(),signal.SIGKILL)
        threading.Thread(target=monitor,daemon=True).start()
        from experiments.run_pdp_research import load_inputs
        from experiments.node_sampling_utility import state_hash
        from experiments.pdp_research_evaluation import evaluate
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as production
        guard,ledger=production._PROCESS_GUARD,base._PROCESS_LEDGER
        before=(guard._used,ledger.models)
        registry,spec,graph,test,message,_,_,evidence=load_inputs(task['dataset'],ROOT/REGISTRIES[task['dataset']],REFERENCES[task['dataset']])
        configuration=dict(task=task,constants=manifest['public_constants'],noise=noise_plan(task['method'],task['T']),
            registry_fingerprint=registry.fingerprint,input_checks=evidence,
            role='six-cell exploratory development candidate',original_table6_1_unchanged=True,
            publication_allowed=False,machine_dp_certified=False)
        save(directory/'configuration.json',configuration,exclusive=True)
        begin=time.perf_counter()
        if task['method']=='nonprivate':model,counts=fit_nonprivate(task,directory)
        else:
            from .kernel import fit_private
            model,counts=fit_private(task,registry,spec,graph,directory,
                lambda info:save(directory/'training_progress.json',info))
        training_seconds=time.perf_counter()-begin
        if counts['updates']!=task['T']:raise RuntimeError('update budget mismatch')
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise RuntimeError('nonfinite checkpoint')
        checkpoint=directory/'model.pt'
        if checkpoint.exists():raise FileExistsError('never overwrite checkpoint')
        torch.save(model.state_dict(),checkpoint);checkpoint.chmod(0o600)
        # Exactly one final test query. No initial/intermediate test query or CELF recomputation.
        save(directory/'evaluation_started.json',dict(attempts=1),exclusive=True)
        begin=time.perf_counter();result=evaluate(model,test,message,REFERENCES[task['dataset']]);evaluation_seconds=time.perf_counter()-begin
        save(directory/'final_evaluation.json',result,exclusive=True)
        save(directory/'training_counts.json',counts,exclusive=True)
        if (guard._used,ledger.models)!=before or guard is not production._PROCESS_GUARD or ledger is not base._PROCESS_LEDGER:
            raise RuntimeError('production control mutation')
        checked_manifest(root)
        if peak_rss()>manifest['limits']['peak_rss_bytes']:raise RuntimeError('RSS limit')
        stop.set()
        save(directory/'completion.json',dict(status='COMPLETED',task=task,attempts=1,failures=0,retries=0,n=1,std=None,
            training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,wall_seconds=time.perf_counter()-started,
            peak_rss_bytes=peak_rss(),checkpoint_sha256=sha(checkpoint),final_model_sha256=state_hash(model),
            evaluation_sha256=sha(directory/'final_evaluation.json'),counts=counts,production_controls_unchanged=True),exclusive=True)
    except BaseException as error:
        if not (directory/'failure.json').exists():
            save(directory/'failure.json',dict(status='FAILED_NO_RETRY',exception_type=type(error).__name__,message=str(error)),exclusive=True)
        raise


def collect(root,manifest):
    rows=[];states=[]
    for t in manifest['tasks']:
        p=root/'models'/t['id'];state='UNEXECUTED'
        if (p/'failure.json').exists():state='FAILED_NO_RETRY'
        elif (p/'completion.json').exists():
            c=read(p/'completion.json');e=read(p/'final_evaluation.json')
            if c['task']!=t or c['counts']['updates']!=t['T'] or c['status']!='COMPLETED':raise ValueError('completion mismatch')
            if sha(p/'model.pt')!=c['checkpoint_sha256'] or sha(p/'final_evaluation.json')!=c['evaluation_sha256']:raise ValueError('result/checkpoint digest mismatch')
            if read(p/'configuration.json')['task']!=t or e['reference_spread']!=REFERENCES[t['dataset']]:raise ValueError('configuration/denominator mismatch')
            rows.append(dict(task=t,coverage=e['coverage'],spread=e['spread'],celf=e['reference_spread'],
                training_seconds=c['training_seconds'],evaluation_seconds=c['evaluation_seconds'],source=str((p/'final_evaluation.json').relative_to(ROOT))))
            state='COMPLETED'
        elif p.exists():state='CLAIMED_OR_INTERRUPTED'
        states.append(dict(**t,status=state,attempts=int(p.exists()),failures=int(state=='FAILED_NO_RETRY'),retries=0,unexecuted=int(state=='UNEXECUTED')))
    return rows,states


def write_reports(root,manifest):
    with (root/'reports.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _write_reports_locked(root,manifest)


def _write_reports_locked(root,manifest):
    rows,states=collect(root,manifest)
    report=summarize(manifest['tasks'],rows)
    for cell in report['cells']:
        group=[s for s in states if (s['dataset'],s['method'],s['candidate'])==(cell['dataset'],cell['method'],cell['candidate'])]
        for field in ('attempts','failures','retries','unexecuted'):cell[field]=sum(s[field] for s in group)
    save(root/'summary.json',report);save(root/'task_status.json',states)
    for name,records in [('task_status.csv',states),('per_seed.csv',[dict(**r['task'],**{k:v for k,v in r.items() if k!='task'}) for r in rows]),('cell_statistics.csv',report['cells'])]:
        if records:
            with (root/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
                writer.writerows({k:('NA' if v is None else v) for k,v in r.items()} for r in records)
            (root/name).chmod(0o600)
    status=dict(status=report['status'],completed=len(rows),planned=len(states),
        failed=sum(s['failures'] for s in states),unexecuted=sum(s['unexecuted'] for s in states),
        claimed_or_interrupted=sum(s['status']=='CLAIMED_OR_INTERRUPTED' for s in states),retries=0)
    save(root/'status.json',status);print(json.dumps(status),flush=True)
    return status


def status_readonly(root,manifest):
    rows,states=collect(root,manifest)
    result=dict(status='COMPLETE_DEVELOPMENT_SCREEN' if len(rows)==len(states) else 'INCOMPLETE',
        completed=len(rows),planned=len(states),attempts=sum(s['attempts'] for s in states),
        failed=sum(s['failures'] for s in states),unexecuted=sum(s['unexecuted'] for s in states),
        claimed_or_interrupted=sum(s['status']=='CLAIMED_OR_INTERRUPTED' for s in states),retries=0)
    budget=root/'worker_budget.json'
    if budget.exists():result['worker_budget']=read(budget)
    print(json.dumps(result),flush=True)
    return result


def remaining_worker_budget(root,limit):
    path=root/'worker_budget.json'
    if not path.exists():
        if (root/'models').exists():raise RuntimeError('missing worker budget for existing execution')
        return float(limit)
    budget=read(path)
    if budget['active_task'] is not None:raise RuntimeError('interrupted active worker cannot resume')
    value=budget['consumed_seconds']
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('invalid budget')
    return max(0.,limit-value)


def charge_worker_budget(root,elapsed):
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid elapsed worker time')
    budget=read(root/'worker_budget.json')
    save(root/'worker_budget.json',dict(consumed_seconds=budget['consumed_seconds']+elapsed,active_task=None))


def run(root,resume=False):
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=preflight(root)
        if not resume and (root/'models').exists():raise FileExistsError('existing execution; use --resume for successfully completed/pending tasks only')
        collect(root,manifest)
        tasks=pending_tasks(root,manifest['tasks'])
        write_reports(root,manifest)
        for task in tasks:
            checked_manifest(root)
            remaining=remaining_worker_budget(root,manifest['limits']['total_worker_wall_seconds'])
            if remaining<=0:raise RuntimeError('total worker budget exhausted; no additional task claimed')
            if shutil.disk_usage(root).free<manifest['limits']['minimum_free_disk_bytes']:raise RuntimeError('less than2GiB free disk')
            directory=claim(root,task['id'],task)
            prior=manifest['limits']['total_worker_wall_seconds']-remaining
            save(root/'worker_budget.json',dict(consumed_seconds=prior,active_task=task['id']))
            worker_started=time.monotonic()
            print(f'START {task["id"]} T={task["T"]} n={task["n"]} M={task["M"]}',flush=True)
            process=None
            try:
                with (directory/'worker.log').open('x') as log:
                    process=subprocess.Popen([sys.executable,'-m','experiments.six_cell_followup.run','worker',
                        '--output',str(root),'--task-id',task['id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=process.wait(timeout=min(manifest['limits']['task_wall_seconds'],remaining))
                    if code!=0:raise RuntimeError(f'worker exit {code}; see {directory/"worker.log"}')
                if not (directory/'completion.json').exists():raise RuntimeError('missing completion marker')
            except BaseException as error:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                if not (directory/'failure.json').exists():
                    save(directory/'failure.json',dict(status='FAILED_NO_RETRY',exception_type=type(error).__name__,message=str(error)),exclusive=True)
                write_reports(root,manifest)
                raise
            finally:
                charge_worker_budget(root,time.monotonic()-worker_started)
            write_reports(root,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','preflight','run','status','summarize','worker'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--task-id')
    args=parser.parse_args();os.umask(0o077);root=output_root(args.output)
    if args.action=='plan':freeze(root)
    elif args.action=='preflight':preflight(root)
    elif args.action=='run':run(root,args.resume)
    elif args.action=='worker':
        if not args.task_id:parser.error('worker requires task-id')
        worker(root,args.task_id)
    elif args.action=='status':status_readonly(root,checked_manifest(root))
    else:write_reports(root,checked_manifest(root))


if __name__=='__main__':main()
