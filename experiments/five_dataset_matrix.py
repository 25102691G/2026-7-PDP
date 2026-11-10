"""Frozen three-method scientific task matrix; no training algorithm lives here."""
import argparse
import csv
import json
import os
import statistics
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATASETS=('email','bitcoin','lastfm','hepph','facebook')
REFERENCES=dict(email=432,bitcoin=1684,lastfm=1269,hepph=1741,facebook=3157)
HISTORY={
 'email':'outputs/research/email-v2-seed0-20261102-v1/restricted',
 'bitcoin':'outputs/research/bitcoin-v2-prepared-seed0-20261102-v1/training/restricted',
 'lastfm':'outputs/research/lastfm-v2-prepared-seed0-20261102-v1/training/restricted',
 'hepph':'outputs/research/hepph-v2-prepared-seed0-20261102-v1/training/restricted',
 'facebook':'outputs/research/facebook-v2-prepared-seed0-20261102-v1/training/restricted'}
REGISTRY={name:str(Path(path).parents[1]/'restricted/registry.json') for name,path in HISTORY.items() if name!='email'}
REGISTRY['email']='outputs/node_pdp/email-first-real-20261031-v1/restricted/registry.json'


def read(path):return json.loads(Path(path).read_text())


def write(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if exclusive:
        with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
    else:
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        temporary.chmod(0o600);os.replace(temporary,path)
    path.chmod(0o600)


def matrix(root):
    result=[]
    for dataset in DATASETS:
        for algorithm,epsilon in [('PrivIM*',1),('PrivIM*',4),('node_pdp_v1',None),('node_pdp_v2_projected',None)]:
            for seed in range(5):
                label=f'privim_eps{epsilon}' if epsilon else algorithm
                identity=f'{dataset}_{label}_seed{seed}'
                result.append(dict(id=identity,dataset=dataset,algorithm=algorithm,epsilon=epsilon,
                    model_seed=seed,partition_seed=0,status='NEED_RERUN',output=str(Path(root)/'models'/identity)))
    return result


def should_execute(task):return task['status']=='NEED_RERUN'


def pdp_command(task,registry):
    return [sys.executable,str(ROOT/'experiments/run_pdp_research.py'),'--dataset',task['dataset'],
            '--algorithm',task['algorithm'],'--registry',str(registry),'--model-seed',str(task['model_seed']),
            '--reference',str(REFERENCES[task['dataset']]),'--output',task['output']]


def privim_command(task):
    """Original reproduction trainer, one round with an explicit model seed."""
    return [sys.executable,str(ROOT/'experiments/train_models.py'),
        '--dataset',task['dataset'],'--settings','PrivIM+',
        '--target_epsilon',str(task['epsilon']),'--n_iteration','100',
        '--lr','0.005','--loss-penalty','0.5','--batchsize','16',
        '--lr-decay','1.0','--lr-decay-steps','5','--round','1',
        '--seed',str(task['model_seed']),'--model-seed-mode','fixed',
        '--sampling-seed','0','--sampling-seed-mode','fixed',
        '--batch-seed','0','--batch-seed-mode','fixed',
        '--dp-noise-entropy','--dp-noise-stream','independent',
        '--grat-variant','legacy_released','--grat-initialization','released_default',
        '--feature-protocol','ones','--sampling-protocol','paper',
        '--auto-parameters','--privacy-accountant','paper_binomial','--clip-norm','3.0',
        '--protocol-profile','strict','--training-protocol','fixed-iterations',
        '--optimizer-momentum','0.0','--parameter-averaging','none',
        '--sensitivity-noise-factor','1.0','--r','3','--device','cpu',
        '--output-root',task['output']]


def run_privim(root):
    """Separate status file prevents concurrent PDP controller writes from colliding."""
    root=Path(root);path=root/'restricted/privim_tasks.json'
    if path.exists():
        plan=read(path)
    else:
        frozen=read(root/'restricted/tasks_frozen.json')
        plan={'tasks':[t for t in frozen['tasks'] if t['algorithm']=='PrivIM*']}
        for task in plan['tasks']:task['command']=privim_command(task)
        write(path,plan,exclusive=True)
    for task in plan['tasks']:
        if not should_execute(task):continue
        output=Path(task['output']);log=root/'restricted/logs'/f"{task['id']}.log"
        if output.exists() or log.exists():
            task.update(status='FAILED',failure='pre-existing output/log; no overwrite or retry');write(path,plan);continue
        output.mkdir(parents=True,mode=0o700);log.parent.mkdir(exist_ok=True,mode=0o700)
        task.update(status='RUNNING');write(path,plan);start=time.perf_counter()
        try:
            with log.open('x') as stream:
                result=subprocess.run(task['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'training process exit {result.returncode}')
                metadata_paths=list(output.glob('**/*.json'))
                metas=[(p,read(p)) for p in metadata_paths if p.with_suffix('.pt').exists()]
                if not metas:metas=[(p,read(p)) for p in metadata_paths
                                   if 'actual_optimizer_steps' in read(p)]
                if len(metas)!=1:raise RuntimeError(f'expected one model metadata, got {len(metas)}')
                metadata_path,meta=metas[0]
                if (meta['iterations']!=100 or meta['model_seed']!=task['model_seed']
                    or meta['loss_lambda']!=0.5 or meta['dp_noise_seed'] is not None
                    or not meta['dp_noise_entropy_seeded']):
                    raise RuntimeError('actual model metadata violates frozen seed/update/loss/noise contract')
                evaluation=output/'evaluation'
                command=[sys.executable,str(ROOT/'experiments/evaluate_models.py'),
                         '--dataset',task['dataset'],'--settings','PrivIM+','--device','cpu','--k','50',
                         '--models-root',str(output),'--evaluation-root',str(evaluation)]
                task['evaluation_command']=command
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'evaluation process exit {result.returncode}')
            rows=[]
            for csvpath in evaluation.rglob('*.csv'):
                if csvpath.name=='summary.csv':continue
                with csvpath.open() as f:rows.extend(csv.DictReader(f))
            rows=[r for r in rows if int(float(r.get('k','0')))==50 and int(float(r.get('d','0')))==1]
            if len(rows)!=1:raise RuntimeError(f'expected one k50/d1 result, got {len(rows)}')
            row=rows[0]
            if int(float(row['seed']))!=task['model_seed']:raise RuntimeError('evaluation seed mismatch')
            if float(row['celf_n_covered'])!=REFERENCES[task['dataset']]:raise RuntimeError('CELF reference mismatch')
            task.update(status='COMPLETED',metrics=dict(spread=int(float(row['n_covered'])),
                coverage=float(row['coverage_ratio_percent']),celf=float(row['celf_n_covered']),
                training_seconds=meta.get('total_training_seconds'),registry_fingerprint=None,
                domain_size=None,source=str(output),updates=meta['iterations'],metadata=str(metadata_path)))
        except Exception as error:
            task.update(status='FAILED',failure=f'{type(error).__name__}: {error}; see {log}')
        task['elapsed_seconds']=time.perf_counter()-start
        if log.exists():log.chmod(0o600)
        write(path,plan)
        print(task['id'],task['status'],task.get('metrics',{}).get('coverage'),flush=True)
    return plan


def metrics_from_normal(path):
    path=Path(path);cfg=read(path/'configuration.json');complete=read(path/'completion.json');evaluation=read(path/'final_evaluation.json')
    return dict(spread=evaluation['spread'],coverage=evaluation['coverage'],celf=evaluation['reference_spread'],
        training_seconds=complete['training_seconds'],registry_fingerprint=cfg['registry_fingerprint'],
        domain_size=len(cfg['base']['universe']),source=str(path),updates=complete['updates'])


def freeze(root):
    root=Path(root).resolve()
    if not root.is_relative_to(ROOT/'outputs'):raise ValueError('study must stay under outputs')
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    private=root/'restricted';private.mkdir(mode=0o700)
    from experiments.node_dp_slots import mechanism as base
    from experiments.node_sampling_utility import state_hash
    initial={str(seed):state_hash(base.new_model(seed)) for seed in range(5)}
    tasks=matrix(root);registry_paths={}
    expected=dict(B=16,C=3.,K=2,T=100,lr=.005,momentum=0.,s_sum=260.,delta=1e-5,n=40,
                  features='ones32',model='GRAT3 legacy_released',loss_lambda=.5,partition_seed=0)
    for dataset in DATASETS:
        registry=read(ROOT/REGISTRY[dataset]);destination=private/'registries'/f'{dataset}.json'
        write(destination,registry,exclusive=True);registry_paths[dataset]=str(destination)
        path=ROOT/HISTORY[dataset];cfg=read(path/'configuration.json');complete=read(path/'completion.json');diag=read(path/'training_diagnostics.json')
        checks=[cfg['algorithm']=='node_pdp_v2_projected',cfg['model_seed']==0,
                all(cfg['base'][k]==v for k,v in expected.items()),cfg['registry_fingerprint']==registry['fingerprint'],
                complete['status']=='completed',complete['updates']==100,complete['initial_model_sha256']==initial['0'],
                diag['noise_calls']==100,diag['noise_coordinates']==234000,diag['outer_sampling_calls']==1]
        task=next(t for t in tasks if t['dataset']==dataset and t['algorithm']=='node_pdp_v2_projected' and t['model_seed']==0)
        if all(checks):task.update(status='REUSE',evidence=[str(path/'configuration.json'),str(path/'completion.json'),str(path/'training_diagnostics.json'),str(path/'final_evaluation.json')],metrics=metrics_from_normal(path))
        else:task['reuse_rejection']='historical configuration/seed/steps mismatch'
    # Original v1 Email has a different, fully recorded pre-CLI artifact schema.
    old=ROOT/'outputs/node_pdp/email-first-real-20261031-v1/restricted'
    protocol=read(old/'protocol.json');complete=read(old/'completion.json');audit=read(old/'execution_audit.json');evaluation=read(old/'final_evaluation.json')
    if (all(protocol['base'][k]==v for k,v in expected.items()) and protocol['base']['model_seed']==0
        and complete['status']=='completed' and complete['updates']==100 and audit['updates']==100
        and audit['noise_calls']==100 and audit['noise_coordinates']==234000 and audit['outer_sample_calls']==1
        and complete['registry_fingerprint']==read(registry_paths['email'])['fingerprint']
        and protocol['evaluation']==dict(celf_spread=432,dataset='email',ic_p=1,k=50,
           privacy_status='restricted; no public/DP-output assumption',split='test',steps=1)):
        task=next(t for t in tasks if t['dataset']=='email' and t['algorithm']=='node_pdp_v1' and t['model_seed']==0)
        task.update(status='REUSE',evidence=[str(old/n) for n in ['protocol.json','completion.json','execution_audit.json','final_evaluation.json']],
          metrics=dict(spread=evaluation['influence_spread'],coverage=evaluation['coverage_ratio_percent'],celf=432,
            training_seconds=audit['training_seconds'],registry_fingerprint=complete['registry_fingerprint'],domain_size=1005,source=str(old),updates=100))
    for task in tasks:
        if task['algorithm']=='PrivIM*':
            task['evidence']=['PrivIM复现.md: means/std only; original private run directories absent',
                              'curated outputs/runs summaries contain eps_10000 non-private, not epsilon1/4 per-seed results']
            task['reason']='No verifiable matching private100-step/lambda.5 per-initialization result; means cannot be expanded into seeds'
    plan=dict(version='five-dataset-three-methods-v1',root=str(root),registries=registry_paths,initial_model_hashes=initial,
              fixed_pdp=expected,tasks=tasks,cancelled=['Random','Out-degree','base PrivIM','PrivIM+SCS','Non-Private','EGN','HP','HP-GRAT','no-noise ablations'],
              labels=['exploratory','idealized-only','machine-uncertified'],production_release=False)
    write(private/'tasks_frozen.json',plan,exclusive=True);write(private/'tasks.json',plan,exclusive=True)
    with (private/'history_reuse.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','algorithm','epsilon','model_seed','decision','evidence']);writer.writeheader()
        for task in tasks:writer.writerow(dict(dataset=task['dataset'],algorithm=task['algorithm'],epsilon=task['epsilon'],model_seed=task['model_seed'],decision=task['status'],evidence='; '.join(task.get('evidence',[]))))
    (private/'history_reuse.csv').chmod(0o600)
    return plan


def run_pdp(root):
    path=Path(root)/'restricted/tasks.json';plan=read(path)
    for task in plan['tasks']:
        if not task['algorithm'].startswith('node_pdp') or not should_execute(task):continue
        output=Path(task['output'])
        if output.exists():
            task.update(status='FAILED',failure='pre-existing output; never overwrite or retry');write(path,plan);continue
        command=pdp_command(task,plan['registries'][task['dataset']])
        task.update(status='RUNNING',command=command);write(path,plan)
        log=Path(root)/'restricted/logs'/f"{task['id']}.log";log.parent.mkdir(exist_ok=True,mode=0o700)
        start=time.perf_counter()
        with log.open('x') as stream:
            completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        log.chmod(0o600);task['elapsed_seconds']=time.perf_counter()-start
        if completed.returncode==0:
            try:
                completion=read(output/'restricted/completion.json')
                if completion['initial_model_sha256']!=plan['initial_model_hashes'][str(task['model_seed'])]:
                    raise RuntimeError('initialization seed fingerprint mismatch')
                task.update(status='COMPLETED',metrics=metrics_from_normal(output/'restricted'))
            except Exception as error:task.update(status='FAILED',failure=f'{type(error).__name__}: {error}')
        else:task.update(status='FAILED',failure=f'process exit {completed.returncode}; see {log}')
        write(path,plan)
        print(task['id'],task['status'],task.get('metrics',{}).get('coverage'),flush=True)
    return plan


def summarize_tasks(tasks):
    groups={}
    for task in tasks:
        key=(task['dataset'],task['algorithm'],task['epsilon'])
        groups.setdefault(key,[]).append(task)
    rows=[]
    for (dataset,algorithm,epsilon),members in groups.items():
        successful=[t for t in members if t['status'] in {'REUSE','COMPLETED'}]
        values=[t['metrics']['coverage'] for t in successful]
        spreads=[t['metrics']['spread'] for t in successful]
        durations=[t['metrics']['training_seconds'] for t in successful
                   if t['metrics'].get('training_seconds') is not None]
        rows.append(dict(dataset=dataset,algorithm=algorithm,epsilon=epsilon,
            n=len(values),failed_n=sum(t['status']=='FAILED' for t in members),
            pending_n=sum(t['status'] not in {'REUSE','COMPLETED','FAILED'} for t in members),
            coverage_mean=statistics.mean(values) if values else None,
            coverage_sample_sd=statistics.stdev(values) if len(values)>1 else None,
            spread_mean=statistics.mean(spreads) if spreads else None,
            spread_sample_sd=statistics.stdev(spreads) if len(spreads)>1 else None,
            training_seconds_mean=statistics.mean(durations) if durations else None))
    return rows


def csv_write(path,rows,fields=None):
    path=Path(path);fields=fields or list(rows[0])
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:('NA' if row.get(k) is None else row.get(k)) for k in fields})
    path.chmod(0o600)


def aggregate(root):
    root=Path(root);plan=read(root/'restricted/tasks.json')
    originals=read(root/'restricted/privim_tasks.json')
    by_id={task['id']:task for task in originals['tasks']}
    tasks=[by_id.get(task['id'],task) for task in plan['tasks']]
    rows=[]
    for task in tasks:
        metrics=task.get('metrics',{})
        rows.append(dict(dataset=task['dataset'],algorithm=task['algorithm'],epsilon=task['epsilon'],
            model_seed=task['model_seed'],partition_seed=None if task['algorithm']=='PrivIM*' else 0,
            status=task['status'],spread=metrics.get('spread'),coverage=metrics.get('coverage'),
            celf=metrics.get('celf'),training_seconds=metrics.get('training_seconds'),
            wall_seconds_including_eval=task.get('elapsed_seconds'),
            prepared_domain_size=metrics.get('domain_size'),registry_fingerprint=metrics.get('registry_fingerprint'),
            updates=metrics.get('updates'),source=metrics.get('source',task['output']),
            configuration=task.get('command'),failure=task.get('failure'),
            privacy_status='exploratory / idealized-only / machine-uncertified' if task['algorithm']!='PrivIM*'
                           else 'original-paper accounting / exploratory / uncertified'))
    csv_write(root/'restricted/per_seed.csv',rows)
    summary=summarize_tasks(tasks);csv_write(root/'restricted/summary.csv',summary)
    write(root/'restricted/all_tasks.json',{'tasks':tasks},exclusive=True)
    pairs=[]
    for dataset in DATASETS:
        for seed in range(5):
            matching={t['algorithm']:t for t in tasks if t['dataset']==dataset and t['model_seed']==seed
                      and t['algorithm'] in {'node_pdp_v1','node_pdp_v2_projected'}
                      and t['status'] in {'REUSE','COMPLETED'}}
            complete=len(matching)==2
            pairs.append(dict(dataset=dataset,model_seed=seed,
                spread_difference=matching['node_pdp_v2_projected']['metrics']['spread']-
                                  matching['node_pdp_v1']['metrics']['spread'] if complete else None,
                coverage_difference_pp=matching['node_pdp_v2_projected']['metrics']['coverage']-
                                  matching['node_pdp_v1']['metrics']['coverage'] if complete else None,
                status='descriptive; independent hidden trajectories' if complete else 'incomplete'))
    csv_write(root/'restricted/v2_minus_v1.csv',pairs)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run-pdp','run-privim','aggregate']);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077)
    if args.action=='freeze':
        plan=freeze(args.root);print('frozen',len(plan['tasks']),'reuse',sum(t['status']=='REUSE' for t in plan['tasks']))
    elif args.action=='run-pdp':run_pdp(args.root)
    elif args.action=='run-privim':run_privim(args.root)
    else:
        for row in aggregate(args.root):print(row)


if __name__=='__main__':main()
