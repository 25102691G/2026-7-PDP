"""Read historical JSON/CSV only: no model runtime, inference, sampling or training.

Writes a new preparation directory; never rewrites historical artifacts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/'outputs/research/five-dataset-three-methods-20261102-v1/restricted'
DATASETS=('email','bitcoin','lastfm','hepph','facebook')
REFERENCES=dict(zip(DATASETS,(432,1684,1269,1741,3157)))
DOMAINS=dict(zip(DATASETS,(1005,4411,6292,10513,20091)))
EXPECTED=dict(B=16,C=3.,K=2,T=100,dimension=2340,s_sum=260.,s_mean=16.25,lr=.005,
    momentum=0.,partition_seed=0,sampling_seed=0,batch_seed=0,quota=[4]*4,
    model='GRAT3 legacy_released',features='ones32',loss_lambda=.5,n=40,M_SCS=4,
    walk_length=200,tau=.3,mu=1.,boundary_divisor=2,Q_per_slot=64,delta=1e-5)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(ROOT))


def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    evidence={};rows=[];candidates=[];baseconfigs={}
    def load(path):
        evidence[relative(path)]=sha(path)
        return read(path)
    def table(path):
        evidence[relative(path)]=sha(path)
        with Path(path).open() as f:return list(csv.DictReader(f))
    frozen=load(HISTORY/'tasks_frozen.json');tasks=load(HISTORY/'all_tasks.json')['tasks']
    registries={d:load(HISTORY/'registries'/f'{d}.json') for d in DATASETS}
    # Historical checkpoints are not executed. Verification uses recorded public initialization hashes.
    for task in tasks:
        if task['algorithm'] not in ('node_pdp_v1','node_pdp_v2_projected'):continue
        d=task['dataset'];seed=task['model_seed'];alg=task['algorithm'];projected=alg.endswith('projected')
        p=Path(task['metrics']['source']);checks={}
        old=d=='email' and seed==0 and not projected
        if old:
            cfg=load(p/'protocol.json');complete=load(p/'completion.json');diag=load(p/'execution_audit.json')
            ev=load(p/'final_evaluation.json');model=load(p/'idealized_model.json')
            b=cfg['base'];spread=ev['influence_spread'];coverage=ev['coverage_ratio_percent']
            checks['outer_once']=diag['outer_sample_calls']==1
            checks['double_clip']=diag['double_clip_calls']==100 and diag['per_position_clip_calls']==1600
            checks['evaluation']=cfg['evaluation']['ic_p']==1 and cfg['evaluation']['steps']==1 and cfg['evaluation']['k']==50 and cfg['evaluation']['celf_spread']==432
            checks['seed']=b['model_seed']==0
            fingerprint=cfg['registry_fingerprint'];seconds=diag['training_seconds']
            checks['projection']=(model['configuration']['base_protocol']=='stable-slot-node-dp-v1-idealized' and any(v<0 for v in model['parameters'][2242:2274]))
            account=complete['base_account']
        else:
            cfg=load(p/'configuration.json');complete=load(p/'completion.json');diag=load(p/'training_diagnostics.json')
            ev=load(p/'final_evaluation.json');model=load(p/'model.json');inputs=load(p/'input_checks.json')
            b=cfg['base'];spread=ev['spread'];coverage=ev['coverage']
            checks['algorithm']=cfg['algorithm']==alg and model['configuration']['algorithm']==alg
            checks['seed']=cfg['model_seed']==seed and b['model_seed']==seed
            checks['initial_fingerprint']=complete['initial_model_sha256']==frozen['initial_model_hashes'][str(seed)]
            checks['outer_once']=diag['outer_sampling_calls']==1
            checks['projection']=diag['projection_calls']==(100 if projected else 0)
            if projected:checks['nonnegative']=diag['all_projected_weights_nonnegative'] and all(f['nonnegative'] for f in diag['steps'])
            checks['evaluation']=cfg['evaluation']==dict(ic_p=1,steps=1,k=50,celf_reference=REFERENCES[d]) and ev['reference_spread']==REFERENCES[d]
            checks['input_contract']=all(inputs[k] for k in ('prepared_split_unchanged','stable_ids_checked','train_test_disjoint'))
            checks['model_config']=model['configuration']['base']==b
            fingerprint=cfg['registry_fingerprint'];seconds=complete['training_seconds']
            account=complete['base_account_reference']
        checks['configuration']=all(b.get(k)==v for k,v in EXPECTED.items())
        checks['complete']=complete['status']=='completed' and complete['updates']==diag['updates']==100
        checks['Gaussian']=diag['noise_calls']==100 and diag['noise_coordinates']==234000
        checks['model_finite']=len(model['parameters'])==2340 and all(math.isfinite(v) for v in model['parameters'])
        reg=registries[d];public=reg['registry']
        checks['registry']=fingerprint==reg['fingerprint']==hashlib.sha256(json.dumps(public,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        checks['domain']=b['universe']==public['universe'] and len(b['universe'])==DOMAINS[d]
        checks['account']=account['s_sum']==260 and account['T']==100 and account['delta']==1e-5 and abs(account['epsilon']-1.982092738991924)<1e-12
        checks['metrics']=spread==task['metrics']['spread'] and abs(coverage-task['metrics']['coverage'])<1e-10 and abs(100*spread/REFERENCES[d]-coverage)<1e-10
        if not checks['metrics']:raise RuntimeError('historical metric mismatch: stop without rewriting results')
        baseconfigs[d,seed,alg]=b
        rows.append(dict(dataset=d,method='PDP v2' if projected else 'Matched PDP',model_seed=seed,
            decision='REUSE' if all(checks.values()) else 'INCOMPATIBLE',spread=spread,coverage=coverage,
            celf=REFERENCES[d],training_seconds=seconds,source=relative(p),checks=json.dumps(checks,sort_keys=True),
            reason='all frozen checks pass' if all(checks.values()) else ','.join(k for k,v in checks.items() if not v)))
    matches=[]
    for d in DATASETS:
        for seed in range(5):
            a=baseconfigs[d,seed,'node_pdp_v1'];b=baseconfigs[d,seed,'node_pdp_v2_projected']
            matches.append(dict(dataset=d,model_seed=seed,base_config_identical=a==b,
                claim='same mechanism except update projection; independently sampled hidden trajectories, no causal pairing'))
    # Four original reproduction summaries: retain aggregate values; never fabricate per-seed values.
    for p in sorted((ROOT/'outputs/runs').glob('*/manifest.json')):
        m=load(p);cfg=m['configuration'];summary=p.parent/'evaluation/summary.csv'
        for s in table(summary):
            d=s['graph'].split('_test')[0]
            if d not in DATASETS:continue
            for seed in range(5):
                candidates.append(dict(dataset=d,model_seed=seed,decision='INCOMPATIBLE',source=relative(p),
                    feature=s['feature_protocol'],steps=cfg['iterations'],n=s['subgraph_size'],
                    full_train_graph='command-level yes; no per-seed input receipt',dp_clip=False,gaussian=False,outer_bernoulli=False,
                    projection='not enabled in original trainer',sampling='auto-parameters; public seed0 fixed',
                    evaluation='IC p1/d1/k50; no raw per-seed CELF/Spread receipt',celf=None,spread=None,coverage=None,
                    aggregate_spread_mean=s['influence_mean'],aggregate_coverage_mean=s['coverage_ratio_mean'],
                    reason='500 vs frozen100 steps; n/feature mismatch where applicable; only aggregate results'))
    p=ROOT/'outputs/utility_sampling/email-eps2-nonoise-20261030-v1'
    m=load(p/'manifest.json');cfg=m['configuration'];structure=load(Path(m['source_experiment'])/'manifest.json')
    for s in table(p/'results.csv'):
        candidates.append(dict(dataset='email',model_seed=int(s['model_seed']),decision='INCOMPATIBLE',source=relative(p/'results.csv'),
            feature=cfg['feature_protocol'],steps=int(s['updates']),n=structure['dataset_parameters']['email']['n'],
            full_train_graph=s['arm']=='original',dp_clip=False,gaussian=False,outer_bernoulli=s['arm']!='original',
            projection=False,sampling='old derived per-seed extraction and batch streams; differs from frozen seed0',
            evaluation='IC p1/d1/k50; CELF432',celf=432,spread=int(s['influence_spread']),coverage=float(s['coverage_ratio_percent']),
            aggregate_spread_mean=None,aggregate_coverage_mean=None,
            reason='old sampling/batch configuration differs' if s['arm']=='original' else 'outer induced graph, not complete prepared train graph'))
    # Additional no-noise diagnostics are clipped/stable-slot variants, not ordinary full-graph SGD.
    for name in ['email-eps2-clip3-zero-20261030-v1','email-stable-slots-partition0-zero-20261031-v1/restricted']:
        p=ROOT/'outputs/utility_sampling'/name/'manifest.json';load(p)
    for d in DATASETS:
        for seed in range(5):
            relevant=[c for c in candidates if c['dataset']==d and c['model_seed']==seed]
            rows.append(dict(dataset=d,method='Non-Private',model_seed=seed,decision='NEED_RERUN',spread=None,coverage=None,
                celf=REFERENCES[d],training_seconds=None,source=';'.join(sorted({c['source'] for c in relevant})),checks='{}',
                reason='no matching per-seed record; incompatible candidates retained' if relevant else 'no matching per-seed history located (including HepPh)'))
    summary=[]
    for d in DATASETS:
        for method in ['Matched PDP','PDP v2']:
            group=[r for r in rows if r['dataset']==d and r['method']==method and r['decision']=='REUSE'];v=[r['coverage'] for r in group]
            summary.append(dict(dataset=d,method=method,n=len(v),coverage_mean=statistics.mean(v) if v else None,
                coverage_sample_sd=statistics.stdev(v) if len(v)>1 else None,
                minimum=min(v) if v else None,low_below50=sum(x<50 for x in v),
                training_seconds_sum=sum(r['training_seconds'] for r in group)))
    def write(name,obj):(output/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    def csvwrite(name,rs):
        with (output/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows([{k:('NA' if v is None else v) for k,v in r.items()} for r in rs])
    csvwrite('reuse_per_seed.csv',rows);csvwrite('nonprivate_candidates.csv',candidates);csvwrite('pdp_summary.csv',summary)
    write('matched_configuration_checks.json',matches);write('evidence_sha256.json',evidence)
    write('audit_summary.json',dict(real_training_attempts=0,model_inference_calls=0,
        decisions={m:{v:sum(r['method']==m and r['decision']==v for r in rows) for v in ['REUSE','NEED_RERUN','INCOMPATIBLE']} for m in ['Non-Private','Matched PDP','PDP v2']},
        all_matched_base_configs_equal=all(x['base_config_identical'] for x in matches),
        caveat='historical recorded execution plus current source/regression evidence, not replay of secret randomness'))
    print(json.dumps(read(output/'audit_summary.json'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
