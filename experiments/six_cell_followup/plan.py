"""Frozen six-cell candidate design and public scalar arithmetic only."""
import hashlib
import json
import math
import statistics
from decimal import Decimal, localcontext
from pathlib import Path
from experiments.dataset_parameter_screen.plan import REFERENCES as OLD_REFERENCES, REGISTRIES as OLD_REGISTRIES

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/pdp_research/six_cell_followup_design_20261112'
DEFAULT_OUTPUT=ROOT/'outputs/research/six-cell-followup-20261112-v1'
DESIGN_SHA256='d5542f78b2d483164608f35eb40665be3e4dbf7d666cc60fcf9df69e06a46b7d'
_design_path=DOC/'CANDIDATES_DRAFT.json'
if hashlib.sha256(_design_path.read_bytes()).hexdigest()!=DESIGN_SHA256:
    raise ValueError('reviewed candidate design changed')
DESIGN=json.loads(_design_path.read_text())
GROUPS={tuple(key.split('/')):configs for key,configs in DESIGN['groups'].items()}
DATASETS=('lastfm','hepph','facebook')
METHODS=('nonprivate','uniform1','pdp_v2','uniform4')
REFERENCES={d:OLD_REFERENCES[d] for d in DATASETS}
REGISTRIES={d:OLD_REGISTRIES[d] for d in DATASETS}
SIGMA100={'uniform1':486.,'pdp_v2':260.,'uniform4':139.}


def matrix():
    return [dict(id=f'{d}-{m}-p{i}-seed{s}',dataset=d,method=m,candidate=f'p{i}',seed=s,**cfg)
            for (d,m),configs in GROUPS.items() for i,cfg in enumerate(configs) for s in (0,1,2)]


def validate_task(task):
    if task not in matrix():raise ValueError('task outside frozen six-cell matrix')
    return task


def noise_plan(method,steps):
    allowed={100,500,1000} if method=='nonprivate' else {100,150,300}
    if method not in METHODS or type(steps) is not int or steps not in allowed:
        raise ValueError('unsupported method or update budget')
    if method=='nonprivate':return dict(sigma_sum=None,temporal_factor=None,single_model_dp=False)
    with localcontext() as ctx:
        ctx.prec=100
        target=Decimal(steps)/100;factor=math.sqrt(steps/100)
        # Check both exact float factor and the rounded product actually reported.
        while (Decimal.from_float(factor)**2<target or
               Decimal.from_float(SIGMA100[method]*factor)**2<Decimal(int(SIGMA100[method]))**2*target):
            factor=math.nextafter(factor,math.inf)
    return dict(sigma_sum=SIGMA100[method]*factor,temporal_factor=factor,sigma100=SIGMA100[method],
                reference_steps=100,conditional_rdp_not_greater_than_original=True,
                machine_dp_certified=False,joint_publication_accounted=False)


def eligible(dataset,method,mean):
    rule=DESIGN['thresholds'][dataset+'/'+method]
    return (('mean_gt' not in rule or mean>rule['mean_gt']) and
            ('mean_ge' not in rule or mean>=rule['mean_ge']) and
            ('mean_le' not in rule or mean<=rule['mean_le']))


def summarize(tasks,rows):
    expected={t['id']:t for t in tasks}
    if len(expected)!=len(tasks):raise ValueError('duplicate planned task')
    seen={}
    for r in rows:
        t=r['task'];key=t['id'];x=r['coverage']
        if key in seen or expected.get(key)!=t:raise ValueError('duplicate/unknown/mutated result')
        if not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('nonfinite Coverage')
        if not isinstance(r['celf'],(int,float)) or r['celf']<=0 or abs(x-100*r['spread']/r['celf'])>1e-9:
            raise ValueError('Coverage arithmetic mismatch')
        seen[key]=r
    complete=set(seen)==set(expected);cells=[]
    for (d,m),configs in GROUPS.items():
        for i,cfg in enumerate(configs):
            name=f'p{i}';group=sorted([r for r in rows if (r['task']['dataset'],r['task']['method'],r['task']['candidate'])==(d,m,name)],key=lambda r:r['task']['seed'])
            if complete and [r['task']['seed'] for r in group]!=[0,1,2]:raise ValueError('complete seeds required')
            x=[r['coverage'] for r in group];by_seed={r['task']['seed']:r['coverage'] for r in group}
            avg=statistics.mean(x) if x else None
            cells.append(dict(dataset=d,method=m,candidate=name,**cfg,n_runs=len(x),mean=avg,
                sample_sd=statistics.stdev(x) if len(x)>1 else None,sample_variance=statistics.variance(x) if len(x)>1 else None,
                minimum=min(x) if x else None,maximum=max(x) if x else None,
                below50=sum(v<50 for v in x) if x else None,below50_fraction=sum(v<50 for v in x)/len(x) if x else None,
                seed0=by_seed.get(0),seed1=by_seed.get(1),seed2=by_seed.get(2)))
    result=dict(status='COMPLETE_DEVELOPMENT_SCREEN' if complete else 'INCOMPLETE',completed=len(rows),planned=len(tasks),
                cells=cells,selection_permitted=complete,independent_confirmation=False,automatic_main_table_edit=False)
    if complete:
        decisions=[]
        for d,m in GROUPS:
            group=[c for c in cells if (c['dataset'],c['method'])==(d,m)]
            for c in group:
                c['mean_target_met']=eligible(d,m,c['mean'])
                c['pareto_mean_sd']=not any(b['mean']>=c['mean'] and b['sample_sd']<=c['sample_sd'] and (b['mean']>c['mean'] or b['sample_sd']<c['sample_sd']) for b in group)
            candidates=[c for c in group if c['mean_target_met']]
            best=min(candidates,key=lambda c:(c['sample_sd'],-c['minimum'],c['T'],c['candidate'])) if candidates else None
            decisions.append(dict(dataset=d,method=m,status='ELIGIBLE_CANDIDATE_FOUND' if best else 'TARGET_NOT_MET',
                                  selected_candidate=best['candidate'] if best else None,eligible_candidates=len(candidates),
                                  pareto_candidates=[c['candidate'] for c in group if c['pareto_mean_sd']]))
        result['decisions']=decisions
    return result
