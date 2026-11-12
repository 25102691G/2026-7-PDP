"""Public finite design and scalar analysis only; imports no graph/model code."""
import math
import statistics
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('email', 'bitcoin', 'lastfm', 'hepph', 'facebook')
METHODS = ('nonprivate', 'uniform1', 'pdp_v2', 'uniform4')
SEEDS = (0, 1, 2)
REFERENCES = dict(email=432, bitcoin=1684, lastfm=1269, hepph=1741, facebook=3157)
SIGMA100 = dict(uniform1=486., pdp_v2=260., uniform4=139.)
DOC = ROOT / 'docs/pdp_research/dataset_parameter_screen_20261111'
DEFAULT_OUTPUT = ROOT / 'outputs/research/dataset-parameter-screen-20261111-v1'
REGISTRIES = {d: f'outputs/research/five-dataset-three-methods-20261102-v1/restricted/registries/{d}.json' for d in DATASETS}


def candidates(dataset):
    if dataset not in DATASETS: raise ValueError('only the five retained datasets')
    a, b = (20, 60) if dataset in ('email', 'bitcoin') else (60, 80)
    lo, hi = (6, 8) if dataset == 'email' else (2, 6)
    joint_n = 40 if dataset == 'email' else 60
    return [dict(candidate=name, n=n, M=m, T=t) for name,n,m,t in (
        ('baseline',40,4,100), ('size_a',a,4,100), ('size_b',b,4,100),
        ('frequency_a',40,lo,100), ('frequency_b',40,hi,100),
        ('steps300',40,4,300), ('steps500',40,4,500), ('joint',joint_n,lo,300))]


def matrix():
    return [dict(id=f'{d}-{m}-{c["candidate"]}-seed{s}',dataset=d,method=m,seed=s,**c)
            for d in DATASETS for c in candidates(d) for m in METHODS for s in SEEDS]


def validate_task(task):
    if task not in matrix(): raise ValueError('task outside frozen candidate/seed matrix')
    return task


def noise_plan(method, steps):
    if method not in METHODS or type(steps) is not int or steps not in (100,300,500):
        raise ValueError('unsupported method or update budget')
    if method == 'nonprivate': return dict(sigma_sum=None, temporal_factor=None, single_model_dp=False)
    factor = math.sqrt(steps / 100)
    with localcontext() as ctx:
        ctx.prec = 80
        while Decimal.from_float(factor)**2 < Decimal(steps)/100:
            factor = math.nextafter(factor, math.inf)
    return dict(sigma_sum=SIGMA100[method]*factor, temporal_factor=factor,
                sigma100=SIGMA100[method], reference_steps=100,
                conditional_rdp_not_greater_than_original=True,
                machine_dp_certified=False, joint_publication_accounted=False)


def summarize(tasks, rows):
    expected = {t['id']:t for t in tasks}
    if len(expected) != len(tasks): raise ValueError('duplicate planned identity')
    seen = {}
    for row in rows:
        t=row['task']; key=t['id']
        if key in seen or expected.get(key)!=t: raise ValueError('duplicate/unknown/mutated result')
        x=row['coverage']
        if not isinstance(x,(int,float)) or not math.isfinite(x): raise ValueError('invalid coverage')
        if abs(x-100*row['spread']/row['celf'])>1e-9: raise ValueError('coverage arithmetic mismatch')
        seen[key]=row
    complete=set(seen)==set(expected)
    cells=[]
    for d in DATASETS:
        for m in METHODS:
            for cfg in candidates(d):
                group=sorted([r for r in rows if (r['task']['dataset'],r['task']['method'],r['task']['candidate'])==(d,m,cfg['candidate'])],key=lambda r:r['task']['seed'])
                if complete and [r['task']['seed'] for r in group]!=[0,1,2]: raise ValueError('complete seed triplet required')
                x=[r['coverage'] for r in group]
                by_seed={r['task']['seed']:r['coverage'] for r in group}
                cells.append(dict(dataset=d,method=m,**cfg,n_runs=len(x),
                    mean=statistics.mean(x) if x else None,sample_sd=statistics.stdev(x) if len(x)>1 else None,
                    minimum=min(x) if x else None,maximum=max(x) if x else None,
                    below50=sum(v<50 for v in x) if x else None,
                    below50_fraction=sum(v<50 for v in x)/len(x) if x else None,
                    seed0=by_seed.get(0),seed1=by_seed.get(1),seed2=by_seed.get(2)))
    if not complete:return dict(status='INCOMPLETE',completed=len(rows),planned=len(tasks),cells=cells,selection_permitted=False)
    for c in cells:
        group=[x for x in cells if (x['dataset'],x['method'])==(c['dataset'],c['method'])]
        b=next(x for x in group if x['candidate']=='baseline')
        c['mean_difference_from_baseline']=c['mean']-b['mean']
        c['sd_difference_from_baseline']=c['sample_sd']-b['sample_sd']
        c['passes_baseline_guards']=(c['candidate']!='baseline' and c['mean']>=b['mean'] and c['sample_sd']<=b['sample_sd']
            and c['minimum']>=b['minimum'] and c['below50']<=b['below50']
            and (c['mean']>b['mean'] or c['sample_sd']<b['sample_sd']))
        c['pareto_mean_sd']=not any(x['mean']>=c['mean'] and x['sample_sd']<=c['sample_sd']
            and (x['mean']>c['mean'] or x['sample_sd']<c['sample_sd']) for x in group)
    return dict(status='COMPLETE_DEVELOPMENT_SCREEN',completed=len(rows),planned=len(tasks),cells=cells,
                selection_permitted=True,independent_confirmation=False,
                interpretation='descriptive three-seed development; all candidates retained; no automatic manuscript replacement')
