"""Offline G2 summaries; reads recorded outcomes only, no numerical runtime."""
import csv
import json
from pathlib import Path
from statistics import mean,stdev
from experiments.pdp_g2_controlled import variance_components


def table(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    Path(path).chmod(0o600)


def analyze(restricted):
    restricted=Path(restricted);tasks=json.loads((restricted/'tasks.json').read_text());records={}
    rows=[]
    for task in tasks:
        row={k:task[k] for k in ('id','dataset','model_seed','outer_index','gaussian_replicate','status')}
        if task['status']=='COMPLETED':
            record=json.loads(Path(task['result']).read_text());records[task['id']]=record;m=record['metrics']
            row.update({k:m[k] for k in ('spread','coverage','reference_spread','A','R','mean_selected_outdegree',
                'spearman_all','spearman_nonzero_degree','N_plus','bias_tie_nodes','last_zero_aggregate_nodes',
                'boundary_tie_count','max_tie_count','selected_zero_outdegree','last_zero_weights')})
            row.update({k:record[k] for k in ('training_seconds','evaluation_seconds','updates','gaussian_calls',
                'gaussian_coordinates','zero_gradient_slots','projection_trigger_steps','projection_displacement_sum')})
            row['low_coverage_below50']=m['coverage']<50
        else:row['error_type']=task.get('error_type')
        rows.append(row)
    table(restricted/'per_run.csv',rows)
    pairs=[];components=[];initializations=[];datasets=[]
    for dataset in ('bitcoin','lastfm','facebook'):
        init_means={}
        for seed in (1,2):
            groups=[]
            for h in (0,1):
                names=[f'{dataset}_i{seed}_h{h}_z{z}' for z in (0,1)]
                if not all(n in records for n in names):continue
                a,b=[records[n]['metrics'] for n in names]
                pair=dict(dataset=dataset,model_seed=seed,outer_index=h,coverage_z0=a['coverage'],coverage_z1=b['coverage'],
                    coverage_difference=b['coverage']-a['coverage'],coverage_abs_difference=abs(b['coverage']-a['coverage']),
                    coverage_mean=(a['coverage']+b['coverage'])/2,
                    top50_overlap=len(set(a['topk_local_ids'])&set(b['topk_local_ids']))/50,
                    crosses_low50=(a['coverage']<50)!=(b['coverage']<50))
                for k in ('spread','A','R','mean_selected_outdegree','spearman_all','spearman_nonzero_degree'):
                    pair[k+'_z0']=a[k];pair[k+'_z1']=b[k]
                    pair[k+'_difference']=None if a[k] is None or b[k] is None else b[k]-a[k]
                pairs.append(pair);groups.append([a['coverage'],b['coverage']])
            if len(groups)==2:
                comp=dict(dataset=dataset,model_seed=seed,**variance_components(groups));components.append(comp)
                values=[v for g in groups for v in g];init_means[seed]=mean(values)
                initializations.append(dict(dataset=dataset,model_seed=seed,model_n=4,outer_graph_n=2,
                    coverage_mean=mean(values),coverage_sample_sd=stdev(values),
                    outer_mean_h0=mean(groups[0]),outer_mean_h1=mean(groups[1]),
                    outer_mean_difference=mean(groups[1])-mean(groups[0])))
        ds=[r for r in rows if r['dataset']==dataset and r['status']=='COMPLETED']
        if ds:
            values=[r['coverage'] for r in ds]
            datasets.append(dict(dataset=dataset,model_n=len(ds),outer_graph_n=len({(r['model_seed'],r['outer_index']) for r in ds}),
                coverage_mean=mean(values),coverage_sample_sd=stdev(values) if len(values)>1 else None,
                coverage_min=min(values),coverage_max=max(values),low_below50_count=sum(v<50 for v in values),
                initialization_i2_minus_i1=None if len(init_means)!=2 else init_means[2]-init_means[1],
                coverage_vs_A_spearman=_rankcorr([r['coverage'] for r in ds],[r['A'] for r in ds]),
                coverage_vs_degree_spearman=_rankcorr([r['coverage'] for r in ds],[r['mean_selected_outdegree'] for r in ds])))
    table(restricted/'gaussian_pairs.csv',pairs);table(restricted/'outer_initialization.csv',initializations)
    table(restricted/'dataset_summary.csv',datasets)
    from experiments.run_pdp_g2 import write
    write(restricted/'analysis.json',dict(datasets=datasets,gaussian_pairs=pairs,nested_variance=components,
        initialization_summaries=initializations,independent_confirmation=False,
        initialization_contrasts_confounded_with_nested_graph_draws=True,
        low_cutoff='historical descriptive Coverage<50%, not a new acceptance gate'))
    return rows,pairs,components,datasets


def _rankcorr(a,b):
    from experiments.pdp_g0g1_diagnostic import spearman
    return spearman(a,b)


if __name__=='__main__':
    import sys
    analyze(sys.argv[1])
