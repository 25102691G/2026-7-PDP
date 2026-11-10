"""Frozen paired-outcome analysis; no graph/model or randomness computation."""
import csv
import json
import math
from pathlib import Path
from statistics import mean,stdev

T95={1:12.706204736,2:4.30265273,3:3.182446305,4:2.776445105}


def interval(values):
    if len(values)<2:return None
    radius=T95[len(values)-1]*stdev(values)/math.sqrt(len(values))
    return [mean(values)-radius,mean(values)+radius]


def summary(last,average):
    if len(last)!=len(average) or not 1<=len(last)<=5:raise ValueError('one to five complete actual pairs required')
    differences=[b-a for a,b in zip(last,average)];n=len(last)
    checks=dict(five_complete_pairs=n==5,positive_mean_delta=mean(differences)>0,
        sample_sd_not_increased=n>1 and stdev(average)<=stdev(last),
        minimum_not_decreased=min(average)>=min(last),
        low_count_not_increased=sum(x<50 for x in average)<=sum(x<50 for x in last),
        no_high_last_drop_ge5=not any(a>=80 and d<=-5 for a,d in zip(last,differences)))
    return dict(n=n,deltas=differences,delta_mean=mean(differences),delta_sample_sd=stdev(differences) if n>1 else None,
        delta_t95_approx=interval(differences),last_mean=mean(last),avg_mean=mean(average),
        last_sample_sd=stdev(last) if n>1 else None,avg_sample_sd=stdev(average) if n>1 else None,
        last_t95_approx=interval(last),avg_t95_approx=interval(average),last_min=min(last),avg_min=min(average),
        low_last_count=sum(x<50 for x in last),low_avg_count=sum(x<50 for x in average),
        maximum_decline_pp=max(0.,-min(differences)),maximum_improvement_pp=max(0.,max(differences)),
        delta_min=min(differences),delta_max=max(differences),
        high_last_drop_ge5_count=sum(a>=80 and d<=-5 for a,d in zip(last,differences)),
        checks=checks,screen_pass=all(checks.values()))


def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    Path(path).chmod(0o600)


def analyze(restricted):
    r=Path(restricted);tasks=json.loads((r/'tasks.json').read_text());rows=[];pairs=[];strata=[];topsets={};outcomes={}
    for t in tasks:
        identity={k:t[k] for k in ('id','dataset','model_seed','status')}
        if t['status']!='COMPLETED':
            pairs.append(dict(**identity,error_type=t.get('error_type')));continue
        record=json.loads(Path(t['result']).read_text());metrics=record['metrics']
        for label,m in metrics.items():
            row=dict(**identity,output=label)
            row.update({k:m[k] for k in ('spread','coverage','reference_spread','A','R','mean_selected_outdegree',
                'N_plus','bias_tie_nodes','max_tie_count','boundary_tie_count','selected_zero_outdegree',
                'last_zero_aggregate_nodes','last_zero_weights','spearman_all','spearman_nonzero_degree')})
            row.update(training_seconds=record['training_seconds'],evaluation_seconds=record['evaluation_seconds'][label],n=1,std=None)
            rows.append(row);topsets.setdefault(t['id'],{})[label]=dict(local_ids=m['topk_local_ids'],stable_ids=m['topk_stable_ids'])
            for band,s in m['spearman_degree_strata'].items():strata.append(dict(**identity,output=label,degree_band=band,**s))
        a,b=metrics['v2_last'],metrics['PDP-v2-Avg']
        reach=(b['A']-a['A'])*(1-(a['R']+b['R'])/2);redundancy=-(b['R']-a['R'])*(a['A']+b['A'])/2
        assert abs(reach+redundancy-(b['spread']-a['spread']))<1e-8
        pair=dict(**identity,last_spread=a['spread'],avg_spread=b['spread'],last_coverage=a['coverage'],avg_coverage=b['coverage'],
            delta_pp=b['coverage']-a['coverage'],spread_delta=b['spread']-a['spread'],
            top50_overlap=len(set(a['topk_local_ids'])&set(b['topk_local_ids']))/50,
            last_A=a['A'],avg_A=b['A'],last_R=a['R'],avg_R=b['R'],
            reach_term=reach,redundancy_term=redundancy,
            last_degree=a['mean_selected_outdegree'],avg_degree=b['mean_selected_outdegree'],
            training_seconds=record['training_seconds'],high_last_drop_ge5=a['coverage']>=80 and b['coverage']-a['coverage']<=-5)
        pairs.append(pair);outcomes.setdefault(t['dataset'],[]).append(pair)
    summaries={}
    for dataset,complete in outcomes.items():
        complete.sort(key=lambda x:x['model_seed'])
        summaries[dataset]=summary([x['last_coverage'] for x in complete],[x['avg_coverage'] for x in complete])
        summaries[dataset]['model_seeds']=[x['model_seed'] for x in complete]
    table(r/'per_output.csv',rows);table(r/'paired_differences.csv',pairs);table(r/'degree_strata.csv',strata)
    table(r/'dataset_summary.csv',[dict(dataset=d,**{k:v for k,v in s.items() if not isinstance(v,(list,dict))}) for d,s in summaries.items()])
    from experiments.run_pdp_g2 import write
    write(r/'analysis.json',dict(datasets=summaries,all_five_screen_pass=len(summaries)==5 and all(s['screen_pass'] for s in summaries.values()),
        ci_interpretation='descriptive small-sample Student-t approximation only; fixed development seeds, not independent external confirmation',
        low_definition='Coverage<50%',high_last_deterioration='last>=80%, paired delta<=-5pp',independent_confirmation=False))
    write(r/'top50_sets.json',topsets)
    return summaries


if __name__=='__main__':
    import sys
    analyze(sys.argv[1])
