"""Deterministic W1 arithmetic checks. No model/graph imports or resampling."""
from pathlib import Path
import json,csv,hashlib,math,statistics
from collections import Counter
import scipy.stats as st
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'dissertation'
SRC=ROOT/'docs/pdp_research/stage_u6a_posttraining_analysis/runtime/outputs'
manifest=json.loads((SRC/'OUTPUT_MANIFEST.json').read_text())
for name,h in manifest['files'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==h,name
rows=json.loads((SRC/'observations.json').read_text()); a=json.loads((SRC/'analysis.json').read_text())
assert len(rows)==2900 and len({r['run_id'] for r in rows})==2900
D=['email','bitcoin','lastfm','hepph','facebook']; M=['nonprivate','uniform1','pdp_v2','uniform4']
N=dict(zip(M,[40,200,200,140])); CELF=dict(zip(D,[432,1684,1269,1741,3157]))
g={}; cells={}; maxerr=0
for d in D:
 cells[d]={}
 for m in M:
  rr=[r for r in rows if r['dataset']==d and r['method']==m]; assert len(rr)==N[m]
  for r in rr:
   assert r['celf']==CELF[d]
   assert abs(r['coverage']-100*r['spread']/r['celf'])<1e-10
  x=[r['coverage'] for r in rr];g[d,m]=x;k=sum(y<50 for y in x);n=len(x)
  c=dict(n=n,mean=statistics.mean(x),sample_sd=statistics.stdev(x),minimum=min(x),maximum=max(x),low_tail_lt50_count=k,low_tail_frequency=k/n,low_tail_cp95=[0 if k==0 else st.beta.ppf(.025,k,n-k+1),1 if k==n else st.beta.ppf(.975,k+1,n-k)])
  for key in ['mean','sample_sd','minimum','maximum','low_tail_lt50_count']:
   err=abs(c[key]-a['cells'][d][m][key]); maxerr=max(maxerr,err);assert err<1e-10,(d,m,key)
  assert max(abs(u-v) for u,v in zip(c['low_tail_cp95'],a['cells'][d][m]['low_tail_cp95']))<1e-10
  cells[d][m]=c

def welch(ds,ref):
 est=statistics.mean(statistics.mean(g[d,'pdp_v2'])-statistics.mean(g[d,ref]) for d in ds)
 terms=[statistics.variance(g[d,m])/len(g[d,m])/len(ds)**2 for d in ds for m in ['pdp_v2',ref]]
 v=sum(terms);df=v*v/sum(t*t/(len(g[d,m])-1) for t,(d,m) in zip(terms,[(d,m) for d in ds for m in ['pdp_v2',ref]]));se=math.sqrt(v)
 return dict(estimate_pp=est,standard_error_pp=se,degrees_of_freedom=df,p_one_sided=float(st.t.sf(est/se,df)),lower_one_sided_95_pp=est-float(st.t.ppf(.95,df))*se,ci_two_sided_95_pp=[est-float(st.t.ppf(.975,df))*se,est+float(st.t.ppf(.975,df))*se])
primary=welch(D,'uniform1')
for k in ['estimate_pp','p_one_sided','lower_one_sided_95_pp']:
 assert abs(primary[k]-a['primary'][k])<1e-10
assert max(abs(u-v) for u,v in zip(primary['ci_two_sided_95_pp'],a['primary']['ci_two_sided_95_pp']))<1e-10
for ref in ['uniform4','nonprivate']:
 w=welch(D,ref);old=a['secondary_comparisons'][ref]['equal_five_dataset']
 assert abs(w['estimate_pp']-old['estimate_pp'])<1e-10
 assert max(abs(u-v) for u,v in zip(w['ci_two_sided_95_pp'],old['ci_two_sided_95_pp']))<1e-10
for d in D:
 w=welch([d],'uniform1');assert abs(w['estimate_pp']-a['per_dataset_primary'][d]['estimate_pp'])<1e-10
low={m:sum(cells[d][m]['low_tail_lt50_count'] for d in D) for m in M}; assert low==dict(zip(M,[0,115,103,58]))
report=dict(status='PASS',rows=2900,unique_run_ids=2900,all20_cells_match=True,manifest_hashes_match=True,maximum_summary_absolute_error=maxerr,primary=primary,low_tail_totals=low,scope='deterministic saved-scalar verification only; no bootstrap rerun, no RNG, no graphs/models')
(OUT/'audit/SCALAR_VERIFICATION.json').write_text(json.dumps(report,indent=2))
with (OUT/'audit/final_scalar_rows.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'audit/final_cell_statistics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['dataset','method','n','mean','sample_sd','min','max','below50','cp95_low','cp95_high'])
 for d in D:
  for m in M:
   c=cells[d][m];w.writerow([d,m,c['n'],c['mean'],c['sample_sd'],c['minimum'],c['maximum'],c['low_tail_lt50_count'],*c['low_tail_cp95']])
labels=dict(zip(D,['Email','Bitcoin','LastFM','HepPh','Facebook']));ml=dict(zip(M,['Non-Private',r'Uniform $\\epsilon=1$', 'PDP v2',r'Uniform $\\epsilon=4$']))
def table(name,caption,label,cols,head,body):
 text=r'\begin{table}[htbp]'+'\n'+r'\centering\small'+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\begin{tabular}{'+cols+'}\n'+r'\toprule'+'\n'+head+r'\\ \midrule'+'\n'+'\n'.join(body)+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\end{table}'+'\n'
 (OUT/'tables'/name).write_text(text.replace('\\\\epsilon','\\epsilon'))
body=[]
for d in D:
 body.append(labels[d]+' & '+' & '.join(f"{cells[d][m]['mean']:.3f} $\\pm$ {cells[d][m]['sample_sd']:.3f}" for m in M)+r' \\')
table('coverage.tex','Independent confirmation: Coverage mean $\\pm$ sample SD (\\%). Per graph: Non-Private $n=40$, Uniform1 $n=200$, PDP v2 $n=200$, Uniform4 $n=140$. Source: sealed U6-A saved scalar matrix.','tab:coverage','lrrrr',r'Graph & Non-Private & Uniform1 & PDP v2 & Uniform4',body)
body=[]
for d in D:
 for m in M:
  c=cells[d][m]; lo,hi=c['low_tail_cp95']
  body.append(f"{labels[d]} & {ml[m]} & {c['minimum']:.3f}--{c['maximum']:.3f} & {c['low_tail_lt50_count']}/{c['n']} & {100*c['low_tail_frequency']:.2f} & [{100*lo:.2f}, {100*hi:.2f}]"+r' \\')
table('low_tail.tex','Full-range and severe-low-utility results. Coverage and cellwise 95\\% Clopper--Pearson intervals are percentages. Intervals concern each cell under the IID repeat model, without simultaneous adjustment. Source: U6-A.','tab:low','llrrrr',r'Graph & Method & Min--max & $<50\%$ & Rate (\%) & CP interval (\%)',body)
body=[]
for d in D:
 c=a['per_dataset_primary'][d];lo,hi=c['ci_two_sided_95_pp'];b=c['ci_two_sided_99_bonferroni_pp']
 body.append(f"{labels[d]} & {c['estimate_pp']:+.3f} & [{lo:.3f}, {hi:.3f}] & {c['p_one_sided']:.4f} & [{b[0]:.3f}, {b[1]:.3f}]"+r' \\')
table('effects.tex','PDP v2 minus Uniform1 (pp). All five primary-gated Holm claims remain closed. 99\\% individual intervals form the Bonferroni family; all Welch intervals are approximate. Source: U6-A.','tab:effects','lrrrr',r'Graph & Effect & 95\% CI & One-sided $p$ & 99\% CI',body)
body=[]
for d in D:
 parts=[]
 for ref in ['uniform4','nonprivate']:
  c=a['secondary_comparisons'][ref]['per_dataset'][d];lo,hi=c['ci_two_sided_95_pp'];parts += [f"{c['estimate_pp']:+.3f}",f'[{lo:.3f}, {hi:.3f}]']
 body.append(labels[d]+' & '+' & '.join(parts)+r' \\')
table('reference_gaps.tex','PDP v2 minus the relaxed Uniform4 and Non-Private references (pp), with individual approximate 95\\% intervals. These are secondary estimations, not alternative primary success tests. Source: U6-A.','tab:references','lrrrr',r'Graph & vs Uniform4 & 95\% CI & vs Non-Private & 95\% CI',body)
print(json.dumps(report,indent=2))
