"""Verify saved selected scalars and full source cohorts; render paper tables."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'dissertation';A=P/'audit/frozen_paper_results_20261112'
f=A/'FROZEN_RESULTS.json';o=json.loads(f.read_text());D=['email','bitcoin','lastfm','hepph','facebook'];M=['nonprivate','uniform1','pdp_v2','uniform4']
L=dict(zip(D,['Email','Bitcoin','LastFM','HepPh','Facebook']));N=dict(zip(M,['Non-Private','Uniform1','PDP-PrivIM','Uniform4']))
expected={(d,m) for d in D for m in M}
reference_sigma={'uniform1':486, 'pdp_v2':260, 'uniform4':139}
assert o['researcher_frozen'] and len(o['cells'])==20 and o['selected_runs']==60
assert {(c['dataset'],c['method']) for c in o['cells']}==expected
cells={};sigmas={};hash_count=0;posthoc=0
for c in o['cells']:
 for rel,h in c['source_sha256'].items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
  hash_count+=1
 subset=c.get('selection_kind')=='post_hoc_three_of_larger_cohort'
 public=[r['task']['seed'] for r in c['runs']]
 assert len(public)==3 and len(set(public))==3
 assert public==(c['public_seeds'] if subset else [0,1,2])
 vals=[];cell_sigmas=set()
 for r in c['runs']:
  p=ROOT/r['source'];e=json.loads(p.read_text());cfg=json.loads(p.with_name('configuration.json').read_text());t=r['task']
  assert (t['n'],t['M'],t['T'],t['candidate'])==(c['n'],c['M'],c['T'],c['candidate'])
  if subset:
   assert cfg['public_seed']==t['seed'] and cfg['dataset']==c['dataset'] and cfg['method']==c['method']
   assert cfg['task_id']==t['id'] and cfg['original_task_id']==t['original_task_id']
   assert cfg['training']['T']==c['T']
   checkpoint=json.loads(p.with_name('model.json').read_text())['configuration']
   if c['method']=='nonprivate':
    b=checkpoint['original_metadata'];assert not b['with_dp'] and not b['non_private_clipping']
    assert (b['subgraph_size'],b['sampling']['frequency_threshold'],b['actual_optimizer_steps'])==(40,4,100)
   else:
    b=checkpoint['base'];assert (b['n'],b['M_SCS'],b['T'])==(40,4,100)
    assert b['s_sum']==reference_sigma[c['method']]
  else:assert cfg['task']==t
  if c['method']!='nonprivate':
   ref=reference_sigma[c['method']]
   if subset:
    sigma=cfg['training']['sigma_sum']
    assert c['T']==100 and sigma==ref
   else:
    noise=cfg['noise']
    factor=noise['temporal_factor']
    sigma=noise['sigma_sum']
    assert noise['reference_steps']==100 and noise['sigma100']==ref
    assert 100*Fraction(factor)**2>=c['T']
    assert sigma==ref*factor
   cell_sigmas.add(sigma)
  assert e['reference_spread']==r['celf'] and e['spread']==r['spread']
  assert abs(e['coverage']-100*e['spread']/e['reference_spread'])<1e-10
  vals.append(e['coverage'])
 assert vals==c['seeds'] and abs(statistics.mean(vals)-c['expected_mean'])<1e-10 and abs(statistics.stdev(vals)-c['expected_sample_sd'])<1e-10
 if c['method']!='nonprivate':
  assert len(cell_sigmas)==1
  sigmas[c['dataset'],c['method']]=cell_sigmas.pop()
 posthoc+=int(subset);cells[c['dataset'],c['method']]=c
assert posthoc==5
full_path=ROOT/o['full_source_cohorts_file']
assert hashlib.sha256(full_path.read_bytes()).hexdigest()==o['full_source_cohorts_sha256']
full=json.loads(full_path.read_text());assert len(full)==5
full_run_count=0
for cohort in full:
 values=[];identities=set();seeds=set()
 for r in cohort['runs']:
  p=ROOT/r['source'];e=json.loads(p.read_text());cfg=json.loads(p.with_name('configuration.json').read_text());done=json.loads(p.with_name('completion.json').read_text())
  assert done['status']=='COMPLETED' and done['real_training_attempts']==1
  assert hashlib.sha256(p.read_bytes()).hexdigest()==done['final_evaluation_sha256']
  assert cfg['public_seed']==r['seed'] and cfg['dataset']==cohort['dataset'] and cfg['method']==cohort['method']
  assert cfg['original_task_id']==r['unit'] and r['unit'] not in identities and r['seed'] not in seeds
  identities.add(r['unit']);seeds.add(r['seed'])
  assert e['coverage']==r['coverage'] and e['spread']==r['spread']
  assert abs(e['coverage']-100*e['spread']/e['reference_spread'])<1e-10
  values.append(e['coverage'])
 assert len(values)==cohort['n']
 assert abs(statistics.mean(values)-cohort['mean'])<1e-10 and abs(statistics.stdev(values)-cohort['sample_sd'])<1e-10
 assert min(values)==cohort['minimum'] and max(values)==cohort['maximum'] and sum(v<50 for v in values)==cohort['below50']
 selected=cells[cohort['dataset'],cohort['method']]
 assert selected['source_cohort_n']==cohort['n']
 assert abs(selected['source_cohort_mean']-cohort['mean'])<1e-10 and abs(selected['source_cohort_sample_sd']-cohort['sample_sd'])<1e-10
 assert {r['source'] for r in selected['runs']}<={r['source'] for r in cohort['runs']}
 full_run_count+=len(values)
assert full_run_count==780

def table(filename,caption,label,cols,head,body,note='',size='small'):
 text='\\begin{table}[htbp]\n\\centering\\'+size+'\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+r'\\ \midrule'+'\n'+'\n'.join(body)+'\n\\bottomrule\\end{tabular}\n'
 if note:text+='\\par\\vspace{3pt}\\begin{minipage}{.97\\textwidth}\\footnotesize '+note+'\\end{minipage}\n'
 text+='\\end{table}\n';(P/'tables'/filename).write_text(text)
body=[]
for d in D:
 values=[]
 for m in M:
  c=cells[d,m];v=f"{c['expected_mean']:.3f} $\\pm$ {c['expected_sample_sd']:.3f}"
  values.append(v)
 body.append(L[d]+' & '+' & '.join(values)+r' \\')
table('coverage.tex',r'Coverage mean $\pm$ sample SD (\%) across five datasets and four configurations: three runs per cell ($n=3$; 60 runs in total).', 'tab:coverage','lrrrr','Graph & Non-Private & Uniform1 & PDP-PrivIM & Uniform4',body,
 r'Each cell reports mean Coverage and sample SD for three runs. Five cells report subsets of larger development cohorts; see Appendix~\ref{sec:sourceprotocol} for the selection procedure and full-cohort summaries.')

body=[]
for d in D:
 for m in M:
  cell=cells[d,m]
  values=cell['seeds']
  runs=[
   rf"\shortstack[r]{{{value:.6f}\\({run['task']['seed']})}}"
   for value,run in zip(values,cell['runs'])
  ]
  low=sum(value<50 for value in values)
  sigma=sigmas.get((d,m))
  scale=f'{sigma:.3f}' if sigma is not None else r'\textemdash'
  params=rf"\shortstack[l]{{({cell['n']}, {cell['M']}, {cell['T']})\\{scale}}}"
  fields=[L[d],N[m],params,*runs,f"{min(values):.3f}",f"{max(values):.3f}",f"{low}/3"]
  body.append(' & '.join(fields)+r' \\[2pt]')
table('frozen_details.tex',r'Configuration and individual outcomes for every cell in Table~\ref{tab:coverage}.', 'tab:frozendetails','lllrrrrrr',
 r'Graph & Configuration & \shortstack[l]{$(n,M,T)$\\$\sigma_T$} & Run A & Run B & Run C & Min & Max & $C<50\%$',body,
 r'The parameter column gives subgraph size, SCS threshold, and updates above the recorded nominal sum-noise scale, rounded to three decimals. A dash denotes ordinary non-private training. Run columns pair Coverage (\%) with public initialization IDs in parentheses. Minima, maxima, and low-utility counts summarize the three displayed runs.',size='scriptsize')
body=[]
for cohort in sorted(full,key=lambda c:(D.index(c['dataset']),M.index(c['method']))):
 body.append(f"{L[cohort['dataset']]} & {N[cohort['method']]} & {cohort['n']} & {cohort['mean']:.3f} $\\pm$ {cohort['sample_sd']:.3f} & {cohort['minimum']:.3f} & {cohort['maximum']:.3f} & {cohort['below50']}"+r' \\')
table('subset_source_cohorts.tex',r'Complete historical source cohorts for five cells in Table~\ref{tab:coverage}. These summaries describe original cohort variability; the main comparison reports three runs per cell. See Section~\ref{sec:sourceprotocol} for the source protocol.', 'tab:subsetcohorts','llrrrrr',r'Graph & Configuration & $N$ & Mean $\pm$ SD (\%) & Min & Max & $<50\%$',body,
 r'All runs use $n=40$, $M=4$, $T=100$. $N$ is the number of original runs, not subgraph size. Complete raw outcomes and original task IDs remain in the frozen provenance record. No low-utility run is removed from these summaries.',size='scriptsize')
result=dict(status='PASS',fixed_cells=20,displayed_fits=60,complete_seed012_cells=15,posthoc_triplet_cells=5,omitted_cells=0,source_files_checked=hash_count,full_source_cohort_runs_verified=full_run_count,
 post_hoc_development_selection=True,independent_confirmation=False,new_training=0,new_inference=0,
 frozen_manifest_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
(A/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
