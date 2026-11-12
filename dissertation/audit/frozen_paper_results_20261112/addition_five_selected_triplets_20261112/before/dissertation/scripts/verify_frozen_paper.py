"""Verify fixed saved scalar cells and render partial Table6.1. No model/graph access."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'dissertation';A=P/'audit/frozen_paper_results_20261112'
f=A/'FROZEN_RESULTS.json';o=json.loads(f.read_text());D=['email','bitcoin','lastfm','hepph','facebook'];M=['nonprivate','uniform1','pdp_v2','uniform4']
L=dict(zip(D,['Email','Bitcoin','LastFM','HepPh','Facebook']));N=dict(zip(M,['Non-Private','Uniform1','PDP v2','Uniform4']))
expected={('email','nonprivate'),('bitcoin','nonprivate'),('lastfm','nonprivate'),('hepph','uniform1'),('bitcoin','pdp_v2'),('lastfm','pdp_v2'),('facebook','pdp_v2'),('email','uniform4'),('bitcoin','uniform4'),('lastfm','uniform4')}
expected.update({('facebook','nonprivate'),('email','uniform1'),('bitcoin','uniform1'),('email','pdp_v2')})
expected.add(('facebook','uniform4'))
assert o['researcher_frozen'] and o['seeds']==[0,1,2] and len(o['cells'])==15
assert {(c['dataset'],c['method']) for c in o['cells']}==expected
cells={};hash_count=0
for c in o['cells']:
 for rel,h in c['source_sha256'].items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
  hash_count+=1
 vals=[]
 assert [r['task']['seed'] for r in c['runs']]==[0,1,2]
 for r in c['runs']:
  e=json.loads((ROOT/r['source']).read_text());cfg=json.loads((ROOT/r['source']).with_name('configuration.json').read_text())
  t=r['task'];assert cfg['task']==t
  assert (t['n'],t['M'],t['T'],t['candidate'])==(c['n'],c['M'],c['T'],c['candidate'])
  assert e['reference_spread']==r['celf'] and e['spread']==r['spread']
  assert abs(e['coverage']-100*e['spread']/e['reference_spread'])<1e-10
  vals.append(e['coverage'])
 assert vals==c['seeds'] and abs(statistics.mean(vals)-c['expected_mean'])<1e-10 and abs(statistics.stdev(vals)-c['expected_sample_sd'])<1e-10
 cells[c['dataset'],c['method']]=c

def table(filename,caption,label,cols,head,body,note=''):
 text='\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+r'\\ \midrule'+'\n'+'\n'.join(body)+'\n\\bottomrule\\end{tabular}\n'
 if note:text+='\\par\\vspace{3pt}\\begin{minipage}{.97\\textwidth}\\footnotesize '+note+'\\end{minipage}\n'
 text+='\\end{table}\n';(P/'tables'/filename).write_text(text)
body=[]
for d in D:
 values=[]
 for m in M:
  c=cells.get((d,m));values.append(f"{c['expected_mean']:.3f} $\\pm$ {c['expected_sample_sd']:.3f}" if c else r'\textemdash')
 body.append(L[d]+' & '+' & '.join(values)+r' \\')
table('coverage.tex',r'Researcher-fixed Coverage mean $\pm$ sample SD (\%): fifteen retained configurations, each with all three initialization seeds 0--2 ($n=3$; 45 fits).', 'tab:coverage','lrrrr','Graph & Non-Private & Uniform1 & PDP v2 & Uniform4',body,
 r'Configurations were selected after a 480-fit development screen and a 144-fit follow-up; these are not independent confirmation results. A dash denotes a cell excluded from the current fixed paper selection, not an unexecuted experiment. Historical results remain archived. Parameters and all retained seed values are listed in Appendix~\ref{app:development}.')
body=[];seeds=[]
for d in D:
 for m in M:
  c=cells.get((d,m))
  if not c:continue
  body.append(f"{L[d]} & {N[m]} & {c['n']} & {c['M']} & {c['T']}"+r' \\')
  seeds.append(f"{L[d]} & {N[m]} & "+' & '.join(f'{v:.6f}' for v in c['seeds'])+r' \\')
table('frozen_parameters.tex',r'Fixed selected configurations for Table~\ref{tab:coverage}. Here $n$ is training-subgraph size, $M$ the SCS threshold, and $T$ the number of updates.', 'tab:frozenparams','llrrr','Graph & Configuration & $n$ & $M$ & $T$',body)
table('frozen_seed_rows.tex',r'All three Coverage values (\%) for each fixed selected configuration. No initialization seed is removed within a retained configuration.', 'tab:frozenseeds','llrrr','Graph & Configuration & Seed 0 & Seed 1 & Seed 2',seeds)
result=dict(status='PASS',fixed_cells=15,displayed_fits=45,seeds=[0,1,2],omitted_cells=5,source_files_checked=hash_count,
 post_hoc_development_selection=True,independent_confirmation=False,new_training=0,new_inference=0,
 frozen_manifest_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
(A/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
