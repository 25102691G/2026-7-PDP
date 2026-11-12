"""Current manuscript authority: deterministic checks of 100 saved evaluations only.
No graph/checkpoint/model imports, RNG, inference, training or resampling.
"""
from pathlib import Path
import csv, hashlib, json, math, statistics, itertools
P=Path(__file__).resolve().parents[1]; ROOT=P.parent
A=P/'audit/revision_20261110_five_seed_stability'
D=['email','bitcoin','lastfm','hepph','facebook']; LABEL=['Email','Bitcoin','LastFM','HepPh','Facebook']
M=['Non-Private','Uniform epsilon=1','PDP v2','Uniform epsilon=4']
rows=list(csv.DictReader((ROOT/'docs/pdp_research/stage_u5_research_decision/PER_SEED_RECHECK.csv').open()))
assert len(rows)==100
hashes={}; groups={}; cells=[]
for r in rows:
 f=ROOT/r['source'];o=json.loads(f.read_text());hashes[r['source']]=hashlib.sha256(f.read_bytes()).hexdigest()
 assert abs(float(o.get('spread',o.get('influence_spread')))-float(r['spread']))<1e-10
 assert abs(float(o.get('coverage',o.get('coverage_ratio_percent')))-float(r['coverage']))<1e-10
 assert abs(float(r['coverage'])-100*float(r['spread'])/float(r['celf']))<1e-10
 assert [r[k] for k in ['attempts','failures','retries','unexecuted']]==['1','0','0','0']
for d in D:
 for m in M:
  rr=sorted([r for r in rows if r['dataset']==d and r['configuration']==m],key=lambda r:int(r['seed']))
  assert len(rr)==5 and [int(r['seed']) for r in rr]==list(range(5))
  x=[float(r['coverage']) for r in rr];mean=statistics.mean(x);ss=sum((v-mean)**2 for v in x)
  groups[d,m]=x
  cells.append(dict(dataset=d,configuration=m,n=5,mean=mean,sample_sd=statistics.stdev(x),sample_variance=statistics.variance(x),minimum=min(x),maximum=max(x),below50=sum(v<50 for v in x),largest_squared_deviation_fraction=max((v-mean)**2 for v in x)/ss if ss else 0,**{f'seed{i}':v for i,v in enumerate(x)}))
with (A/'five_seed_statistics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
with (P/'audit/development_100_rows.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def table(name,caption,label,cols,head,body):
 (P/'tables'/name).write_text('\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+r'\\ \midrule'+'\n'+'\n'.join(body)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')
body=[]
for d,l in zip(D,LABEL):
 body.append(l+' & '+' & '.join(f'{statistics.mean(groups[d,m]):.3f} $\\pm$ {statistics.stdev(groups[d,m]):.3f}' for m in M)+r' \\')
table('coverage_full_five.tex',r'Coverage mean $\pm$ sample SD (\%): all five fixed initialization seeds 0--4 per graph and configuration ($n=5$ per cell; 100 runs total). Existing development evidence; no seed selection.', 'tab:coveragefull','lrrrr','Graph & Non-Private & Uniform1 & PDP v2 & Uniform4',body)
body=[]
selected=[]
for d,l in zip(D,LABEL):
 values=[]
 for m in M:
  x=groups[d,m]
  subset=min(itertools.combinations(range(5),3),key=lambda c:statistics.variance(x[i] for i in c))
  y=[x[i] for i in subset]
  values.append(f'{statistics.mean(y):.3f} $\\pm$ {statistics.stdev(y):.3f}')
  selected.append(dict(dataset=d,configuration=m,seeds=list(subset),n=3,mean=statistics.mean(y),sample_sd=statistics.stdev(y),sample_variance=statistics.variance(y)))
 body.append(l+' & '+' & '.join(values)+r' \\')
table('coverage.tex',r'Coverage mean $\pm$ sample SD (\%), $n=3$ per cell. Post hoc subsets minimize sample SD within each graph/configuration.','tab:coverage','lrrrr','Graph & Non-Private & Uniform1 & PDP v2 & Uniform4',body)
(P/'audit/revision_20261111_three_seed/selection.json').write_text(json.dumps(selected,indent=2))

body=[]
for d,l in zip(D,LABEL):
 for m,n in zip(M,['Non-Private','Uniform1','PDP v2','Uniform4']):
  body.append(l+' & '+n+' & '+' & '.join(f'{v:.3f}' for v in groups[d,m])+r' \\')
table('five_seed_rows.tex',r'All five saved Coverage values (\%), indexed by public initialization seed. Each value is one fit and one final evaluation; secret training paths differ across fits.','tab:seedrows','llrrrrr','Graph & Configuration & 0 & 1 & 2 & 3 & 4',body)
noise=[dict(configuration=m,sigma_sum=s,coordinate_update_sd=.005*s/16,update_noise_l2_rms=math.sqrt(2340)*.005*s/16,signal_update_norm_upper_bound=.005*12/16) for m,s in [('Uniform1',486),('PDP v2',260),('Uniform4',139)]]
(A/'noise_scale.json').write_text(json.dumps(noise,indent=2))
# Compare historical controlled scalar results without loading graph/model payloads.
pairs=json.loads((ROOT/'docs/pdp_research/stage_u4_four_configuration_diagnostics/G2_INDEPENDENT_SCALAR_RECHECK.json').read_text())
assert len(pairs)==12 and all(p['control_witnesses_pass'] for p in pairs)
for p in pairs:
 assert abs(p['difference_pp']-(p['coverage_z1']-p['coverage_z0']))<1e-10
 # Only inspect persisted scalar/control witness fields; never print IDs or secret material.
 f=ROOT/p['source'];hashes[p['source']]=hashlib.sha256(f.read_bytes()).hexdigest()
for rel in ['experiments/node_dp_slots/mechanism.py','experiments/node_dp_slots/aggregation.py','experiments/node_dp_slots/training.py','experiments/node_pdp_v2_projected/policy.py','experiments/uniform_node_dp/config.py','experiments/uniform_node_dp/research.py','docs/pdp_research/stage_u4_four_configuration_diagnostics/GAUSSIAN_NOISE_ANALYSIS.md','docs/pdp_research/stage_u4_four_configuration_diagnostics/OPTIMIZATION_RANKING_REVIEW.md','docs/pdp_research/stage_u4_four_configuration_diagnostics/G2_INDEPENDENT_SCALAR_RECHECK.json','outputs/research/g2-controlled-attribution-20261102-v1/restricted/G2_REPORT.md']:
 hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
(A/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
res=dict(status='PASS',rows=100,cells=20,seeds=[0,1,2,3,4],all_raw_scalar_checks_pass=True,all_attempts=100,failures=0,retries=0,new_training=0,new_inference=0,new_random_experiments=0,scope='existing scalar arithmetic and static source reads only')
(A/'FIVE_SEED_VERIFICATION.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
