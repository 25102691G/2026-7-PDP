"""Check all100 development scalar rows, raw saved evaluations and summaries."""
from pathlib import Path
import csv,json,statistics,hashlib
P=Path(__file__).resolve().parents[1];ROOT=P.parent
src=ROOT/'docs/pdp_research/stage_u5_research_decision/PER_SEED_RECHECK.csv'
rows=list(csv.DictReader(src.open()));assert len(rows)==100
summary=list(csv.DictReader((ROOT/'docs/pdp_research/stage_u4_four_configuration_diagnostics/GROUP_STATISTICS.csv').open()))
raw_hashes={}; maxerr=0
for r in rows:
 f=ROOT/r['source'];o=json.loads(f.read_text()); raw_hashes[r['source']]=hashlib.sha256(f.read_bytes()).hexdigest()
 spread=o.get('spread',o.get('influence_spread'));cov=o.get('coverage',o.get('coverage_ratio_percent'))
 assert spread is not None and cov is not None,f
 assert abs(float(spread)-float(r['spread']))<1e-10
 assert abs(float(cov)-float(r['coverage']))<1e-10
 assert abs(float(r['coverage'])-100*float(r['spread'])/float(r['celf']))<1e-10
 assert [r[k] for k in ['attempts','failures','retries','unexecuted']]==['1','0','0','0']
for s in summary:
 rr=[r for r in rows if r['dataset']==s['dataset'] and r['configuration']==s['configuration']]
 assert len(rr)==5 and {int(r['seed']) for r in rr}==set(range(5))
 x=[float(r['coverage']) for r in rr]
 for got,key in [(statistics.mean(x),'mean'),(statistics.stdev(x),'sample_sd'),(min(x),'minimum'),(max(x),'maximum')]:
  err=abs(got-float(s[key]));assert err<1e-10;maxerr=max(maxerr,err)
 assert sum(y<50 for y in x)==int(s['strict_below50'])
# Record local document/source hashes without reading graph/checkpoint payloads.
files=[Path('AGENTS.md'),Path('Dissertation - Guidelines and Assessment.pdf'),Path('PrivIM_Report.pdf')]
for dirname in ['stage_u0_uniform_v2','stage_u1_experiment_design_audit','stage_u2_uniform_eps4_mechanism','stage_u3_uniform_eps4_development','stage_u4_four_configuration_diagnostics','stage_u5_research_decision','stage_u6p_confirmation_protocol','stage_u6d_final_decision','stage_u6_independent_confirmation','stage_u6r2_selective_continuation','stage_u6a_posttraining_analysis','stage_r0_research_positioning','new_matched_pdp_development','stage_c3_rowsum']:
 d=Path('docs/pdp_research')/dirname
 files.extend(p for p in d.iterdir() if p.is_file() and p.suffix in {'.md','.py','.csv'} and p.stat().st_size<5000000)
raw_hashes.update({str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
(P/'audit/evidence_source_sha256.json').write_text(json.dumps(raw_hashes,indent=2))
res={'status':'PASS','all100_raw_evaluations_match':True,'all20_development_summaries_match':True,'max_absolute_error':maxerr,'development_seeds_retained':[0,1,2,3,4],'no_model_or_graph_access':True}
(P/'audit/DEVELOPMENT_VERIFICATION.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
