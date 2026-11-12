"""Plot all existing five-seed outcomes. Deterministic offsets; no RNG or inference."""
from pathlib import Path
import os,csv,statistics
P=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(P/'tooling/mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
rows=list(csv.DictReader((P/'audit/development_100_rows.csv').open()))
D=['email','bitcoin','lastfm','hepph','facebook'];M=['Non-Private','Uniform epsilon=1','PDP v2','Uniform epsilon=4']
colors=['#333333','#bf6b2f','#21638c','#65904a'];names=['NP','Uniform1','PDP v2','Uniform4']
fig,axs=plt.subplots(3,2,figsize=(7.3,8.6),sharey=True)
for d,l,ax in zip(D,['Email','Bitcoin','LastFM','HepPh','Facebook'],axs.flat):
 for i,(m,c) in enumerate(zip(M,colors)):
  rr=sorted([r for r in rows if r['dataset']==d and r['configuration']==m],key=lambda r:int(r['seed']))
  assert [int(r['seed']) for r in rr]==list(range(5))
  y=[float(r['coverage']) for r in rr]
  ax.scatter([i+(s-2)*.055 for s in range(5)],y,s=19,color=c,alpha=.8,zorder=3)
  ax.errorbar(i,statistics.mean(y),yerr=statistics.stdev(y),fmt='D',ms=4,color=c,capsize=4,lw=1.2)
 ax.axhline(50,color='#888888',ls='--',lw=.7)
 ax.set_title(l);ax.set_xticks(range(4),names,rotation=18);ax.set_ylim(-5,122);ax.grid(axis='y',alpha=.15)
axs[2,1].axis('off');axs[2,1].text(.05,.65,'Each point: one saved fit\nDiamond: mean of all 5 fits\nBar: sample SD (not CI)\nDashed line: Coverage = 50%\nFixed initialization seeds 0–4',transform=axs[2,1].transAxes,fontsize=10,va='top',linespacing=1.8)
fig.supylabel('Coverage (% of frozen CELF reference)');fig.tight_layout();fig.savefig(P/'figures/five_seed_coverage.pdf',bbox_inches='tight');plt.close(fig)
print('Created five-seed figure from all 100 saved outcomes; no RNG/model/graph access.')
