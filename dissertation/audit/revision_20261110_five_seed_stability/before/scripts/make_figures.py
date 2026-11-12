"""Scientific figures using existing scalar records only; no random sampling."""
from pathlib import Path
import os,json
P=Path(__file__).resolve().parents[1];ROOT=P.parent
os.environ['MPLCONFIGDIR']=str(P/'tooling/mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
src=ROOT/'docs/pdp_research/stage_u6a_posttraining_analysis/runtime/outputs'
rows=json.loads((src/'observations.json').read_text());a=json.loads((src/'analysis.json').read_text())
D=['email','bitcoin','lastfm','hepph','facebook'];labels=['Email','Bitcoin','LastFM','HepPh','Facebook']
methods=['nonprivate','uniform1','pdp_v2','uniform4']; names=['Non-Private','Uniform1','PDP v2','Uniform4'];colors=['#333333','#bf6b2f','#21638c','#65904a']
fig,axs=plt.subplots(3,2,figsize=(7.5,8.3),sharex=True,sharey=True)
for d,label,ax in zip(D,labels,axs.flat):
 for m,name,c in zip(methods,names,colors):
  x=np.sort([r['coverage'] for r in rows if r['dataset']==d and r['method']==m]); y=np.arange(1,len(x)+1)/len(x)
  ax.step(x,y,where='post',label=name,color=c,lw=1.4)
 ax.axvline(50,color='#888888',ls='--',lw=.8);ax.set_title(label);ax.set_xlim(0,105);ax.set_ylim(0,1.02);ax.grid(alpha=.15)
axs[2,1].axis('off');handles,ls=axs[0,0].get_legend_handles_labels();axs[2,1].legend(handles,ls,loc='center',frameon=False)
fig.supxlabel('Coverage (% of frozen CELF reference)');fig.supylabel('Empirical cumulative probability');fig.tight_layout()
fig.savefig(P/'figures/coverage_distributions.pdf',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.4))
cs=[a['per_dataset_primary'][d] for d in D]+[a['primary']]
for i,c in enumerate(cs):
 e=c['estimate_pp'];lo,hi=c['ci_two_sided_95_pp'];ax.errorbar(e,5-i,xerr=[[e-lo],[hi-e]],fmt='D' if i==5 else 'o',color='#21638c' if i<5 else '#111111',capsize=4,lw=1.3,ms=6)
ax.axvline(0,color='#777777',lw=1);ax.axvline(3,color='#b36b2b',ls=':',lw=1);ax.text(3.1,.35,'3 pp planning target',fontsize=9,color='#9d5d21')
ax.set_yticks(range(6),list(reversed(labels+['Equal-weight mean'])));ax.set_xlabel('PDP v2 minus Uniform1 (percentage points)');ax.grid(axis='x',alpha=.18);fig.tight_layout();fig.savefig(P/'figures/primary_effects.pdf',bbox_inches='tight');plt.close(fig)
print('Created 2 figures from 2900 saved rows; no random computation.')
