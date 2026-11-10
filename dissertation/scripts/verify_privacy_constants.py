"""Public-constant arithmetic only; not a privacy accountant replacement/proof."""
from decimal import Decimal,localcontext
from pathlib import Path
import json
P=Path(__file__).resolve().parents[1]
with localcontext() as c:
 c.prec=70;D=Decimal;delta=D('0.00001');out={}
 for sig in [260,486,139]:
  pairs=[]
  for i in range(11,1000):
   alpha=D(i)/10;rho=D(100)*alpha*144/(2*D(sig)**2)
   eps=rho-(delta.ln()+alpha.ln())/(alpha-1)+((alpha-1)/alpha).ln()
   pairs.append((eps,alpha))
  e,a=min(pairs);out[str(sig)]={'epsilon_decimal':str(e),'order':str(a)}
 p=(D(1).exp()-1)/(D(2).exp()-1);e0=D(out['260']['epsilon_decimal']);ep=(1+p*(e0.exp()-1)).ln();dp=p*delta
 assert D(out['260']['epsilon_decimal'])<D('1.982093')<2
 assert D(out['486']['epsilon_decimal'])<D('.998694')<1
 assert D(out['139']['epsilon_decimal'])<D('3.997')<4
 assert abs(ep-D('.986940353796761'))<D('1e-14')
 out['target1']={'p':str(p),'epsilon_bound_decimal':str(ep),'delta_bound_decimal':str(dp)}
 out['status']='PASS';out['scope']='Deterministic70-digit public formula check, not a new accountant, exact certificate, privacy proof or machine certification.'
(P/'audit/PRIVACY_CONSTANTS_VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
