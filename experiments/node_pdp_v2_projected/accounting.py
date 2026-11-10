"""Fresh idealized derivation/check; never import the historical PrivIM formula."""
import math
from experiments.node_dp_slots import accounting as gaussian
from experiments.node_pdp import accounting as pdp


def base_account(spec):
    # Independently evaluate the frozen complete-training RDP conversion.
    rows=[]
    for alpha in gaussian.ORDERS:
        rho=spec.T*alpha*(2*spec.C*spec.K)**2/(2*spec.s_sum**2)
        eps=rho-(math.log(1e-5)+math.log(alpha))/(alpha-1)+math.log((alpha-1)/alpha)
        rows.append((max(0.,eps),alpha))
    eps,order=min(rows)
    reference=pdp.base_account(spec)
    if abs(eps-reference['epsilon'])>1e-12 or order!=reference['rdp_order']:
        raise ValueError('independent and shared frozen accountants disagree')
    return dict(epsilon=eps,delta=1e-5,s_sum=spec.s_sum,rdp_order=order,T=spec.T,
                full_training=True,sensitivity_sum=2*spec.C*spec.K,idealized_only=True)


def node_accounts(registry,spec):
    pdp.validate_public(registry,spec)
    actual=base_account(spec);rows=pdp.node_accounts(registry,spec)
    for row in rows:
        p=row['retention_probability'];e=actual['epsilon'];d=actual['delta']
        forward=math.log1p(p*math.expm1(e))
        reverse=-math.log(1-p+p*math.exp(-e))
        reverse_delta=p*d/(math.exp(e)*(1-p)+p)
        if (abs(row['epsilon_prime']-forward)>1e-12 or abs(row['delta_prime']-p*d)>1e-15 or
            abs(row['epsilon_reverse']-reverse)>1e-12 or abs(row['delta_reverse']-reverse_delta)>1e-15):
            raise ValueError('bidirectional personalized accountant mismatch')
    return rows
