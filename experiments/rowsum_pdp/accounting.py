"""Independent candidate accounting: phi sensitivity12,100 Gaussian queries; amplify once."""
import math
from .config import Spec,validate


def base_account(spec):
    if type(spec) is not Spec:raise TypeError('row-sum Spec required')
    if (spec.dimension,spec.T,spec.s_sum,spec.C,spec.K)!=(1283,100,260.,3.,2):raise ValueError('frozen accountant inputs changed')
    values=[]
    for k in range(11,1000):
        a=k/10;rho=100*a*144/(2*260.**2)
        eps=rho-(math.log(1e-5)+math.log(a))/(a-1)+math.log((a-1)/a)
        values.append((max(0.,eps),a))
    epsilon,order=min(values)
    return dict(epsilon=epsilon,delta=1e-5,rdp_order=order,T=100,dimension=1283,s_sum=260.,sensitivity_sum=12.,idealized_only=True,machine_pdp_certified=False)


def node_accounts(registry,spec):
    validate(registry,spec);base=base_account(spec);rows=[]
    for identity in registry.universe:
        entry=registry.entry(identity);p=entry.retention_probability
        e=math.log1p(p*math.expm1(base['epsilon']));d=p*base['delta']
        rows.append(dict(identity=identity,epsilon_target=entry.epsilon_target,epsilon_prime=e,delta_prime=d,
                         target_satisfied=e<=entry.epsilon_target and d<=entry.delta_target))
    if not all(r['target_satisfied'] for r in rows):raise ValueError('frozen per-node targets not met')
    return rows
