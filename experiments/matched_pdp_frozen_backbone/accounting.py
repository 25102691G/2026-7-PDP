"""Independent33D ideal accountant. See new_matched_pdp/PRIVACY_PROOF.md.

No production accountant, ledger or authorization semantics are modified.
"""
import math
from experiments.node_pdp.registry import Registry
from .config import validate

ORDERS = tuple(i/10 for i in range(11,1000))


def base_account(spec):
    validate(spec)
    values = []
    for alpha in ORDERS:
        rho = spec.T*alpha*(2*spec.C*spec.K)**2/(2*spec.s_sum**2)
        eps = rho-(math.log(1e-5)+math.log(alpha))/(alpha-1)+math.log((alpha-1)/alpha)
        values.append((max(0.,eps),alpha))
    epsilon, order = min(values)
    return dict(epsilon=epsilon, delta=1e-5, rdp_order=order, T=100,
                sensitivity_sum=12., s_sum=260., dimension=33,
                full_training=True, conditional_ideal_only=True)


def amplify(epsilon, delta, p):
    if not all(math.isfinite(x) for x in (epsilon,delta,p)) or epsilon < 0 or not 0 <= delta <= 1 or not 0 <= p <= 1:
        raise ValueError('finite epsilon and valid probabilities required')
    if p == 0:
        forward = reverse = 0.
    elif p == 1:
        forward = reverse = epsilon
    else:
        forward = (math.log1p(p*math.expm1(epsilon)) if epsilon < 700 else
                   epsilon+math.log(p+(1-p)*math.exp(-epsilon)))
        reverse = -math.log1p(-p*(-math.expm1(-epsilon)))
    denominator = (1-p)+p*math.exp(-epsilon)
    factor = math.exp(-epsilon)/denominator if p != 1 else 1.
    return dict(epsilon_prime=max(0.,forward), delta_prime=p*delta,
                epsilon_reverse=max(0.,reverse), delta_reverse=p*delta*factor)


def node_accounts(registry, spec):
    validate(spec)
    if type(registry) is not Registry or registry.universe != spec.universe:
        raise ValueError('same complete fixed public U/Registry required')
    b = base_account(spec)
    rows = []
    for identity in registry.universe:
        entry = registry.entry(identity)
        row = dict(vars(entry), **amplify(b['epsilon'],b['delta'],entry.retention_probability))
        row['target_satisfied'] = row['epsilon_prime'] <= entry.epsilon_target and row['delta_prime'] <= entry.delta_target
        rows.append(row)
    if not all(row['target_satisfied'] for row in rows):
        raise ValueError('frozen target bound failed')
    return rows
