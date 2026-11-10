"""One amplification of COMPLETE base training; independent of old PrivIM accounting."""
import math

from experiments.node_dp_slots import accounting as base
from experiments.node_dp_slots.config import Spec
from .registry import Registry


def validate_public(registry, spec):
    if type(registry) is not Registry or type(spec) is not Spec:
        raise TypeError('frozen external Registry and base Spec required')
    if registry.universe != spec.universe:
        raise ValueError('same canonical public U required for registry and base')
    expected = (16, 3., 100, 2, 2340, 260., .005)
    if (spec.B, spec.C, spec.T, spec.K, spec.dimension, spec.s_sum, spec.lr) != expected:
        raise ValueError('frozen base configuration changed')
    if spec.public()['delta'] != 1e-5:
        raise ValueError('frozen base delta changed')


def base_account(spec):
    if type(spec) is not Spec:
        raise TypeError('frozen base Spec required')
    epsilon, order = base.epsilon(std=spec.s_sum, models=1)
    return dict(epsilon=epsilon, delta=spec.public()['delta'], s_sum=spec.s_sum,
                rdp_order=order, T=spec.T, full_training=True)


def amplify(epsilon, delta, p):
    if not all(math.isfinite(x) for x in (epsilon, delta, p)) or epsilon < 0 or not 0 <= delta <= 1 or not 0 <= p <= 1:
        raise ValueError('finite nonnegative epsilon and probabilities in [0,1] required')
    # Stable algebra: log[(1-p)+p exp(epsilon)] without exp(epsilon) overflow.
    if p == 0:
        forward = reverse = 0.
    elif p == 1:
        forward = reverse = epsilon
    else:
        forward = (math.log1p(p*math.expm1(epsilon)) if epsilon < 700 else
                   epsilon + math.log(p + (1-p)*math.exp(-epsilon)))
        reverse = -math.log1p(-p*(-math.expm1(-epsilon)))
    reverse_factor = math.exp(-epsilon) / (1-p+p*math.exp(-epsilon)) if p != 1 else 1.
    return dict(epsilon_prime=max(0., forward), delta_prime=p*delta,
                epsilon_reverse=max(0., reverse), delta_reverse=p*delta*reverse_factor)


def node_accounts(registry, spec):
    validate_public(registry, spec); b = base_account(spec)
    rows = []
    for v in registry.universe:
        entry = registry.entry(v)
        row = dict(vars(entry), **amplify(b['epsilon'], b['delta'], entry.retention_probability))
        row['target_satisfied'] = (row['epsilon_prime'] <= entry.epsilon_target and
                                   row['delta_prime'] <= entry.delta_target)
        rows.append(row)
    if not all(row['target_satisfied'] for row in rows):
        raise ValueError('public personalized target not met by frozen base account')
    return rows
