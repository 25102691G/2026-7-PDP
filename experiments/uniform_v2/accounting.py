"""Uniform-v2 independent Gaussian/RDP conversion; no Ledger or amplification.

Applies to ideal full-rank Gaussian and the conditional fixed-domain theorem.
The delta below is the complete 100-step value, never a per-step value.
"""
import math
from experiments.node_pdp.registry import Registry
from .config import validate


def account():
    rows = []
    for k in range(11, 1000):
        alpha = k / 10
        rho = 100 * alpha * 144 / (2 * 486.**2)
        epsilon = rho - (math.log(1e-5) + math.log(alpha)) / (alpha - 1) + math.log((alpha - 1) / alpha)
        rows.append((max(0., epsilon), alpha))
    epsilon, order = min(rows)
    if not 0 <= epsilon <= 1:
        raise ValueError('UNIFORM_V2_THEORY_BLOCKED: numeric policy bound')
    return dict(epsilon=epsilon, delta=1e-5, rdp_order=order, sensitivity_sum=12.,
                s_sum=486., T=100, dimension=2340, full_training=True,
                outer_amplification=False, idealized_only=True, machine_dp_certified=False)


def node_accounts(registry, spec):
    validate(spec)
    if type(registry) is not Registry or registry.universe != tuple(sorted(spec.universe)):
        raise ValueError('fixed Registry must cover exactly public U')
    registry.__post_init__()
    result = account()
    # Intentionally do not use Registry.entry(...).retention_probability:
    # those probabilities belong to PDP, whereas this mechanism has p=1.
    return [dict(identity=v, epsilon_target=target, delta_target=1e-5,
                 retention_probability=1., epsilon_upper=result['epsilon'], delta_upper=result['delta'])
            for v, target in registry.budgets]
