"""Independent exact rational certificate of the PUBLIC Uniform-v2 DP target."""
from fractions import Fraction as F


def log_bounds(x, n=256):
    x = F(x)
    if x < 1:
        raise ValueError('log enclosure requires x >= 1')
    z = (x - 1) / (x + 1)
    lower = 2 * sum((z**(2*k+1) / (2*k+1) for k in range(n)), F(0))
    return lower, lower + 2*z**(2*n+1) / ((2*n+1)*(1-z*z))


def verify():
    alpha = F(89, 5)
    l10, u10 = log_bounds(10)
    la, ua = log_bounds(alpha)
    lr, ur = log_bounds(alpha / (alpha-1))
    rho = 100 * alpha * 144 / (2 * 486**2)
    lower = rho + (5*l10-ua)/(alpha-1) - ur
    upper = rho + (5*u10-la)/(alpha-1) - lr
    if not (upper < F('0.998694') < 1 and upper-lower < F(1, 10**20)):
        raise ValueError('UNIFORM_V2_THEORY_BLOCKED: exact certificate failed')
    return dict(alpha=str(alpha), terms=256, arithmetic='exact Fraction',
                epsilon_lower_reference=float(lower), epsilon_upper_reference=float(upper),
                certified_strict_upper='0.998694', epsilon_strictly_below_one=True,
                interval_width_less_than='1e-20', delta='1e-5', dimension=2340,
                sigma_sum=486, sensitivity_sum=12, T=100, machine_dp_certified=False)
