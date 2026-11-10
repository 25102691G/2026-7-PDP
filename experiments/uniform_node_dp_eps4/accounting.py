"""Independent unamplified full100-step ideal Gaussian accounting and certificate."""
from functools import lru_cache
from fractions import Fraction as F
import math
from experiments.node_pdp.registry import Registry
from .config import validate


def gaussian_bound(*,sigma_sum=139.,steps=100,delta=1e-5,sensitivity=12.,dimension=2340,p_v=1.):
    if (not all(math.isfinite(x) for x in (sigma_sum,delta,sensitivity,p_v)) or sigma_sum<=0 or
            not 0<delta<1 or sensitivity<0 or type(steps) is not int or steps<=0 or
            dimension!=2340 or p_v!=1.):
        raise ValueError('invalid full-dimensional unamplified Gaussian domain')
    # Form the ratio before squaring: finite huge numerator/denominator must
    # never become inf/inf -> NaN and then be clamped to an epsilon of zero.
    ratio=sensitivity/sigma_sum
    squared_ratio=ratio*ratio
    if (not math.isfinite(squared_ratio) or (sensitivity>0 and squared_ratio==0)):
        raise ValueError('Gaussian RDP ratio outside reliable float range')
    candidates=[]
    for k in range(11,1000):
        alpha=k/10
        try:rho=(steps*alpha/2)*squared_ratio
        except OverflowError:raise ValueError('nonfinite Gaussian RDP intermediate') from None
        if not math.isfinite(rho):raise ValueError('nonfinite Gaussian RDP intermediate')
        eps=rho-(math.log(delta)+math.log(alpha))/(alpha-1)+math.log((alpha-1)/alpha)
        if not math.isfinite(eps):raise ValueError('nonfinite RDP conversion result')
        candidates.append((max(0.,eps),alpha))
    epsilon,alpha=min(candidates)
    return dict(epsilon_numeric=epsilon,rdp_order=alpha,delta=delta,sigma_sum=sigma_sum,
                steps=steps,sensitivity_sum=sensitivity,dimension=dimension,outer_amplification=False)


def require_target(bound,target):
    epsilon=bound['epsilon_numeric']
    if not math.isfinite(target) or target<0 or not math.isfinite(epsilon) or epsilon<0 or epsilon>target:
        raise ValueError('Gaussian calibration does not satisfy target')


def _log_bounds(x):
    x=F(x)
    if x<1:raise ValueError('certificate logarithm requires x>=1')
    z=(x-1)/(x+1)
    lower=2*sum((z**(2*k+1)/F(2*k+1) for k in range(256)),F(0))
    return lower,lower+2*z**513/(513*(1-z*z))


@lru_cache(maxsize=1)
def _certificate_values():
    a=F(61,10);l10,u10=_log_bounds(10);la,ua=_log_bounds(a);lr,ur=_log_bounds(a/(a-1))
    rho=100*a*144/(2*139**2)
    lower=rho+(5*l10-ua)/(a-1)-ur;upper=rho+(5*u10-la)/(a-1)-lr
    if not lower<=upper<F('3.997')<4 or upper-lower>=F(1,10**20):
        raise ValueError('UNIFORM_EPS4_THEORY_BLOCKED: exact bound fails')
    return str(lower),str(upper)


def exact_certificate(spec):
    validate(spec)
    lower,upper=_certificate_values()
    return dict(alpha='61/10',terms=256,lower_fraction=lower,upper_fraction=upper,
                certified_strict_upper='3.997',interval_width_less_than='1e-20',arithmetic='exact Fraction')


def account(spec):
    validate(spec)
    bound=gaussian_bound(sigma_sum=spec.s_sum,steps=spec.T,delta=spec.target_delta,
                         sensitivity=spec.sensitivity_sum,dimension=spec.dimension,p_v=spec.p_v)
    require_target(bound,spec.target_epsilon)
    certificate=exact_certificate(spec)
    return dict(bound,epsilon_upper_bound=float(certificate['certified_strict_upper']),
                target_epsilon=4.,target_delta=1e-5,full_training=True,
                idealized_only=True,machine_dp_certified=False,
                epsilon_numeric_is_reference=True,certificate_strict_upper='3.997')


def node_accounts(registry,spec):
    validate(spec)
    if type(registry) is not Registry or registry.universe!=tuple(sorted(spec.universe)):
        raise ValueError('fixed public Registry must exactly cover U')
    registry.__post_init__();bound=account(spec)
    return [dict(identity=v,p_v=1.,target_epsilon=4.,delta=1e-5,
                 epsilon_upper_bound=bound['epsilon_upper_bound'],original_policy_target=target,
                 original_policy_certified=bound['epsilon_upper_bound']<=target)
            for v,target in registry.budgets]
