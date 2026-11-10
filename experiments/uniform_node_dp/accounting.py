"""Independent G0 §§4–6 conditional Gaussian/adaptive RDP conversion.

Fixed add/remove sensitivity 2CK=12, T100, sigma486, delta1e-5.
No subsampling amplification. No production accountant or ledger mutation.
The theorem applies to ideal Gaussian innovations and fixed public auxiliary data;
this numeric calculation does not certify the finite-precision sampler.
"""
import math


def account():
    values=[]
    for k in range(11,1000):
        alpha=k/10
        rho=100*alpha*12**2/(2*486.**2)
        eps=rho-(math.log(1e-5)+math.log(alpha))/(alpha-1)+math.log((alpha-1)/alpha)
        values.append((max(0.,eps),alpha))
    epsilon,order=min(values)
    return dict(epsilon=epsilon,delta=1e-5,epsilon_target=1.,rdp_order=order,
                sensitivity_sum=12.,s_sum=486.,T=100,full_training=True,
                outer_amplification=False,idealized_only=True,machine_dp_certified=False,
                proof='docs/pdp_research/G0_THEORY_BOUNDARY_REVIEW.md sections4-6',
                release_boundary='fixed model parameters and genuinely public configuration only')
