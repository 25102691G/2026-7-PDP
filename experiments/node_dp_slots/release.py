"""Two-field serialization boundary; strict export blocked pending machine proof."""
import math
from .accounting import epsilon
from .mechanism import Run


def export(run,*,permit_idealized=False):
    if type(run) is not Run or not run._completed or run._mode!='privacy':
        raise PermissionError('testing/incomplete runs are never releasable')
    if not permit_idealized:
        raise PermissionError('UNRESOLVED finite-precision/OS-Gaussian proof; strict DP export blocked')
    if len(run._parameters)!=2340 or not all(math.isfinite(x) for x in run._parameters):
        raise PermissionError('invalid fixed-dimensional parameters')
    public=run._spec.public();public['idealized_epsilon'],public['rdp_order']=epsilon()
    return dict(parameters=list(run._parameters),configuration=public)
