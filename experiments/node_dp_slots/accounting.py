"""Unamplified conditional Gaussian RDP; no dependency on legacy accountant."""
import math
import threading
import numpy as np

ORDERS=tuple(k/10 for k in range(11,1000))


def curve(std=260.,models=1):
    if not math.isfinite(std) or std<=0 or type(models) is not int or models<0:
        raise ValueError('positive fixed scale and nonnegative model count required')
    return np.asarray(ORDERS)*100*12**2*models/(2*std**2)


def epsilon(std=260.,models=1):
    alpha=np.asarray(ORDERS)
    values=curve(std,models)-(math.log(1e-5)+np.log(alpha))/(alpha-1)+np.log((alpha-1)/alpha)
    index=int(np.argmin(values))
    return max(0.,float(values[index])),ORDERS[index]


def minimum_scale():
    from scipy.optimize import brentq
    return brentq(lambda s:epsilon(s)[0]-2,100,1000)


class Ledger:
    """Process-local composition guard, NOT a persistent deployment ledger."""
    def __init__(self):self._models=0;self._lock=threading.Lock()

    @property
    def models(self):return self._models

    def charge(self):
        with self._lock:
            if epsilon(models=self._models+1)[0]>2.:
                raise ValueError('composed idealized budget exceeds2; no new graph access permitted')
            self._models+=1
