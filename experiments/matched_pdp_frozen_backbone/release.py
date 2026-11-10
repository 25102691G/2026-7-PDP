"""No production release authorized for this independent research mechanism."""


def export(run, *, permit_idealized=False):
    raise PermissionError('new Matched PDP is research-only; production release not authorized')
