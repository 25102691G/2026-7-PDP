"""No production release permission for the new scientific mechanism."""


def export(run,*,permit_idealized=False):
    raise PermissionError('Uniform epsilon4 research-only; production release forbidden')
