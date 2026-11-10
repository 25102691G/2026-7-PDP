"""No production publication path is authorized for C2 candidate outputs."""


def export(run,*,permit_idealized=False):
    raise PermissionError('C2 candidate research/artificial outputs are not publication-authorized')
