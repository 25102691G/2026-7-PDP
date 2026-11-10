"""Synthetic review is not authorization to publish a private model."""


def export(run,*,permit_idealized=False):
    raise PermissionError('C2-R artificial outputs are not production-publication authorized')
