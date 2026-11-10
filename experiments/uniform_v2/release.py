"""Uniform-v2 scientific results never acquire production release authority."""


def export(run, *, permit_idealized=False):
    raise PermissionError('Uniform-v2 research-only; production release not authorized')
