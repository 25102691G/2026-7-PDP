"""Exact discrete Gaussian via rational rejection sampling.

Independent implementation of the algorithms in Canonne, Kamath, Steinke,
The Discrete Gaussian for Differential Privacy, Section 5, Algorithms 1--3.
https://arxiv.org/html/2004.00010

Only integer uniform draws and exact rational comparisons determine samples.
No floating exp, approximate Gaussian, public noise seed, or retry cutoff.
"""

from __future__ import annotations

from fractions import Fraction
import secrets
from typing import Callable, List

from .accounting import positive_integer


def _unit_exponential_coin(
    probability: Fraction, uniform: Callable[[int], int],
) -> bool:
    """Bernoulli(exp(-x)) for rational x in [0,1], via alternating series."""
    index = 1
    while uniform(probability.denominator * index) < probability.numerator:
        index += 1
    return bool(index % 2)


def _exponential_coin(
    exponent: Fraction, uniform: Callable[[int], int],
) -> bool:
    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    whole, remainder = divmod(exponent.numerator, exponent.denominator)
    for _ in range(whole):
        if not _unit_exponential_coin(Fraction(1), uniform):
            return False
    return _unit_exponential_coin(
        Fraction(remainder, exponent.denominator), uniform
    )


def _two_sided_geometric(
    scale: int, uniform: Callable[[int], int],
) -> int:
    while True:
        residue = uniform(scale)
        if not _exponential_coin(Fraction(residue, scale), uniform):
            continue
        blocks = 0
        while _unit_exponential_coin(Fraction(1), uniform):
            blocks += 1
        magnitude = residue + blocks * scale
        negative = uniform(2)
        if negative and magnitude == 0:
            continue
        return -magnitude if negative else magnitude


def _gaussian_integer(
    scale: int, uniform: Callable[[int], int],
) -> int:
    variance = scale * scale
    proposal_scale = scale + 1
    while True:
        candidate = _two_sided_geometric(proposal_scale, uniform)
        displacement = abs(candidate) * proposal_scale - variance
        exponent = Fraction(
            displacement * displacement,
            2 * variance * proposal_scale * proposal_scale,
        )
        if _exponential_coin(exponent, uniform):
            return candidate


class DiscreteGaussian:
    """Production noise source: fresh secret system randomness, no seed API."""

    def __init__(self, sigma_lattice: int):
        self.sigma_lattice = positive_integer(sigma_lattice, "sigma_lattice")

    def sample(self, dimension: int) -> List[int]:
        positive_integer(dimension, "dimension")
        return [
            _gaussian_integer(self.sigma_lattice, secrets.randbelow)
            for _ in range(dimension)
        ]
