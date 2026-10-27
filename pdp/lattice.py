"""Exact integer-ball enforcement after arbitrary floating-point gradients."""

from __future__ import annotations

import math
from typing import Iterable, List

from .accounting import positive_integer


def project_integer_ball(values: Iterable[int], radius: int) -> List[int]:
    positive_integer(radius, "radius")
    vector = list(values)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in vector):
        raise ValueError("integer projection requires Python integers")
    squared = sum(value * value for value in vector)
    if squared <= radius * radius:
        return vector
    length = math.isqrt(squared)
    if length * length < squared:
        length += 1
    return [
        (1 if value >= 0 else -1) * (abs(value) * radius // length)
        for value in vector
    ]


def quantize_gradient(
    values: Iterable[float], *, radius: int, grid_bits: int
) -> List[int]:
    """Clip approximately, truncate, then enforce the exact proved radius.

    A nonfinite component zeroes the entire per-subgraph contribution.
    The exact final projection, rather than floating norm accuracy, is the
    privacy invariant. Gradient magnitudes below the clip bound are retained.
    """
    positive_integer(radius, "radius")
    if (isinstance(grid_bits, bool) or not isinstance(grid_bits, int)
            or not 0 <= grid_bits <= 40):
        raise ValueError("grid_bits must be an integer in [0,40]")
    vector = [float(value) for value in values]
    if any(not math.isfinite(value) for value in vector):
        return [0] * len(vector)
    maximum = max((abs(value) for value in vector), default=0.0)
    if maximum == 0:
        return [0] * len(vector)
    normalized = [value / maximum for value in vector]
    length = math.sqrt(math.fsum(value * value for value in normalized))
    # maximum*2**grid_bits may overflow to infinity; min still chooses the
    # finite radius/length. Integer conversion therefore remains finite.
    factor = min(maximum * (1 << grid_bits), radius / length)
    quantized = [math.trunc(value * factor) for value in normalized]
    return project_integer_ball(quantized, radius)


def sum_contributions(
    contributions: Iterable[Iterable[int]], *, radius: int,
    batch_size: int, dimension: int,
) -> List[int]:
    """Revalidate each integer contribution and reject a batch exceeding B."""
    positive_integer(batch_size, "batch_size")
    positive_integer(dimension, "dimension")
    result = [0] * dimension
    for index, contribution in enumerate(contributions):
        if index >= batch_size:
            raise ValueError("more than B gradient contributions")
        vector = project_integer_ball(contribution, radius)
        if len(vector) != dimension:
            raise ValueError("gradient dimension differs from the public model")
        result = [left + right for left, right in zip(result, vector)]
    return result
