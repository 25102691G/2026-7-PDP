"""Data-independent range accounting for B0; see docs/b0_integer_privacy.md."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction
from typing import Any, Dict


def positive_decimal(value: Any, name: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def epsilon_upper(rho: Fraction, delta: Decimal) -> Decimal:
    """Outward-rounded sufficient bound rho + 2 sqrt(rho log(1/delta)).

    ln and sqrt are correctly rounded to nearest by Decimal, irrespective
    of the active rounding mode; next_plus encloses their exact value.
    All remaining positive arithmetic is rounded toward +infinity.
    """
    if rho < 0 or not delta.is_finite() or not 0 < delta < 1:
        raise ValueError("rho must be nonnegative and delta must be in (0,1)")
    if not rho:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_CEILING
        rho_up = Decimal(rho.numerator) / Decimal(rho.denominator)
        log_up = delta.ln().copy_negate().next_plus()
        root_up = (rho_up * log_up).sqrt().next_plus()
        return rho_up + Decimal(2) * root_up


@dataclass(frozen=True)
class RangePrivacy:
    epsilon: str = "1"
    delta: str = "0.000001"
    clip_norm: str = "3"
    batch_size: int = 16
    steps: int = 100
    grid_bits: int = 20

    def __post_init__(self) -> None:
        positive_decimal(self.epsilon, "epsilon")
        delta = positive_decimal(self.delta, "delta")
        if delta >= 1:
            raise ValueError("delta must be less than one")
        positive_decimal(self.clip_norm, "clip_norm")
        positive_integer(self.batch_size, "batch_size")
        positive_integer(self.steps, "steps")
        if (isinstance(self.grid_bits, bool)
                or not isinstance(self.grid_bits, int)
                or not 0 <= self.grid_bits <= 40):
            raise ValueError("grid_bits must be an integer in [0,40]")
        if not 1 <= self.radius <= 2**52:
            raise ValueError("clip_norm/grid must be in [1,2**52]")

    @property
    def radius(self) -> int:
        scaled = Fraction(str(self.clip_norm)) * (1 << self.grid_bits)
        return scaled.numerator // scaled.denominator

    @property
    def sensitivity_lattice(self) -> int:
        return 2 * self.batch_size * self.radius

    def rho(self, sigma_lattice: int) -> Fraction:
        positive_integer(sigma_lattice, "sigma_lattice")
        return Fraction(
            self.steps * self.sensitivity_lattice**2,
            2 * sigma_lattice**2,
        )

    def calibrate(self) -> int:
        """Smallest integer scale accepted by the conservative certificate."""
        target = positive_decimal(self.epsilon, "epsilon")
        delta = positive_decimal(self.delta, "delta")
        low, high = 0, 1
        while epsilon_upper(self.rho(high), delta) > target:
            low, high = high, high * 2
        while low + 1 < high:
            middle = (low + high) // 2
            if epsilon_upper(self.rho(middle), delta) <= target:
                high = middle
            else:
                low = middle
        return high

    def certificate(self, sigma_lattice: int) -> Dict[str, Any]:
        rho = self.rho(sigma_lattice)
        bound = epsilon_upper(rho, positive_decimal(self.delta, "delta"))
        if bound > positive_decimal(self.epsilon, "epsilon"):
            raise ValueError("noise scale does not meet the requested budget")
        return {
            "schema": "b0-integer-range-v1",
            "adjacency": "unbounded_node_add_remove_with_attributes_and_edges",
            "protection_start": "prepared_training_graph",
            "release": "fixed_architecture_model_parameters_only",
            "noise_distribution": "exact_discrete_gaussian_integer",
            "randomness_assumption": "secret_independent_uniform_system_random_bits",
            "epsilon_target": str(self.epsilon),
            "delta": str(self.delta),
            "epsilon_upper": str(bound),
            "rho_numerator": rho.numerator,
            "rho_denominator": rho.denominator,
            "clip_norm": str(self.clip_norm),
            "grid_bits": self.grid_bits,
            "clip_radius_lattice": self.radius,
            "batch_size": self.batch_size,
            "optimizer_steps": self.steps,
            "sensitivity_sum_lattice": self.sensitivity_lattice,
            "sensitivity_mean": str(Fraction(2 * self.radius, 1 << self.grid_bits)),
            "sigma_lattice": sigma_lattice,
            "noise_scale_sum": str(Fraction(sigma_lattice, 1 << self.grid_bits)),
            "noise_scale_mean": str(Fraction(
                sigma_lattice, self.batch_size * (1 << self.grid_bits)
            )),
            "subgraph_privacy_amplification": False,
            "proof": "docs/b0_integer_privacy.md",
        }
