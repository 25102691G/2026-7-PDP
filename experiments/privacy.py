"""Self-contained DP-SGD and PrivIM RDP accounting utilities."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
from scipy import special, stats
import torch


PAPER_BINOMIAL = "paper_binomial"
FIXED_BATCH_HYPERGEOMETRIC = "fixed_batch_hypergeometric"
ACCOUNTANTS = (PAPER_BINOMIAL, FIXED_BATCH_HYPERGEOMETRIC)


def sensitivity_occurrences(
    accountant: str,
    max_occurrences: int,
    container_size: int,
    batch_size: int,
) -> int:
    """Return the occurrence multiplier used by the Gaussian mechanism.

    The paper states the loose global bound ``N_g``.  The fixed-size batch
    interpretation uses the tighter deterministic bound
    ``min(N_g, m, B)``.  Keeping this choice in one function prevents the
    accountant and optimizer from silently using different sensitivities.
    """
    if accountant not in ACCOUNTANTS:
        raise ValueError(f"unknown accountant: {accountant}")
    population = min(int(max_occurrences), int(container_size))
    if accountant == PAPER_BINOMIAL:
        return population
    return min(population, int(batch_size))


class PrivateSGD(torch.optim.SGD):
    """SGD with per-subgraph clipping and noise on the summed gradient."""

    def __init__(
        self,
        params,
        *,
        l2_norm_clip: float,
        noise_multiplier: float,
        minibatch_size: int,
        microbatch_size: int = 1,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.minibatch_size = minibatch_size
        self.microbatch_size = microbatch_size
        self.last_microbatch_norms = []
        self.last_clip_fraction = 0.0
        for group in self.param_groups:
            group["accum_grads"] = [
                torch.zeros_like(parameter) if parameter.requires_grad else None
                for parameter in group["params"]
            ]

    def zero_microbatch_grad(self) -> None:
        super().zero_grad()

    def zero_accum_grad(self) -> None:
        self.last_microbatch_norms = []
        self.last_clip_fraction = 0.0
        for group in self.param_groups:
            for gradient in group["accum_grads"]:
                if gradient is not None:
                    gradient.zero_()

    def microbatch_step(self) -> float:
        squared_norm = 0.0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad and parameter.grad is not None:
                    squared_norm += parameter.grad.detach().norm(2).item() ** 2
        total_norm = squared_norm ** 0.5
        self.last_microbatch_norms.append(total_norm)
        coefficient = min(self.l2_norm_clip / (total_norm + 1e-6), 1.0)
        for group in self.param_groups:
            for parameter, accumulated in zip(group["params"], group["accum_grads"]):
                if parameter.requires_grad and parameter.grad is not None:
                    accumulated.add_(parameter.grad.detach(), alpha=coefficient)
        return total_norm

    def step(self, closure=None):
        if self.last_microbatch_norms:
            clipped = sum(
                norm > self.l2_norm_clip for norm in self.last_microbatch_norms
            )
            self.last_clip_fraction = clipped / len(self.last_microbatch_norms)
        for group in self.param_groups:
            for parameter, accumulated in zip(group["params"], group["accum_grads"]):
                if parameter.requires_grad:
                    noisy_sum = accumulated.clone()
                    if self.noise_multiplier != 0:
                        noisy_sum.add_(
                            torch.randn_like(noisy_sum),
                            alpha=self.l2_norm_clip * self.noise_multiplier,
                        )
                    parameter.grad = noisy_sum.mul(
                        self.microbatch_size / self.minibatch_size
                    )
        return super().step(closure)


def rdp_to_dp(orders: np.ndarray, rdp: np.ndarray, delta: float) -> Tuple[float, float]:
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    epsilons = (
        rdp
        - (np.log(delta) + np.log(orders)) / (orders - 1)
        + np.log((orders - 1) / orders)
    )
    index = int(np.argmin(epsilons))
    return max(0.0, float(epsilons[index])), float(orders[index])


def multiterm_rdp(
    orders: Iterable[float],
    steps: int,
    sigma: float,
    container_size: int,
    max_occurrences: int,
    batch_size: int,
    accountant: str = FIXED_BATCH_HYPERGEOMETRIC,
) -> np.ndarray:
    """Compute the composed RDP bound under a named sampling interpretation.

    ``paper_binomial`` implements the binomial mixture printed in Theorem 3.
    ``fixed_batch_hypergeometric`` matches Algorithm 2's uniformly sampled,
    fixed-size batch and the optimizer used by this repository.
    """
    orders = np.asarray(list(orders), dtype=float)
    if sigma <= 0:
        return np.full_like(orders, np.inf)
    if not 0 < batch_size <= container_size:
        raise ValueError("batch_size must be in [1, container_size]")
    if accountant not in ACCOUNTANTS:
        raise ValueError(f"unknown accountant: {accountant}")
    affected_population = min(max_occurrences, container_size)
    sensitivity = sensitivity_occurrences(
        accountant, max_occurrences, container_size, batch_size
    )
    if sensitivity <= 0:
        return np.zeros_like(orders)

    if accountant == PAPER_BINOMIAL:
        probability = affected_population / container_size
        distribution = stats.binom(batch_size, probability)
        counts = np.arange(batch_size + 1)
    else:
        distribution = stats.hypergeom(
            container_size, affected_population, batch_size
        )
        counts = np.arange(min(affected_population, batch_size) + 1)
    log_probabilities = distribution.logpmf(counts)
    values = []
    for order in orders:
        unamplified = order / (2 * sigma ** 2)
        exponent = (
            unamplified
            * (order - 1)
            * np.square(counts / sensitivity)
        )
        values.append(special.logsumexp(log_probabilities + exponent) / (order - 1))
    return np.asarray(values) * steps
