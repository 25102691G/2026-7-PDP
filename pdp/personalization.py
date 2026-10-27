"""One-shot node sampling with conservative, exact-rational probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import hashlib
import secrets
from typing import Any, Dict, Tuple

from .accounting import positive_decimal


@dataclass(frozen=True)
class NodeBudget:
    epsilon: str
    delta: str

    def __post_init__(self) -> None:
        epsilon = Decimal(str(self.epsilon))
        delta = Decimal(str(self.delta))
        if not epsilon.is_finite() or epsilon < 0:
            raise ValueError("node epsilon must be finite and nonnegative")
        if not delta.is_finite() or not 0 <= delta < 1:
            raise ValueError("node delta must be in [0,1)")


def retention_probability(
    budget: NodeBudget, epsilon_0: str, delta_0: str,
) -> Fraction:
    """A rational p no larger than either the epsilon or delta constraint."""
    base_epsilon = positive_decimal(epsilon_0, "epsilon_0")
    base_delta = positive_decimal(delta_0, "delta_0")
    if base_epsilon > 64 or base_delta >= 1:
        raise ValueError("sampling requires epsilon_0 in (0,64], delta_0 in (0,1)")
    epsilon = Decimal(str(budget.epsilon))
    delta = Decimal(str(budget.delta))
    if epsilon == 0 or delta == 0:
        return Fraction(0)
    if epsilon >= base_epsilon:
        epsilon_probability = Fraction(1)
    else:
        with localcontext() as context:
            context.prec = 90
            context.rounding = ROUND_FLOOR
            numerator_low = epsilon.exp().next_minus() - Decimal(1)
            context.rounding = ROUND_CEILING
            denominator_high = base_epsilon.exp().next_plus() - Decimal(1)
            context.rounding = ROUND_FLOOR
            ratio_low = max(Decimal(0), numerator_low) / denominator_high
            numerator = int(ratio_low * (1 << 64))
        epsilon_probability = Fraction(numerator, 1 << 64)
    delta_probability = Fraction(str(budget.delta)) / Fraction(str(delta_0))
    return min(Fraction(1), epsilon_probability, delta_probability)


def amplified_bounds(
    probability: Fraction, epsilon_0: str, delta_0: str,
) -> Tuple[Decimal, Fraction]:
    """Outward epsilon bound and exact delta bound, without graph statistics."""
    if not isinstance(probability, Fraction) or not 0 <= probability <= 1:
        raise ValueError("probability must be a Fraction in [0,1]")
    base_epsilon = positive_decimal(epsilon_0, "epsilon_0")
    base_delta = positive_decimal(delta_0, "delta_0")
    if base_epsilon > 64 or base_delta >= 1:
        raise ValueError("invalid base budget for amplification")
    delta = probability * Fraction(str(delta_0))
    if probability == 0:
        return Decimal(0), delta
    if probability == 1:
        return base_epsilon, delta
    with localcontext() as context:
        context.prec = 90
        context.rounding = ROUND_CEILING
        p_up = Decimal(probability.numerator) / Decimal(probability.denominator)
        exponential_up = base_epsilon.exp().next_plus()
        coefficient_up = Decimal(1) + p_up * (exponential_up - Decimal(1))
        epsilon_up = coefficient_up.ln().next_plus()
    return epsilon_up, delta


@dataclass(frozen=True)
class HashBudgetPolicy:
    """Public policy on all integer IDs; independent of presence and topology.

    The hash assigns budget groups, not private sampling randomness.
    Every existing or absent ID has a well-defined budget under the same policy.
    """
    groups: Tuple[NodeBudget, ...] = (
        NodeBudget("1", "0.000001"),
        NodeBudget("2", "0.000001"),
        NodeBudget("4", "0.000001"),
    )
    salt: str = "privim-pdp-policy-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        if not self.groups or any(not isinstance(x, NodeBudget) for x in self.groups):
            raise ValueError("policy must contain valid external budgets")
        if not isinstance(self.salt, str):
            raise ValueError("public policy salt must be a string")

    def group_for(self, stable_id: int) -> int:
        if isinstance(stable_id, bool) or not isinstance(stable_id, int):
            raise ValueError("stable node IDs must be integers")
        digest = hashlib.sha256(
            (self.salt + ":" + str(stable_id)).encode("utf-8")
        ).digest()
        return int.from_bytes(digest, "big") % len(self.groups)

    def probabilities(self, epsilon_0: str, delta_0: str) -> Tuple[Fraction, ...]:
        return tuple(
            retention_probability(budget, epsilon_0, delta_0)
            for budget in self.groups
        )

    def public_description(self) -> Dict[str, Any]:
        return {"policy": "public_sha256_group_on_stable_integer_id", "policy_salt": self.salt}

    def certificate(self, epsilon_0: str, delta_0: str) -> Dict[str, Any]:
        descriptions = []
        for budget, probability in zip(
            self.groups, self.probabilities(epsilon_0, delta_0)
        ):
            epsilon, delta = amplified_bounds(probability, epsilon_0, delta_0)
            if epsilon > Decimal(str(budget.epsilon)):
                raise ValueError("computed epsilon bound exceeds node policy")
            if delta > Fraction(str(budget.delta)):
                raise ValueError("computed delta bound exceeds node policy")
            descriptions.append({
                "epsilon_target": str(budget.epsilon),
                "delta_target": str(budget.delta),
                "probability_numerator": probability.numerator,
                "probability_denominator": probability.denominator,
                "epsilon_upper": str(epsilon),
                "delta_upper_numerator": delta.numerator,
                "delta_upper_denominator": delta.denominator,
            })
        return {
            "schema": "one-shot-node-pdp-v1",
            **self.public_description(),
            "groups": descriptions,
            "outer_sampling_count": 1,
            "privacy_proof": "docs/p0_pdp_feasibility.md",
            "graph_members_and_sampling_randomness_released": False,
        }


def sample_nodes_once(graph: Any, policy: HashBudgetPolicy,
                      epsilon_0: str, delta_0: str) -> Any:
    """Induce on independently retained vertices, preserving isolates and IDs."""
    probabilities = policy.probabilities(epsilon_0, delta_0)
    if "stable_id" not in graph.vs.attributes():
        raise ValueError("graph must carry stable_id across node deletions")
    stable_ids = list(graph.vs["stable_id"])
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("stable_id values must be unique")
    selected = []
    for index, stable_id in enumerate(stable_ids):
        probability = probabilities[policy.group_for(stable_id)]
        if probability == 1 or (
            probability and secrets.randbelow(probability.denominator) < probability.numerator
        ):
            selected.append(index)
    return graph.induced_subgraph(selected)


def train_pdp(graph: Any, config: Any, policy: HashBudgetPolicy, *,
              trusted_diagnostics: Any = None) -> Any:
    """Apply outer amplification once to the entire verified B0 training."""
    from .training import ParameterRelease, train_b0

    policy_certificate = policy.certificate(config.privacy.epsilon, config.privacy.delta)
    retained = sample_nodes_once(
        graph, policy, config.privacy.epsilon, config.privacy.delta,
    )
    if trusted_diagnostics is not None:
        trusted_diagnostics["outer_input_nodes"] = graph.vcount()
        trusted_diagnostics["outer_retained_nodes"] = retained.vcount()
        trusted_diagnostics["outer_retained_edges"] = retained.ecount()
    base = train_b0(retained, config, trusted_diagnostics=trusted_diagnostics)
    certificate = dict(base.certificate)
    certificate["personalization"] = policy_certificate
    return ParameterRelease(base.parameters, certificate)
