"""Node-PDP from genuinely stable public query slots, not SCS occurrences.

Proof and intentional differences from PrivIM: docs/pdp_stable_slots_design.md.
Each slot reads only its own induced graph and has an isolated public RNG.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

import igraph
import torch

from .accounting import RangePrivacy, positive_integer
from .discrete_noise import DiscreteGaussian
from .lattice import project_integer_ball, sum_contributions
from .personalization import HashBudgetPolicy
from .positive_head import (
    PositiveHeadSettings, apply_head_update, head_contribution, initialize_positive_head,
)
from .training import (
    ParameterRelease, TrainingConfig, _apply_noisy_integer_update,
    _contribution, _trusted_computation, build_samples, initialize_model,
)


@dataclass(frozen=True)
class StableSlots:
    bins_per_round: int = 4
    rounds: int = 4
    appearance_limits: Tuple[int, ...] = (1, 2, 4)
    microbatch_size: int = 1
    salt: str = "privim-stable-slots-v1"
    uniform_epsilon: Optional[str] = None
    uniform_delta: Optional[str] = None
    trainable_mode: str = "full"
    head_settings: PositiveHeadSettings = field(default_factory=PositiveHeadSettings)

    def __post_init__(self) -> None:
        for name in ["bins_per_round", "rounds", "microbatch_size"]:
            positive_integer(getattr(self, name), name)
        if not isinstance(self.salt, str):
            raise ValueError("slot salt must be a public string")
        self.salt.encode("utf-8")
        # Freeze caller-owned sequences before data is read.
        object.__setattr__(self, "appearance_limits", tuple(self.appearance_limits))
        if not self.appearance_limits or any(
            isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= self.rounds
            for x in self.appearance_limits
        ):
            raise ValueError("appearance limits must be integers in [0,rounds]")
        if (self.uniform_epsilon is None) != (self.uniform_delta is None):
            raise ValueError("uniform epsilon and delta must be specified together")
        if self.trainable_mode not in {"full", "positive_head"}:
            raise ValueError("unknown public trainable subspace")
        if isinstance(self.head_settings, dict):
            object.__setattr__(self, "head_settings", PositiveHeadSettings(**self.head_settings))
        if not isinstance(self.head_settings, PositiveHeadSettings):
            raise ValueError("head_settings must be a public PositiveHeadSettings configuration")

    @property
    def count(self) -> int:
        return self.bins_per_round * self.rounds

    def bucket(self, stable_id: int, round_index: int) -> int:
        if isinstance(stable_id, bool) or not isinstance(stable_id, int):
            raise ValueError("stable node IDs must be integers")
        digest = hashlib.sha256(
            f"{self.salt}:{round_index}:{stable_id}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest, "big") % self.bins_per_round


def _public_seed(base: int, slot: int, step: int = -1) -> int:
    digest = hashlib.sha256(f"stable-slot-rng:{base}:{slot}:{step}".encode()).digest()
    return int.from_bytes(digest, "big")


def slot_members(graph: Any, policy: HashBudgetPolicy,
                 slots: StableSlots) -> List[List[int]]:
    """Fixed slot indices; only members of slots containing v change on deletion."""
    if len(slots.appearance_limits) != len(policy.groups):
        raise ValueError("one appearance limit is required per public budget group")
    if graph.vcount() == 0:
        return [[] for _ in range(slots.count)]
    if "stable_id" not in graph.vs.attributes():
        raise ValueError("nonempty input must carry stable integer node IDs")
    identifiers = list(graph.vs["stable_id"])
    if any(isinstance(x, bool) or not isinstance(x, int) for x in identifiers):
        raise ValueError("stable node IDs must be integers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("stable node IDs must be unique")
    members = [[] for _ in range(slots.count)]
    for index, identifier in enumerate(identifiers):
        limit = slots.appearance_limits[policy.group_for(identifier)]
        for round_index in range(limit):
            slot = round_index * slots.bins_per_round + slots.bucket(identifier, round_index)
            members[slot].append(index)
    return members


def slot_accounting(config: TrainingConfig, policy: HashBudgetPolicy,
                    slots: StableSlots) -> Tuple[int, Dict[str, Any]]:
    """Calibrate against every possible public group, never observed members."""
    if config.privacy.batch_size != slots.count:
        raise ValueError("first stable-slot mechanism uses all public slots, B=R*K")
    if len(slots.appearance_limits) != len(policy.groups):
        raise ValueError("appearance limits must match public budget groups")
    plans = []
    for group, limit in zip(policy.groups, slots.appearance_limits):
        epsilon = group.epsilon if slots.uniform_epsilon is None else slots.uniform_epsilon
        delta = group.delta if slots.uniform_delta is None else slots.uniform_delta
        if (Decimal(str(epsilon)) > Decimal(str(group.epsilon))
                or Decimal(str(delta)) > Decimal(str(group.delta))):
            raise ValueError("uniform guarantee must satisfy every external node budget")
        if limit == 0:
            plans.append(None)
        else:
            # RangePrivacy's 2*batch*R algebra is reused with batch=limit;
            # the actual public training denominator remains slots.count.
            plans.append(RangePrivacy(
                epsilon=str(epsilon), delta=str(delta),
                clip_norm=config.privacy.clip_norm, grid_bits=config.privacy.grid_bits,
                batch_size=limit, steps=config.privacy.steps,
            ))
    sigma = max((plan.calibrate() for plan in plans if plan is not None), default=1)
    descriptions = []
    for group, limit, plan in zip(policy.groups, slots.appearance_limits, plans):
        if plan is None:
            bound = {"epsilon_upper": "0", "delta_upper": "0",
                     "rho_numerator": 0, "rho_denominator": 1}
        else:
            certificate = plan.certificate(sigma)
            bound = {
                "epsilon_upper": certificate["epsilon_upper"],
                "delta_upper": str(plan.delta),
                "rho_numerator": certificate["rho_numerator"],
                "rho_denominator": certificate["rho_denominator"],
            }
        descriptions.append({
            "epsilon_target": str(group.epsilon), "delta_target": str(group.delta),
            "appearance_limit": limit,
            "sensitivity_sum_lattice": 2 * limit * config.privacy.radius,
            **bound,
        })
    from fractions import Fraction
    public_training = asdict(config)
    public_training.pop("privacy")
    public_training.update({
        "clip_norm": str(config.privacy.clip_norm),
        "grid_bits": config.privacy.grid_bits, "optimizer_steps": config.privacy.steps,
        "batch_size": config.privacy.batch_size,
    })
    public_slots = asdict(slots)
    public_slots["appearance_limits"] = list(slots.appearance_limits)
    return sigma, {
        "schema": "stable-slot-node-pdp-v1",
        "adjacency": "unbounded_node_add_remove_with_attributes_and_edges",
        "protection_start": "prepared_training_graph",
        "release": "fixed_architecture_model_parameters_only",
        "model": "GRAT3", "grat_variant": "legacy_released", "feature_protocol": "ones",
        "noise_distribution": "exact_discrete_gaussian_integer",
        "randomness_assumption": "secret_independent_uniform_system_random_bits",
        "sigma_lattice": sigma,
        "noise_scale_mean": str(Fraction(
            sigma, slots.count * (1 << config.privacy.grid_bits),
        )),
        "clip_radius_lattice": config.privacy.radius,
        "optimizer_steps": config.privacy.steps,
        "public_training_configuration": public_training,
        "public_slot_configuration": public_slots,
        "sampling_accounting": {
            "n": config.subgraph_size, "theta": None, "r": 3,
            "M_internal": config.frequency_threshold,
            "M_v_public_limits": list(slots.appearance_limits),
            "Ng": "not_used_in_stable_slot_accountant",
            "q_rule": "min(1,expected_starts/max(1,slot_nodes)); trusted_internal",
            "actual_slot_sizes_and_sampling_rates": "trusted_diagnostics_only",
        },
        **policy.public_description(),
        "groups": descriptions,
        "outer_node_sampling": False, "batch_privacy_amplification": False,
        "proof": "docs/pdp_stable_slots_design.md",
    }


def _canonical_slot_graph(graph: Any, vertices: List[int]) -> Any:
    """Normalize local node/edge order using stable IDs, without global features.

    The current ones/IC(p=1) query does not read original edge attributes.
    Vertex attributes are restricted and reordered with their stable labels.
    """
    local = graph.induced_subgraph(vertices)
    identifiers = list(local.vs["stable_id"]) if local.vcount() else []
    order = sorted(range(len(identifiers)), key=identifiers.__getitem__)
    remap = {old: new for new, old in enumerate(order)}
    edges = [(remap[a], remap[b]) for a, b in local.get_edgelist()]
    if not local.is_directed():
        edges = [(min(a, b), max(a, b)) for a, b in edges]
    edges.sort()
    canonical = igraph.Graph(n=local.vcount(), edges=edges, directed=local.is_directed())
    for attribute in local.vs.attributes():
        values = local.vs[attribute]
        canonical.vs[attribute] = [values[i] for i in order]
    canonical.vs["stable_id"] = [identifiers[i] for i in order]
    return canonical


def _prepare_slot_samples(graph: Any, members: List[List[int]],
                          config: TrainingConfig,
                          diagnostic: Dict[str, Any]) -> List[List[Any]]:
    containers = []
    diagnostic["preprocessing_failed_slots"] = []
    for index, vertices in enumerate(members):
        try:
            with _trusted_computation():
                local = _canonical_slot_graph(graph, vertices)
                local_config = replace(config, sampling_seed=_public_seed(config.sampling_seed, index))
                container = build_samples(local, local_config)
        except Exception:
            # Critical: one failed slot never clears or changes other slots.
            container = []
            diagnostic["preprocessing_failed_slots"].append(index)
        containers.append(container)
    diagnostic["subgraph_counts_by_slot"] = [len(x) for x in containers]
    return containers


def _slot_contribution(model: Any, samples: List[Any], config: TrainingConfig,
                       slots: StableSlots, index: int, step: int,
                       dimension: int) -> List[int]:
    rng = random.Random(_public_seed(config.batch_seed, index, step))
    selected = rng.sample(samples, min(len(samples), slots.microbatch_size))
    query = head_contribution if slots.trainable_mode == "positive_head" else _contribution
    vectors = [query(model, sample, config) for sample in selected]
    signal = sum_contributions(
        vectors, radius=config.privacy.radius,
        batch_size=slots.microbatch_size, dimension=dimension,
    )
    averaged = [
        (1 if value >= 0 else -1) * (abs(value) // slots.microbatch_size)
        for value in signal
    ]
    return project_integer_ball(averaged, config.privacy.radius)


def train_slots(graph: Any, config: TrainingConfig, policy: HashBudgetPolicy,
                slots: StableSlots, *, trusted_diagnostics: Optional[Dict[str, Any]] = None,
                noise_enabled: bool = True) -> ParameterRelease:
    """Train with public slot-local queries and per-node full-training costs.

    noise_enabled=False is explicitly a non-private public-benchmark control.
    It never receives a DP certificate.
    """
    if not isinstance(noise_enabled, bool):
        raise ValueError("noise_enabled must be boolean")
    if torch.get_num_threads() != 1:
        raise ValueError("stable-slot queries require a single-thread CPU environment")
    sigma, certificate = slot_accounting(config, policy, slots)
    model = (initialize_positive_head(config, slots.head_settings)
             if slots.trainable_mode == "positive_head" else initialize_model(config))
    model.train()
    dimension = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    diagnostic = trusted_diagnostics if trusted_diagnostics is not None else {}
    members = slot_members(graph, policy, slots)
    diagnostic["slot_node_counts"] = [len(x) for x in members]
    diagnostic["sampling_rate_by_slot"] = [
        min(1.0, config.expected_starts/max(1, len(vertices))) for vertices in members
    ]
    containers = _prepare_slot_samples(graph, members, config, diagnostic)
    diagnostic.update({"completed_steps": 0, "invalid_slot_queries": 0,
                       "nonzero_slot_queries": 0})
    noise = DiscreteGaussian(sigma)
    for step in range(config.privacy.steps):
        contributions = []
        for index, container in enumerate(containers):
            try:
                with _trusted_computation():
                    vector = _slot_contribution(
                        model, container, config, slots, index, step, dimension,
                    )
            except Exception:
                vector = [0] * dimension
                diagnostic["invalid_slot_queries"] += 1
            contributions.append(vector)
            diagnostic["nonzero_slot_queries"] += int(any(vector))
        signal = sum_contributions(
            contributions, radius=config.privacy.radius,
            batch_size=slots.count, dimension=dimension,
        )
        perturbation = noise.sample(dimension) if noise_enabled else [0] * dimension
        noisy = [a + z for a, z in zip(signal, perturbation)]
        if slots.trainable_mode == "positive_head":
            apply_head_update(model, noisy, config, slots.head_settings)
        else:
            _apply_noisy_integer_update(model, noisy, config)
        diagnostic["completed_steps"] += 1
    certificate["parameter_count"] = sum(p.numel() for p in model.parameters())
    certificate["private_parameter_count"] = dimension
    if slots.trainable_mode == "positive_head":
        certificate["proof"] = "docs/pdp_positive_head_design.md"
    if not noise_enabled:
        certificate = {
            "schema": "nonprivate-stable-slot-control-v1", "privacy_guarantee": "none",
            "baseline": "stable_slots_zero_noise",
            "public_training_configuration": certificate["public_training_configuration"],
            "public_slot_configuration": certificate["public_slot_configuration"],
        }
    return ParameterRelease(
        {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()},
        certificate,
    )
