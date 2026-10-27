"""B0 training isolated from the legacy reproduction launcher and optimizer.

All data-dependent diagnostics are explicitly trusted-only. The release
contains only fixed-shape model parameters and a data-independent certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("DGLBACKEND", "pytorch")

import torch

from experiments.graph_util import get_adj_mat, get_rev_dgl
from experiments.sampling import sample_privim_star
from model.GRAT import GRAT3
from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss

from .accounting import RangePrivacy, positive_integer
from .discrete_noise import DiscreteGaussian
from .lattice import quantize_gradient, sum_contributions


@dataclass(frozen=True)
class TrainingConfig:
    privacy: RangePrivacy = field(default_factory=RangePrivacy)
    learning_rate: float = 0.005
    loss_penalty: float = 0.5
    input_dim: int = 32
    hidden_dim: int = 32
    subgraph_size: int = 40
    frequency_threshold: int = 4
    expected_starts: int = 256
    restart_probability: float = 0.3
    walk_length: int = 200
    frequency_decay: float = 1.0
    boundary_divisor: int = 2
    include_boundary: bool = True
    model_seed: int = 0
    sampling_seed: int = 0
    batch_seed: int = 0
    parameter_bound: float = 1e6
    noisy_integer_cap: int = 2**48

    def __post_init__(self) -> None:
        if not isinstance(self.privacy, RangePrivacy):
            raise ValueError("privacy must be a RangePrivacy configuration")
        for name in [
            "input_dim", "hidden_dim", "subgraph_size", "frequency_threshold",
            "expected_starts", "walk_length", "boundary_divisor", "noisy_integer_cap",
        ]:
            positive_integer(getattr(self, name), name)
        if self.noisy_integer_cap > 2**52:
            raise ValueError("noisy_integer_cap must not exceed 2**52")
        for name in ["learning_rate", "loss_penalty", "parameter_bound"]:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1e30:
                raise ValueError(f"{name} must be finite and in (0,1e30]")
        if (not math.isfinite(self.restart_probability)
                or not 0 <= self.restart_probability <= 1):
            raise ValueError("restart_probability must be in [0,1]")
        if not math.isfinite(self.frequency_decay) or self.frequency_decay < 0:
            raise ValueError("frequency_decay must be finite and nonnegative")
        for name in ["model_seed", "sampling_seed", "batch_seed"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.include_boundary, bool):
            raise ValueError("include_boundary must be boolean")


@dataclass(frozen=True)
class ParameterRelease:
    parameters: Dict[str, torch.Tensor]
    certificate: Dict[str, Any]


@contextmanager
def _trusted_computation():
    """Library warnings/prints about private graph structure stay internal."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield


def initialize_model(config: TrainingConfig) -> GRAT3:
    """Public fixed initialization, isolated from unrelated global RNG use."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.model_seed)
        model = GRAT3(
            config.input_dim, config.hidden_dim, config.hidden_dim,
            variant="legacy_released", initialization="released_default",
        ).cpu()
    if list(model.named_buffers()):
        raise ValueError("B0 does not release unaccounted model buffers")
    return model


def build_samples(graph: Any, config: TrainingConfig) -> List[Tuple[Any, Any]]:
    """One persistent, trusted-only SCS/BES container on the input graph."""
    sampling_graph = graph.copy()
    sampling_graph.simplify(multiple=True, loops=True)
    result = sample_privim_star(
        sampling_graph,
        size=config.subgraph_size,
        boundary_divisor=config.boundary_divisor,
        sample_rate=min(1.0, config.expected_starts / max(1, sampling_graph.vcount())),
        restart_probability=config.restart_probability,
        walk_length=config.walk_length,
        threshold=config.frequency_threshold,
        decay=config.frequency_decay,
        include_boundary_stage=config.include_boundary,
        rng=random.Random(config.sampling_seed),
    )
    samples = []
    for nodes in result.node_sets:
        subgraph = sampling_graph.induced_subgraph(nodes)
        message_graph = get_rev_dgl(
            subgraph, "ones", config.input_dim, subgraph.is_directed(), False,
        )
        samples.append((message_graph, get_adj_mat(subgraph)))
    return samples


def _gradient_values(
    model: GRAT3, sample: Tuple[Any, Any], config: TrainingConfig,
) -> List[float]:
    graph, adjacency = sample
    parameters = list(model.parameters())
    probabilities = model(graph, graph.ndata["feat"]).reshape(-1)
    loss = KSetMaxCoverAdjLoss(config.loss_penalty)(probabilities, adjacency)
    if not bool(torch.isfinite(loss)):
        return [0] * sum(parameter.numel() for parameter in parameters)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            flattened.extend([0.0] * parameter.numel())
        else:
            flattened.extend(gradient.detach().to(torch.float64).reshape(-1).tolist())
    return flattened


def _contribution(
    model: GRAT3, sample: Tuple[Any, Any], config: TrainingConfig,
) -> List[int]:
    return quantize_gradient(
        _gradient_values(model, sample, config), radius=config.privacy.radius,
        grid_bits=config.privacy.grid_bits,
    )


def _apply_noisy_integer_update(
    model: GRAT3, noisy_sum: List[int], config: TrainingConfig,
) -> None:
    """Only postprocessing: clamp integers, convert, average and update."""
    cap = config.noisy_integer_cap
    bounded = [max(-cap, min(cap, value)) for value in noisy_sum]
    update = torch.tensor(bounded, dtype=torch.float64)
    update.mul_(math.ldexp(1.0, -config.privacy.grid_bits) / config.privacy.batch_size)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            direction = update[offset:offset+count].reshape(parameter.shape)
            parameter.add_(direction.to(parameter.dtype), alpha=-config.learning_rate)
            parameter.nan_to_num_(
                nan=0.0, posinf=config.parameter_bound, neginf=-config.parameter_bound,
            )
            parameter.clamp_(-config.parameter_bound, config.parameter_bound)
            offset += count
    if offset != len(noisy_sum):
        raise ValueError("noisy vector must have the fixed public model dimension")


def train_b0(
    graph: Any, config: TrainingConfig, *,
    trusted_diagnostics: Optional[Dict[str, Any]] = None,
) -> ParameterRelease:
    """Execute exactly T privatized updates, including empty/failed containers.

    trusted_diagnostics, when explicitly supplied, is raw trusted-environment
    material. It is never attached to ParameterRelease or save_release.
    The graph and all its derived structures must remain inside this boundary.
    """
    sigma = config.privacy.calibrate()
    certificate = config.privacy.certificate(sigma)
    model = initialize_model(config)
    model.train()
    dimension = sum(parameter.numel() for parameter in model.parameters())
    diagnostics = trusted_diagnostics if trusted_diagnostics is not None else {}
    diagnostics.update({
        "preprocessing_failed": False, "invalid_contributions": 0,
        "nonzero_contributions": 0, "completed_steps": 0,
    })
    try:
        with _trusted_computation():
            samples = build_samples(graph, config)
    except Exception as error:
        samples = []
        diagnostics["preprocessing_failed"] = True
        diagnostics["preprocessing_error_type"] = type(error).__name__
    diagnostics["subgraph_count"] = len(samples)
    batch_rng = random.Random(config.batch_seed)
    noise = DiscreteGaussian(sigma)
    for _ in range(config.privacy.steps):
        chosen = batch_rng.sample(samples, min(len(samples), config.privacy.batch_size))
        contributions = []
        for sample in chosen:
            try:
                with _trusted_computation():
                    vector = _contribution(model, sample, config)
            except Exception as error:
                vector = [0] * dimension
                diagnostics["invalid_contributions"] += 1
                diagnostics["last_contribution_error_type"] = type(error).__name__
            diagnostics["nonzero_contributions"] += int(any(vector))
            contributions.append(vector)
        signal = sum_contributions(
            contributions, radius=config.privacy.radius,
            batch_size=config.privacy.batch_size, dimension=dimension,
        )
        # Exact Python integer addition precedes every floating operation.
        perturbation = noise.sample(dimension)
        noisy_sum = [left + right for left, right in zip(signal, perturbation)]
        _apply_noisy_integer_update(model, noisy_sum, config)
        diagnostics["completed_steps"] += 1
    certificate.update({
        "model": "GRAT3",
        "grat_variant": "legacy_released",
        "feature_protocol": "ones",
        "public_training_configuration": asdict(config),
        "parameter_count": dimension,
        "sampling_accounting": {
            "n": config.subgraph_size, "theta": None, "r": 3,
            "M": config.frequency_threshold, "Ng": "not_used_in_range_accountant",
            "q_rule": "min(1,expected_starts/max(1,input_nodes)); trusted_internal",
        },
    })
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    return ParameterRelease(state, certificate)


def save_release(release: ParameterRelease, directory: Path) -> None:
    """Create a new public release directory; never overwrite existing output."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(release.parameters, directory / "parameters.pt")
    (directory / "certificate.json").write_text(
        json.dumps(release.certificate, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
