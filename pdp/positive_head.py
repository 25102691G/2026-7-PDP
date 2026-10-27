"""Public fixed GRAT encoder and a privatizable bounded readout subspace.

No graph is read during initialization. See docs/pdp_positive_head_design.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List

import torch

from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
from .lattice import quantize_gradient
from .training import ParameterRelease, TrainingConfig, initialize_model


@dataclass(frozen=True)
class PositiveHeadSettings:
    encoder_average_row_sum: float = 0.1
    encoder_bias_offset: float = 0.1
    initial_weight_sum: float = 0.1
    initial_bias: float = -3.0
    weight_lower: float = 0.0001
    weight_upper: float = 0.03
    bias_lower: float = -8.0
    bias_upper: float = -1.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("head settings must be finite public constants")
        if (self.encoder_average_row_sum <= 0 or self.encoder_bias_offset <= 0
                or self.initial_weight_sum <= 0
                or not 0 < self.weight_lower < self.weight_upper
                or not self.bias_lower <= self.initial_bias <= self.bias_upper < 0):
            raise ValueError("invalid public encoder or readout constraints")


def initialize_positive_head(config: TrainingConfig, settings: PositiveHeadSettings):
    model = initialize_model(config)
    with torch.no_grad():
        layers = [model.grat.grat1, model.grat.grat2, model.grat.grat3]
        for layer in layers:
            layer.attn_fc.weight.zero_()
            layer.attn_fc.bias.fill_(1)
        for layer in layers[:2]:
            layer.fc.weight.abs_()
            average = layer.fc.weight.sum(dim=1).mean()
            if average > 0:
                layer.fc.weight.mul_(settings.encoder_average_row_sum/average)
            else:
                layer.fc.weight.fill_(settings.encoder_average_row_sum/layer.fc.in_features)
            layer.fc.bias.abs_().add_(settings.encoder_bias_offset)
        head = model.grat.grat3.fc
        head.weight.abs_()
        total = head.weight.sum()
        if total > 0:
            head.weight.mul_(settings.initial_weight_sum/total)
        else:
            head.weight.fill_(settings.initial_weight_sum/head.in_features)
        head.weight.clamp_(settings.weight_lower, settings.weight_upper)
        head.bias.fill_(settings.initial_bias)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    return model


def head_contribution(model: Any, sample: Any, config: TrainingConfig) -> List[int]:
    graph, adjacency = sample
    parameters = [p for p in model.parameters() if p.requires_grad]
    dimension = sum(p.numel() for p in parameters)
    predictions = model(graph, graph.ndata["feat"]).reshape(-1)
    loss = KSetMaxCoverAdjLoss(config.loss_penalty)(predictions, adjacency)
    if not bool(torch.isfinite(loss)):
        return [0]*dimension
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    values = []
    for parameter, gradient in zip(parameters, gradients):
        values.extend(
            [0.0]*parameter.numel() if gradient is None else
            gradient.detach().double().reshape(-1).tolist()
        )
    return quantize_gradient(values, radius=config.privacy.radius,
                             grid_bits=config.privacy.grid_bits)


def apply_head_update(model: Any, noisy_sum: List[int], config: TrainingConfig,
                      settings: PositiveHeadSettings) -> None:
    """Postprocessing only: no raw graph, gradients or data-dependent limits."""
    bounded = [max(-config.noisy_integer_cap, min(config.noisy_integer_cap, value))
               for value in noisy_sum]
    update = torch.tensor(bounded, dtype=torch.float64)
    update.mul_(math.ldexp(1.0, -config.privacy.grid_bits)/config.privacy.batch_size)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            count = parameter.numel()
            direction = update[offset:offset+count].reshape(parameter.shape)
            parameter.add_(direction.to(parameter.dtype), alpha=-config.learning_rate)
            offset += count
        if offset != len(noisy_sum):
            raise ValueError("noise dimension must equal the public readout dimension")
        head = model.grat.grat3.fc
        head.weight.nan_to_num_(nan=settings.weight_lower,
                               posinf=settings.weight_upper, neginf=settings.weight_lower)
        head.weight.clamp_(settings.weight_lower, settings.weight_upper)
        head.bias.nan_to_num_(nan=settings.initial_bias,
                             posinf=settings.bias_upper, neginf=settings.bias_lower)
        head.bias.clamp_(settings.bias_lower, settings.bias_upper)


def positive_initialization_release(config: TrainingConfig,
                                    settings: PositiveHeadSettings) -> ParameterRelease:
    """A matched data-independent control, with no graph argument at all."""
    model = initialize_positive_head(config, settings)
    return ParameterRelease(
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        {
            "schema": "public-positive-head-initialization-v1",
            "privacy_guarantee": "data_independent",
            "model": "GRAT3", "grat_variant": "legacy_released",
            "feature_protocol": "ones", "optimizer_steps": 0,
            "model_seed": config.model_seed,
            "input_dim": config.input_dim, "hidden_dim": config.hidden_dim,
            "positive_head_configuration": asdict(settings),
        },
    )
