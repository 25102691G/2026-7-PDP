"""Explicitly non-private controls sharing B0 architecture, samples and loss."""

from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any, Dict, Optional

import torch

from .lattice import quantize_gradient, sum_contributions
from .training import (
    ParameterRelease, TrainingConfig, _apply_noisy_integer_update,
    _gradient_values, _trusted_computation, build_samples, initialize_model,
)


def train_non_private(
    graph: Any, config: TrainingConfig, *, clipped: bool = False,
    initialization_only: bool = False,
    trusted_diagnostics: Optional[Dict[str, Any]] = None,
) -> ParameterRelease:
    model = initialize_model(config)
    dimension = sum(parameter.numel() for parameter in model.parameters())
    diagnostic = trusted_diagnostics if trusted_diagnostics is not None else {}
    diagnostic.update({"completed_steps": 0, "invalid_contributions": 0})
    if initialization_only:
        samples = []
    else:
        with _trusted_computation():
            samples = build_samples(graph, config)
    diagnostic["subgraph_count"] = len(samples)
    batch_rng = random.Random(config.batch_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for _ in range(0 if initialization_only else config.privacy.steps):
        chosen = batch_rng.sample(samples, min(len(samples), config.privacy.batch_size))
        vectors = []
        for sample in chosen:
            try:
                with _trusted_computation():
                    vector = _gradient_values(model, sample, config)
            except Exception:
                diagnostic["invalid_contributions"] += 1
                vector = [0.0]*dimension
            vectors.append(vector)
        if clipped:
            signal = sum_contributions(
                [quantize_gradient(v, radius=config.privacy.radius,
                                   grid_bits=config.privacy.grid_bits) for v in vectors],
                radius=config.privacy.radius, batch_size=config.privacy.batch_size,
                dimension=dimension,
            )
            _apply_noisy_integer_update(model, signal, config)
        else:
            # Ordinary batch-mean SGD, matching the legacy Non-Private
            # interpretation; no gradient clipping or perturbation.
            values = torch.zeros(dimension, dtype=torch.float64)
            for vector in vectors:
                values.add_(torch.tensor(vector, dtype=torch.float64))
            values.div_(max(1, len(vectors)))
            optimizer.zero_grad()
            offset = 0
            for parameter in model.parameters():
                count = parameter.numel()
                parameter.grad = values[offset:offset+count].reshape(
                    parameter.shape
                ).to(parameter.dtype)
                offset += count
            optimizer.step()
        diagnostic["completed_steps"] += 1
    configuration = asdict(config)
    configuration.pop("privacy")
    configuration["steps"] = 0 if initialization_only else config.privacy.steps
    configuration["batch_size"] = config.privacy.batch_size
    configuration["clip_norm"] = config.privacy.clip_norm if clipped else None
    configuration["grid_bits"] = config.privacy.grid_bits if clipped else None
    return ParameterRelease(
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        {
            "schema": "nonprivate-control-v1",
            "privacy_guarantee": "none" if not initialization_only else "data_independent",
            "model": "GRAT3", "grat_variant": "legacy_released",
            "feature_protocol": "ones",
            "baseline": (
                "public_initialization" if initialization_only else
                "clipped_no_noise" if clipped else "non_private"
            ),
            "public_training_configuration": configuration,
        },
    )
