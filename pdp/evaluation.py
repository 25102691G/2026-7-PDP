"""Fixed public-target IC(p=1,hop=1,k=50) coverage, reusing original CELF."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from baselines.heuristics import get_influence, greedy
from experiments.graph_util import get_rev_dgl
from .training import ParameterRelease, TrainingConfig, initialize_model


def celf_denominator(public_graph: Any, k: int = 50) -> int:
    if not 0 < k <= public_graph.vcount():
        raise ValueError("k must fit the public evaluation graph")
    _, coverage = greedy(public_graph, k)
    if coverage <= 0:
        raise ValueError("CELF coverage must be positive")
    return int(coverage)


def evaluate_release(
    release: ParameterRelease, config: TrainingConfig, public_graph: Any,
    *, k: int = 50, denominator: Optional[int] = None,
) -> Dict[str, Any]:
    """Auxiliary graph must be public/fixed, never the protected private graph."""
    if not 0 < k <= public_graph.vcount():
        raise ValueError("k must fit the public evaluation graph")
    denominator = celf_denominator(public_graph, k) if denominator is None else denominator
    if denominator <= 0:
        raise ValueError("CELF denominator must be positive")
    model = initialize_model(config)
    model.load_state_dict(release.parameters)
    model.eval()
    message_graph = get_rev_dgl(
        public_graph, "ones", config.input_dim, public_graph.is_directed(), False,
    )
    with torch.no_grad():
        scores = model.logits(message_graph, message_graph.ndata["feat"]).reshape(-1)
    finite_fraction = float(torch.isfinite(scores).float().mean())
    scores = torch.nan_to_num(
        scores, nan=-torch.inf, neginf=-torch.inf, posinf=torch.inf,
    )
    selected = torch.argsort(scores, descending=True, stable=True)[:k]
    spread = int(get_influence(public_graph, selected))
    return {
        "diffusion_model": "IC", "edge_probability": 1,
        "diffusion_hops": 1, "k": k, "spread": spread,
        "celf_denominator": denominator,
        "coverage_ratio_percent": 100*spread/denominator,
        "ranking_finite_fraction": finite_fraction,
        "seed_local_ids": selected.tolist(),
    }
