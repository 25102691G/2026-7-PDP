"""Paper constants and indicator-based parameter selection."""

import math


DATASETS = ("email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla")
EPSILONS = (1, 2, 3, 4, 5, 6)

STRICT_PROTOCOL = "strict"
RELEASED_COMPATIBILITY_PROTOCOL = "released-compatibility"
PROTOCOL_PROFILES = (STRICT_PROTOCOL, RELEASED_COMPATIBILITY_PROTOCOL)


def released_compatibility_overrides():
    """Return audited launcher defaults from released revision ``31166ca``.

    These values intentionally reproduce the released implementation conflicts;
    they are not the paper-faithful defaults.  Keeping them in a named profile
    prevents an under-accounted compatibility run from being confused with the
    strict unbounded node-DP protocol. The current tested Algorithms 1/3
    samplers and prepared data remain in use.
    """
    return {
        "iterations": 20,
        "learning_rate": 0.005,
        "lr_decay": 0.99,
        "lr_decay_steps": 5,
        "grat_variant": "legacy_released",
        "grat_initialization": "released_default",
        "feature_protocol": "ones",
        "privacy_accountant": "fixed_batch_hypergeometric",
        "clip_norm": 3.0,
        "model_seed": 0,
        "model_seed_mode": "fixed",
        "sampling_seed": 0,
        "sampling_seed_mode": "continuous",
        "batch_seed": 0,
        "batch_seed_mode": "fixed",
        "dp_noise_seed": 0,
        "dp_noise_seed_mode": "fixed",
    }


def select_indicator_parameters(dataset: str, vertex_count: int):
    """Select the nearest grid values to the Gamma-density modes (Appendix H)."""
    if vertex_count <= 1:
        raise ValueError("vertex_count must exceed one")
    log_size = math.log(vertex_count)
    beta_n = 0.47 * log_size - 1.03
    beta_m = 4.02 / log_size + 1.22
    n_mode = max(0.0, (beta_n - 1.0) * 25.0)
    m_mode = max(0.0, (beta_m - 1.0) * 5.0)
    n_grid = (10, 20, 40, 60, 80)
    m_grid = (4, 6, 8, 10, 12) if dataset == "email" else (2, 4, 6, 8, 10)
    nearest = lambda value, grid: min(grid, key=lambda candidate: abs(candidate - value))
    return nearest(n_mode, n_grid), nearest(m_mode, m_grid)
