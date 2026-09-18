"""Paper constants and indicator-based parameter selection."""

import math


DATASETS = ("email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla")
EPSILONS = (1, 2, 3, 4, 5, 6)


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
