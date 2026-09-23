"""Explicit random-seed protocol for reproducible PrivIM experiments."""

from dataclasses import asdict, dataclass


SEED_MODES = ("fixed", "per-round")
SAMPLING_SEED_MODES = SEED_MODES + ("continuous",)


@dataclass(frozen=True)
class RoundSeeds:
    model: int
    sampling: int
    batch: int
    dp_noise: int

    def as_dict(self):
        return asdict(self)


def resolve_round_seeds(
    *,
    round_index: int,
    model_seed: int,
    model_seed_mode: str,
    sampling_seed: int,
    batch_seed: int,
    batch_seed_mode: str,
    sampling_seed_mode: str,
    dp_noise_seed: int,
    dp_noise_seed_mode: str,
) -> RoundSeeds:
    """Resolve independent seeds for one experimental round.

    Each source follows its explicit mode. Sampling stays fixed by default so
    repeated training runs share one subgraph container, while model, batch,
    and DP-noise streams vary by default. Holding three streams fixed and
    varying the fourth supports controlled variance attribution.
    """
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    for name, mode, supported in (
        ("model", model_seed_mode, SEED_MODES),
        ("sampling", sampling_seed_mode, SAMPLING_SEED_MODES),
        ("batch", batch_seed_mode, SEED_MODES),
        ("dp_noise", dp_noise_seed_mode, SEED_MODES),
    ):
        if mode not in supported:
            raise ValueError(f"unsupported {name} seed mode: {mode}")
    resolved = lambda base, mode: (
        base if mode in {"fixed", "continuous"} else base + round_index
    )
    return RoundSeeds(
        model=resolved(model_seed, model_seed_mode),
        sampling=resolved(sampling_seed, sampling_seed_mode),
        batch=resolved(batch_seed, batch_seed_mode),
        dp_noise=resolved(dp_noise_seed, dp_noise_seed_mode),
    )


def seed_protocol(
    *, model_seed: int, model_seed_mode: str, sampling_seed: int,
    batch_seed: int, batch_seed_mode: str, sampling_seed_mode: str,
    dp_noise_seed: int, dp_noise_seed_mode: str,
):
    for name, mode, supported in (
        ("model", model_seed_mode, SEED_MODES),
        ("sampling", sampling_seed_mode, SAMPLING_SEED_MODES),
        ("batch", batch_seed_mode, SEED_MODES),
        ("dp_noise", dp_noise_seed_mode, SEED_MODES),
    ):
        if mode not in supported:
            raise ValueError(f"unsupported {name} seed mode: {mode}")
    return {
        "model_seed_base": model_seed,
        "model_seed_mode": model_seed_mode,
        "sampling_seed_base": sampling_seed,
        "sampling_seed_mode": sampling_seed_mode,
        "batch_seed_base": batch_seed,
        "batch_seed_mode": batch_seed_mode,
        "dp_noise_seed_base": dp_noise_seed,
        "dp_noise_seed_mode": dp_noise_seed_mode,
    }
