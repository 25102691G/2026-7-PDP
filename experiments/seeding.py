"""Explicit random-seed protocol for reproducible PrivIM experiments."""

from dataclasses import asdict, dataclass


SEED_MODES = ("fixed", "per-round")
SAMPLING_SEED_MODES = SEED_MODES


@dataclass(frozen=True)
class RoundSeeds:
    model: int
    sampling: int
    batch: int

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
) -> RoundSeeds:
    """Resolve independent seeds for one experimental round.

    Model initialization and batch selection always vary by round. Sampling
    stays fixed by default so repeated training runs share one subgraph
    container; ``per-round`` isolates sampling variability when requested.
    """
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    for name, mode in (
        ("model", model_seed_mode),
        ("sampling", sampling_seed_mode),
        ("batch", batch_seed_mode),
    ):
        if mode not in SEED_MODES:
            raise ValueError(f"unsupported {name} seed mode: {mode}")
    resolved = lambda base, mode: base if mode == "fixed" else base + round_index
    return RoundSeeds(
        model=resolved(model_seed, model_seed_mode),
        sampling=resolved(sampling_seed, sampling_seed_mode),
        batch=resolved(batch_seed, batch_seed_mode),
    )


def seed_protocol(
    *, model_seed: int, model_seed_mode: str, sampling_seed: int,
    batch_seed: int, batch_seed_mode: str, sampling_seed_mode: str,
):
    for name, mode in (
        ("model", model_seed_mode),
        ("sampling", sampling_seed_mode),
        ("batch", batch_seed_mode),
    ):
        if mode not in SEED_MODES:
            raise ValueError(f"unsupported {name} seed mode: {mode}")
    return {
        "model_seed_base": model_seed,
        "model_seed_mode": model_seed_mode,
        "sampling_seed_base": sampling_seed,
        "sampling_seed_mode": sampling_seed_mode,
        "batch_seed_base": batch_seed,
        "batch_seed_mode": batch_seed_mode,
    }
