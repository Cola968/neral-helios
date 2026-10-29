"""Deterministic sampling streams for native SFT; no checkpoint or model access."""
from __future__ import annotations
import random


def training_rngs(seed: int, mode: str = "legacy"):
    """Return (SFT batch RNG, replay RNG); legacy preserves the original stream."""
    if type(seed) is not int or not isinstance(mode, str) or mode not in ("legacy", "independent"):
        raise ValueError("invalid RNG stream configuration")
    sft_rng = random.Random(seed + 1)
    replay_rng = sft_rng if mode == "legacy" else random.Random(seed + 2)
    return sft_rng, replay_rng
