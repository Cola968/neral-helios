"""Pure CPU helpers for the bounded experimental SFT validation protocol.

These helpers do not load checkpoints, authorize data, or establish model quality.
A fixed, spread-out subset is less order-sensitive than the first N rows, but it
is NOT a statistically representative or blind benchmark.
"""
from __future__ import annotations

import math


def validation_indices(total: int, limit: int = 16) -> tuple[int, ...]:
    """Return reproducible, unique indices spanning the entire validation split."""
    if type(total) is not int or type(limit) is not int or total <= 0 or limit < 2:
        raise ValueError("positive validation population and limit >= 2 required")
    if total <= limit:
        return tuple(range(total))
    # Endpoints included, no RNG state and no run-dependent sampling.
    return tuple(index * (total - 1) // (limit - 1) for index in range(limit))


def token_weighted_loss(losses_and_counts: list[tuple[float, int]]) -> float:
    """Aggregate CE per supervised token, not the unweighted mean of row losses."""
    if not losses_and_counts:
        raise ValueError("empty validation sample")
    total_loss = 0.0
    total_tokens = 0
    for loss, count in losses_and_counts:
        if type(count) is not int or count <= 0 or not math.isfinite(loss):
            raise ValueError("nonfinite loss or invalid supervised token count")
        total_loss += loss * count
        total_tokens += count
    result = total_loss / total_tokens
    if not math.isfinite(result):
        raise ValueError("nonfinite aggregated validation loss")
    return result


def dynamic_batch_width(rows: list[dict], maximum: int) -> int:
    """Use longest selected example, never the pack-wide context for short batches."""
    if not rows or type(maximum) is not int or maximum <= 0:
        raise ValueError("empty batch or invalid context")
    widths = [len(row["input_ids"]) for row in rows]
    if min(widths) < 2 or max(widths) > maximum:
        raise ValueError("batch sequence exceeds packed context")
    return max(widths)
