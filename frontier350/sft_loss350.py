"""Opt-in, fully differentiable Arctus350 assistant-only SFT objective.

The default token-mean objective matches the existing training loss exactly.
Per-example aggregation prevents long assistant answers from automatically
receiving proportionally greater weight. This is a research hypothesis, not
a measured quality gain. This module changes no checkpoint or model weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOSS_AGGREGATIONS = ("token", "per-example")


def assistant_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eos_id: int,
    eos_weight: float,
    *,
    aggregation: str = "token",
) -> torch.Tensor:
    """Cross-entropy on assistant targets; pad/context labels must equal -100.

    * token: reproduce the historical weighted token mean, exactly.
    * per-example: weighted average per encoded conversation, then mean of
      conversations. Every row must have supervised tokens (fail closed).

    No source/model identity or native inference is inferred from unit tests.
    """
    if aggregation not in LOSS_AGGREGATIONS or not isinstance(aggregation, str):
        raise ValueError("unsupported assistant-loss aggregation")
    if (not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor)
        or logits.ndim != 3 or targets.ndim != 2
        or logits.shape[:2] != targets.shape
        or targets.dtype != torch.long
        or not isinstance(eos_id, int) or isinstance(eos_id, bool)
        or not 0 <= eos_id < logits.shape[-1]
        or isinstance(eos_weight, bool)
        or not isinstance(eos_weight, (float, int))
        or not math.isfinite(eos_weight) or eos_weight <= 0):
        raise ValueError("invalid assistant-loss logits, targets or EOS weight")

    flat_y = targets.reshape(-1)
    mask = flat_y != -100
    if not bool(mask.any()):
        raise ValueError("batch has no assistant targets")

    # Extract only assistant-token logits, identical to the legacy trainer.
    selected = logits.reshape(-1, logits.shape[-1])[mask].float()
    selected_targets = flat_y[mask]
    per = F.cross_entropy(selected, selected_targets, reduction="none")
    weights = torch.ones_like(per)
    weights[selected_targets == eos_id] = float(eos_weight)

    if aggregation == "token":
        # Preserve legacy numerical behavior. This must stay the default.
        return (per * weights).sum() / weights.sum().clamp_min(1.0)

    batch_size, width = targets.shape
    if not bool((targets != -100).any(dim=1).all()):
        raise ValueError("some examples have no assistant targets")

    # Map selected token losses to their original encoded conversations.
    # No Python item loops, no CPU copies, autograd remains attached.
    row_indices = (torch.arange(batch_size, device=targets.device)
                   .unsqueeze(1).expand(-1, width).reshape(-1)[mask])
    weighted_sums = torch.zeros(batch_size, dtype=per.dtype, device=per.device)
    weight_sums = torch.zeros_like(weighted_sums)
    weighted_sums.scatter_add_(0, row_indices, per * weights)
    weight_sums.scatter_add_(0, row_indices, weights)
    return (weighted_sums / weight_sums).mean()
