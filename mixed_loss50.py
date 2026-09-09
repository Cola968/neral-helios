from __future__ import annotations

"""Single-forward Chat + LM replay objective for Neral-50M.

For equal sequence widths, independent rows do not interact in the Transformer. Concatenating chat
and replay rows along the batch dimension therefore represents the same two objectives while
amortizing embedding, block and LM-head kernel launches into one model forward. Dropout is zero in
Neral-50M, so there is no batch-coupled stochastic layer to invalidate the construction.

This helper is experimental until the RTX probe demonstrates both numerical agreement and an
end-to-end throughput win. It does not change model weights, data or objective coefficients.
"""

import torch
import torch.nn.functional as F

from tokenizer import ByteTokenizer

TOKENIZER = ByteTokenizer()


def _chat_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eos_weight: float) -> torch.Tensor:
    flat_logits = logits.float().reshape(-1, logits.size(-1))
    flat_y = targets.reshape(-1)
    per = F.cross_entropy(flat_logits, flat_y, ignore_index=-100, reduction="none")
    valid = flat_y != -100
    weights = torch.ones_like(per)
    weights[(flat_y == TOKENIZER.eos_id) & valid] = eos_weight
    return (per[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)


def mixed_chat_replay_loss(
    model,
    chat_x: torch.Tensor,
    chat_y: torch.Tensor,
    replay_x: torch.Tensor,
    replay_y: torch.Tensor,
    *,
    eos_weight: float,
    replay_scale: float,
):
    if chat_x.ndim != 2 or replay_x.ndim != 2:
        raise ValueError("mixed batches must be rank-2")
    if chat_x.shape != chat_y.shape or replay_x.shape != replay_y.shape:
        raise ValueError("input/target shapes must match")
    if chat_x.size(1) != replay_x.size(1):
        raise ValueError("single-forward fusion requires equal sequence widths")
    chat_rows = chat_x.size(0)
    x = torch.cat((chat_x, replay_x), dim=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    chat_logits = logits[:chat_rows]
    replay_logits = logits[chat_rows:]
    chat_loss = _chat_loss_from_logits(chat_logits, chat_y, eos_weight)
    replay_loss = F.cross_entropy(
        replay_logits.float().reshape(-1, replay_logits.size(-1)),
        replay_y.reshape(-1),
    )
    total = chat_loss + float(replay_scale) * replay_loss
    return total, chat_loss, replay_loss
