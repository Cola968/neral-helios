"""Fail-closed precision contract for experimental Neral warm-start training.

Historical source weights may be BF16; upcasting cannot recover lost information.
NEW trainable parameters and NEW AdamW moments must remain FP32. BF16 is for
CUDA autocast compute, never the serialized master parameters. No training here.
"""
from __future__ import annotations


def require_fp32_master(model, optimizer, torch, *, require_moments: bool) -> None:
    params = list(model.parameters())
    if not params or any(p.dtype != torch.float32 for p in params):
        raise RuntimeError('FP32 master parameters required: BF16 optimizer would quantize small updates')
    owned = {id(p) for p in params}
    configured = [p for group in optimizer.param_groups for p in group['params']]
    if len(configured) != len(owned) or {id(p) for p in configured} != owned:
        raise RuntimeError('AdamW must own every master parameter exactly once')
    state = optimizer.state
    if not require_moments:
        if state:
            raise RuntimeError('fresh AdamW must start with empty moments')
        return
    if len(state) != len(params):
        raise RuntimeError('AdamW moments missing for at least one master parameter')
    for param in params:
        slot = state.get(param)
        if not isinstance(slot, dict):
            raise RuntimeError('AdamW moment slot absent')
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = slot.get(name)
            if not torch.is_tensor(moment) or moment.dtype != torch.float32 or moment.shape != param.shape:
                raise RuntimeError('AdamW moments must be FP32 with matching shapes')
