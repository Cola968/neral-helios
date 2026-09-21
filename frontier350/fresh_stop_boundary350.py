"""CPU-only preflight: prevent a non-terminal partial AdamW optimizer update.

An intentional mid-update stop produces a different optimizer trajectory on
continuation. Full target completion may contain a final partial update.
"""
from __future__ import annotations


def require_fresh_stop_boundary(*, target: int, stop: int, seq: int,
                                microbatch: int, grad_accum: int) -> int:
    fields = (target, stop, seq, microbatch, grad_accum)
    if any(type(value) is not int or value <= 0 for value in fields):
        raise ValueError('positive integer target, stop and update dimensions required')
    if stop > target:
        raise ValueError('stop cannot exceed target token budget')
    update = seq * microbatch * grad_accum
    if stop < target and stop % update:
        raise ValueError('intermediate stop splits optimizer update; exact resume forbidden')
    return update
