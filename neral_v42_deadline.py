"""Request-scoped cooperative deadlines for NERAL's Windows single-GPU service.

No background threads, signal handlers or CUDA allocations. Model inference is
stopped between generated tokens, and evidence stages must check this clock.
An external OS call that cannot be interrupted may exceed the deadline; the
runner must independently poll the service busy lock before the next task.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import time


MODE_BUDGETS = {"Amber": 25, "Quartz": 40, "Obsidian": 70}
ALIASES = {"Luna": "Amber", "Terra": "Quartz", "Helios": "Obsidian"}
_REQUEST_DEADLINE = ContextVar("neral_v42_request_deadline", default=None)


class DeadlineExceeded(TimeoutError):
    """Cooperative inference cutoff; no prompts, model outputs or secrets."""


def canonical_tier(tier):
    tier = ALIASES.get(tier, tier)
    if tier not in MODE_BUDGETS:
        raise ValueError("Invalid NERAL time budget tier")
    return tier


@contextmanager
def request_deadline(tier):
    seconds = MODE_BUDGETS[canonical_tier(tier)]
    token = _REQUEST_DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def remaining_seconds():
    deadline = _REQUEST_DEADLINE.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def should_stop_generation(*_args, **_kwargs):
    """Callable as a Transformers stopping criterion between decoding steps."""
    left = remaining_seconds()
    return left is not None and left <= 0.0


def check_deadline():
    if should_stop_generation():
        raise DeadlineExceeded("neral_request_deadline")
