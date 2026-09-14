from __future__ import annotations

"""Pure policy helpers for Scale V7.

V7 keeps V6's mathematical objective and SAFE gates.  These helpers only govern
execution-profile compatibility, GPU headroom and Fast2 admission.  Keeping them pure
makes the safety policy unit-testable without a CUDA device.
"""

from copy import deepcopy

EXECUTION_ONLY_CONTRACT_KEYS = frozenset({
    "compile_mode",
    "fast_replay_only",
    "fused_replay_main",
})


def semantic_resume_contract(contract: dict) -> dict:
    """Return the part of a trainer contract that must remain identical on resume.

    A compile-profile fallback must not invalidate an otherwise exact optimizer/RNG/data
    resume.  Data identity, model implementation, LR/replay recipe, token budget, pack
    shape and trainable parameter identities remain part of the contract.
    """

    out = deepcopy(contract)
    for key in EXECUTION_ONLY_CONTRACT_KEYS:
        out.pop(key, None)
    return out


def gpu_guard_reason(
    *,
    reserved_mib: float,
    free_mib: float,
    temp_c: float | None,
    max_reserved_mib: float = 9216.0,
    min_free_mib: float = 1280.0,
    max_temp_c: float = 82.0,
) -> str | None:
    """Return a controlled-restart reason, or None when headroom is healthy."""

    if reserved_mib > max_reserved_mib:
        return "reserved_vram_ceiling"
    if free_mib < min_free_mib:
        return "free_vram_floor"
    if temp_c is not None and temp_c >= max_temp_c:
        return "gpu_temperature_ceiling"
    return None


def fast2_gate(
    report: dict,
    *,
    min_ratio: float = 1.06,
    max_loss_diff: float = 2e-4,
    max_peak_reserved_gib: float = 9.0,
) -> tuple[bool, list[str]]:
    """Quality/speed/memory admission gate for the V6 Fast2 kernel path."""

    failures: list[str] = []
    ratio = float(report.get("fast2_ratio", 0.0))
    loss_diff = float(report.get("loss_abs_diff", float("inf")))
    fast2 = report.get("fast2") or {}
    peak_reserved = float(fast2.get("peak_reserved_gib", float("inf")))
    if ratio < min_ratio:
        failures.append("speed_ratio")
    if loss_diff > max_loss_diff:
        failures.append("loss_equivalence")
    if peak_reserved > max_peak_reserved_gib:
        failures.append("peak_reserved_vram")
    return not failures, failures


def recovery_targets(total_tokens: int, interval_tokens: int) -> list[int]:
    """Intermediate recovery targets; endpoint is intentionally excluded."""

    if total_tokens <= 0 or interval_tokens <= 0:
        raise ValueError("positive token budgets required")
    return list(range(interval_tokens, total_tokens, interval_tokens))
