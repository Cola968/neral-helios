"""Independent-dev Pareto guard for future Neral-50M repair stages.

This module is deliberately benchmark-blind. It operates only on held-out development
metrics produced during training. A candidate is selectable only when every tracked
preservation metric is no worse than its source and there is a real math-generalization
signal. The frozen release bundle remains post-training evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    direction: str  # "higher" or "lower"


METRICS = (
    MetricSpec("id_exact_pct", "higher"),
    MetricSpec("transfer_exact_pct", "higher"),
    MetricSpec("id_nll", "lower"),
    MetricSpec("transfer_nll", "lower"),
    MetricSpec("sft_macro_nll", "lower"),
    MetricSpec("sft_math_nll", "lower"),
    MetricSpec("sft_code_nll", "lower"),
    MetricSpec("sft_reasoning_nll", "lower"),
    MetricSpec("sft_de_nll", "lower"),
    MetricSpec("lm_macro", "lower"),
    MetricSpec("free_eos_pct", "higher"),
    MetricSpec("free_repetition_pct", "lower"),
    MetricSpec("free_health_score", "higher"),
    MetricSpec("context_dev_pct", "higher"),
)


def _require(report: dict, key: str) -> float:
    if key not in report:
        raise KeyError(f"missing Pareto dev metric: {key}")
    value = float(report[key])
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite Pareto dev metric: {key}={value}")
    return value


def not_worse(before: float, after: float, direction: str, atol: float = 1e-6) -> bool:
    if direction == "higher":
        return after + atol >= before
    if direction == "lower":
        return after <= before + atol
    raise ValueError(direction)


def improvement(before: float, after: float, direction: str) -> float:
    """Signed improvement in natural metric units; positive is always better."""
    return after - before if direction == "higher" else before - after


def assess_dev(before: dict, after: dict, atol: float = 1e-6) -> dict:
    checks: dict[str, bool] = {}
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for spec in METRICS:
        b = _require(before, spec.key)
        a = _require(after, spec.key)
        ok = not_worse(b, a, spec.direction, atol=atol)
        checks[spec.key] = ok
        deltas[spec.key] = improvement(b, a, spec.direction)
        if not ok:
            regressions.append(spec.key)

    b_id = _require(before, "id_exact_pct")
    a_id = _require(after, "id_exact_pct")
    b_tr = _require(before, "transfer_exact_pct")
    a_tr = _require(after, "transfer_exact_pct")
    b_id_nll = _require(before, "id_nll")
    a_id_nll = _require(after, "id_nll")
    b_tr_nll = _require(before, "transfer_nll")
    a_tr_nll = _require(after, "transfer_nll")

    exact_growth = (a_id > b_id + atol) or (a_tr > b_tr + atol)
    soft_math_growth = (
        a_id_nll <= b_id_nll * 0.995 + atol
        and a_tr_nll <= b_tr_nll * 0.995 + atol
    )
    math_growth = exact_growth or soft_math_growth

    # Score is used only to rank already Pareto-eligible candidates. Exact-match gains
    # dominate NLL shaping; preservation metrics never compensate for a regression.
    score = (
        4.0 * (a_tr - b_tr)
        + 2.0 * (a_id - b_id)
        + 100.0 * max(0.0, (b_tr_nll - a_tr_nll) / max(b_tr_nll, 1e-9))
        + 50.0 * max(0.0, (b_id_nll - a_id_nll) / max(b_id_nll, 1e-9))
    )
    selectable = not regressions and math_growth
    return {
        "selectable": selectable,
        "regressions": regressions,
        "checks": checks,
        "improvements": deltas,
        "math_growth": {
            "pass": math_growth,
            "exact_growth": exact_growth,
            "soft_nll_growth": soft_math_growth,
        },
        "score": score,
        "policy": (
            "Training-time independent-dev Pareto guard. No tracked metric may regress; "
            "math must improve by exact match or by >=0.5% on both ID and transfer NLL. "
            "Frozen release benchmarks are not used for checkpoint selection."
        ),
    }
