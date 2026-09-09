from __future__ import annotations

"""Release-level Pareto gate for Neral candidates.

This complements benchmark-blind M2 development selection. It consumes completed, comparable
post-training evidence only. Missing metrics fail closed; a single hard regression rejects the
candidate even when another capability improves strongly.
"""

from dataclasses import dataclass, asdict
import json
import math


@dataclass(frozen=True)
class Metric:
    key: str
    direction: str  # up/down
    max_abs_regression: float
    max_rel_regression: float
    group: str


METRICS = (
    Metric("nis2", "up", 0.10, 0.002, "overall"),
    Metric("validation_ce", "down", 0.010, 0.010, "language_model"),
    Metric("validation_bpb", "down", 0.010, 0.010, "language_model"),
    Metric("math_id_exact_pct", "up", 0.25, 0.010, "math"),
    Metric("math_transfer_exact_pct", "up", 0.25, 0.010, "math_transfer"),
    Metric("math_ood_exact_pct", "up", 0.25, 0.010, "math_ood"),
    Metric("code_pct", "up", 1.00, 0.030, "code"),
    Metric("context_pct", "up", 1.00, 0.025, "context"),
    Metric("reasoning_pct", "up", 1.00, 0.030, "reasoning"),
    Metric("instruction_pct", "up", 1.00, 0.030, "instruction"),
    Metric("eos_pct", "up", 2.00, 0.050, "generation"),
    Metric("repetition_pct", "down", 0.50, 0.100, "generation"),
    Metric("utf8_pct", "up", 0.10, 0.001, "generation"),
    Metric("forgetting_score", "down", 0.010, 0.020, "forgetting"),
)


def _finite(value, key: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite metric: {key}")
    return x


def regression_limit(anchor: float, spec: Metric) -> float:
    return max(spec.max_abs_regression, abs(anchor) * spec.max_rel_regression)


def regression(spec: Metric, anchor: float, candidate: float) -> float:
    return (anchor - candidate) if spec.direction == "up" else (candidate - anchor)


def assess(anchor: dict, candidate: dict, *, targeted_math: bool = False, require_nis_gain: float = 0.25) -> dict:
    missing = sorted({m.key for m in METRICS} - set(anchor) | ({m.key for m in METRICS} - set(candidate)))
    if missing:
        return {"accepted": False, "missing": missing, "blockers": ["missing_evidence:" + k for k in missing]}

    rows = []
    blockers = []
    for spec in METRICS:
        a = _finite(anchor[spec.key], spec.key)
        c = _finite(candidate[spec.key], spec.key)
        reg = regression(spec, a, c)
        limit = regression_limit(a, spec)
        ok = reg <= limit + 1e-12
        rows.append({**asdict(spec), "anchor": a, "candidate": c, "regression": reg, "allowed_regression": limit, "pass": ok})
        if not ok:
            blockers.append(f"regression:{spec.key}")

    nis_gain = float(candidate["nis2"]) - float(anchor["nis2"])
    if nis_gain < require_nis_gain:
        blockers.append("insufficient_nis2_gain")

    id_gain = float(candidate["math_id_exact_pct"]) - float(anchor["math_id_exact_pct"])
    transfer_gain = float(candidate["math_transfer_exact_pct"]) - float(anchor["math_transfer_exact_pct"])
    ood_gain = float(candidate["math_ood_exact_pct"]) - float(anchor["math_ood_exact_pct"])
    generalization_gain = max(transfer_gain, ood_gain)

    if targeted_math:
        if id_gain <= 0:
            blockers.append("math_id_no_gain")
        if generalization_gain <= 0:
            blockers.append("math_transfer_ood_no_gain")
        # Prevent a candidate from passing because one transfer slice rises while the other collapses.
        if transfer_gain < -0.25:
            blockers.append("math_transfer_regression")
        if ood_gain < -0.25:
            blockers.append("math_ood_regression")

    # Generation is a compound release property: valid bytes alone cannot hide endless text loops.
    if float(candidate["utf8_pct"]) < 99.0:
        blockers.append("utf8_below_release_floor")
    if float(candidate["repetition_pct"]) >= 5.0:
        blockers.append("repetition_above_release_floor")
    if float(candidate["eos_pct"]) < 30.0:
        blockers.append("eos_below_release_floor")

    blockers = sorted(set(blockers))
    return {
        "accepted": not blockers,
        "blockers": blockers,
        "metrics": rows,
        "nis2_gain": nis_gain,
        "math": {"id_gain": id_gain, "transfer_gain": transfer_gain, "ood_gain": ood_gain, "generalization_gain": generalization_gain},
        "targeted_math": targeted_math,
        "policy": "hard Pareto release gate; no weighted tradeoff can override a hard blocker",
    }


def compare_many(anchor: dict, candidates: dict[str, dict], *, targeted_math: bool = False) -> dict:
    return {name: assess(anchor, row, targeted_math=targeted_math) for name, row in candidates.items()}


if __name__ == "__main__":
    print(json.dumps({"metrics": [asdict(m) for m in METRICS]}, indent=2))
