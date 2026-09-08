"""Transactional Pareto controller for Neral-50M M2.

This module is benchmark-blind. It operates only on independent development metrics and
objective weights. A training block can be committed only when it preserves every tracked
metric relative to both the original source and the last committed anchor while making real
math progress. Rejected blocks are rolled back by the trainer.
"""
from __future__ import annotations

from pareto_dev_guard50 import METRICS, assess_dev, not_worse


def regressions(before: dict, after: dict, atol: float = 1e-4) -> list[str]:
    bad: list[str] = []
    for spec in METRICS:
        if spec.key not in before or spec.key not in after:
            raise KeyError(f"missing M2 metric: {spec.key}")
        b = float(before[spec.key]); a = float(after[spec.key])
        if not_worse(b, a, spec.direction, atol=atol):
            continue
        bad.append(spec.key)
    return bad


def acceptance(source: dict, anchor: dict, candidate: dict, best_score: float, min_score_gain: float = 0.05) -> dict:
    """Return whether a candidate may atomically replace the current safe anchor."""
    source_assessment = assess_dev(source, candidate, atol=1e-4)
    anchor_regressions = regressions(anchor, candidate, atol=1e-4)
    score = float(source_assessment["score"])
    score_ok = score >= float(best_score) + float(min_score_gain)
    accepted = bool(source_assessment["selectable"] and not anchor_regressions and score_ok)
    return {
        "accepted": accepted,
        "source_assessment": source_assessment,
        "anchor_regressions": anchor_regressions,
        "score": score,
        "score_ok": score_ok,
    }


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    keys = ("id_math", "transfer_math", "sft", "lm")
    if set(weights) != set(keys):
        raise ValueError(f"expected weight keys {keys}, got {tuple(weights)}")
    values = {k: max(0.05, float(weights[k])) for k in keys}
    total = sum(values.values())
    return {k: values[k] / total for k in keys}


def adapt_weights(weights: dict[str, float], bad: list[str]) -> dict[str, float]:
    """Shift objective mass toward the capability that caused a rejected block.

    The shifts are deliberately small. They change only future development training, never
    frozen benchmark evaluation. The caller still has to roll back the rejected weights.
    """
    out = dict(normalize_weights(weights))
    bad = set(bad)
    if bad & {"id_exact_pct", "id_nll"}:
        out["id_math"] += 0.04
    if bad & {"transfer_exact_pct", "transfer_nll"}:
        out["transfer_math"] += 0.05
    if bad & {"sft_macro_nll", "sft_math_nll", "sft_code_nll", "sft_reasoning_nll", "sft_de_nll",
              "free_eos_pct", "free_repetition_pct", "free_health_score"}:
        out["sft"] += 0.05
    if bad & {"lm_macro", "context_dev_pct"}:
        out["lm"] += 0.06
    return normalize_weights(out)


def exact_transfer_growth(source: dict, candidate: dict, atol: float = 1e-4) -> bool:
    return float(candidate["transfer_exact_pct"]) > float(source["transfer_exact_pct"]) + atol
