"""Transactional Pareto controller for Neral-50M M2.

This module is benchmark-blind. It operates only on independent development metrics and
objective weights. A training block can be committed only when it preserves every tracked
metric relative to both the original source and the last committed anchor while making real
math progress. M2.1 adds conservative safety margins for the three failure modes exposed by
the first frozen post-training audit: UTF-8 validity, German preservation and free repetition.
Rejected blocks are rolled back by the trainer.
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


def transfer_progress(anchor: dict, candidate: dict, atol: float = 1e-4, min_rel_nll_gain: float = 0.005) -> bool:
    """Require every committed block to move range-disjoint transfer forward.

    Exact-match growth is preferred. Before exact answers appear, a >=0.5% transfer-NLL
    reduction relative to the last safe anchor is accepted as shaping progress. This prevents
    transfer weight from silently shrinking while ID math improves.
    """
    exact = float(candidate["transfer_exact_pct"]) > float(anchor["transfer_exact_pct"]) + atol
    nll = float(candidate["transfer_nll"]) <= float(anchor["transfer_nll"]) * (1.0 - min_rel_nll_gain) + atol
    return exact or nll


def safety_buffers(source: dict, candidate: dict, atol: float = 1e-4) -> list[str]:
    """Extra independent-dev margin for metrics that proved noisy in the first M2 run."""
    bad: list[str] = []
    # A single invalid byte sequence is enough to fail the frozen Pareto gate.
    if float(candidate["free_utf8_pct"]) + atol < float(source["free_utf8_pct"]):
        bad.append("free_utf8_pct")
    # German NLL improved slightly in M2 while frozen German accuracy still fell 5 pp.
    # Demand a full 1% held-out German-NLL gain to create headroom against threshold flips.
    if float(candidate["sft_de_nll"]) > float(source["sft_de_nll"]) * 0.99 + atol:
        bad.append("german_safety_margin")
    # The old free-dev set had a much higher repetition baseline than the frozen free suite.
    # Require a substantial relative improvement so a small distribution shift cannot erase it.
    if float(candidate["free_repetition_pct"]) > float(source["free_repetition_pct"]) * 0.65 + atol:
        bad.append("repetition_safety_margin")
    # Require visible EOS headroom rather than accepting a numerically flat generation profile.
    if float(candidate["free_eos_pct"]) + atol < float(source["free_eos_pct"]) + 2.0:
        bad.append("eos_safety_margin")
    return bad


def acceptance(source: dict, anchor: dict, candidate: dict, best_score: float, min_score_gain: float = 0.05) -> dict:
    """Return whether a candidate may atomically replace the current safe anchor."""
    source_assessment = assess_dev(source, candidate, atol=1e-4)
    blockers = regressions(anchor, candidate, atol=1e-4)
    if not transfer_progress(anchor, candidate):
        blockers.append("transfer_stagnation")
    blockers.extend(safety_buffers(source, candidate))
    blockers = sorted(set(blockers))
    score = float(source_assessment["score"])
    score_ok = score >= float(best_score) + float(min_score_gain)
    accepted = bool(source_assessment["selectable"] and not blockers and score_ok)
    return {
        "accepted": accepted,
        "source_assessment": source_assessment,
        "anchor_regressions": blockers,
        "score": score,
        "score_ok": score_ok,
        "transfer_progress": transfer_progress(anchor, candidate),
        "safety_buffers": safety_buffers(source, candidate),
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
    if bad & {"transfer_exact_pct", "transfer_nll", "transfer_stagnation"}:
        out["transfer_math"] += 0.08 if "transfer_stagnation" in bad else 0.05
    if bad & {"sft_macro_nll", "sft_math_nll", "sft_code_nll", "sft_reasoning_nll", "sft_de_nll",
              "free_utf8_pct", "free_eos_pct", "free_repetition_pct", "free_health_score",
              "german_safety_margin", "repetition_safety_margin", "eos_safety_margin"}:
        out["sft"] += 0.07
    if bad & {"lm_macro", "context_dev_pct"}:
        out["lm"] += 0.06
    return normalize_weights(out)


def exact_transfer_growth(source: dict, candidate: dict, atol: float = 1e-4) -> bool:
    return float(candidate["transfer_exact_pct"]) > float(source["transfer_exact_pct"]) + atol
