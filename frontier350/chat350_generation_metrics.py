"""Arctus 350M generation health: fail-closed metrics and real EOS evidence."""
from __future__ import annotations

import math
import statistics


def repeated_ngram_fraction(ids: list[int], n: int = 3) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    seen = set()
    repeated = 0
    for gram in grams:
        if gram in seen:
            repeated += 1
        seen.add(gram)
    return repeated / len(grams)


def distinct_ngram_fraction(ids: list[int], n: int = 2) -> float:
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def summarize_generations(rows: list[dict], eos_id: int) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("generation rows are empty")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("invalid EOS token ID")
    rep, d2, lengths, eos, utf8 = [], [], [], [], []
    exact = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed generated row")
        ids = row.get("generated_ids")
        if (not isinstance(ids, list)
            or not all(type(token) is int and token >= 0 for token in ids)):
            raise ValueError("generated_ids must contain nonnegative integer IDs")
        hit = eos_id in ids
        content_ids = ids[:ids.index(eos_id)] if hit else ids
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("generation text must be a string")
        try:
            text.encode("utf8", "strict")
            utf8.append(1.0)
        except UnicodeError:
            utf8.append(0.0)
        rep.append(repeated_ngram_fraction(content_ids, 3))
        d2.append(distinct_ngram_fraction(content_ids, 2))
        lengths.append(len(content_ids))
        eos.append(float(hit))
        if "expected_exact" in row:
            target = row["expected_exact"]
            if not isinstance(target, str):
                raise ValueError("expected exact answer must be text")
            exact.append(float(text.strip() == target.strip()))
    return {
        "cases": len(rows),
        "eos_fraction": statistics.fmean(eos),
        "mean_repeated_trigram_fraction": statistics.fmean(rep),
        "mean_distinct2_fraction": statistics.fmean(d2),
        "mean_generated_tokens_before_eos": statistics.fmean(lengths),
        "utf8_fraction": statistics.fmean(utf8),
        "exact_instruction_pct": 100.0 * statistics.fmean(exact) if exact else None,
    }


def _validated_rate(report: dict, key: str, scale: float) -> float:
    if not isinstance(report, dict):
        raise ValueError("generation health report must be an object")
    value = report.get(key)
    if (type(value) not in (int, float) or not math.isfinite(value)
        or value < 0 or value > scale):
        raise ValueError("invalid/missing generation metric: " + key)
    return float(value)


def compare_health(baseline: dict, candidate: dict) -> dict:
    # Missing data must not masquerade as "no regression". Old comparisons
    # treated a missing exact score as success and accepted malformed floats.
    b_eos = _validated_rate(baseline, "eos_fraction", 1.0)
    c_eos = _validated_rate(candidate, "eos_fraction", 1.0)
    b_rep = _validated_rate(baseline, "mean_repeated_trigram_fraction", 1.0)
    c_rep = _validated_rate(candidate, "mean_repeated_trigram_fraction", 1.0)
    _validated_rate(baseline, "utf8_fraction", 1.0)
    c_utf8 = _validated_rate(candidate, "utf8_fraction", 1.0)

    b_exact_raw = baseline.get("exact_instruction_pct")
    c_exact_raw = candidate.get("exact_instruction_pct")
    if b_exact_raw is not None:
        b_exact = _validated_rate(baseline, "exact_instruction_pct", 100.0)
    else:
        b_exact = None
    if c_exact_raw is not None:
        c_exact = _validated_rate(candidate, "exact_instruction_pct", 100.0)
    else:
        c_exact = None

    eos_ok = c_eos >= b_eos
    repetition_ok = c_rep <= b_rep + 1e-12
    utf8_ok = c_utf8 >= 0.999
    exact_available = b_exact is not None and c_exact is not None
    exact_ok = exact_available and c_exact >= b_exact
    return {
        "eos_no_regression": eos_ok,
        "repetition_no_regression": repetition_ok,
        "utf8_ok": utf8_ok,
        "exact_instruction_metric_available": exact_available,
        "exact_instruction_no_regression": exact_ok,
        "health_no_regression": eos_ok and repetition_ok and utf8_ok and exact_ok,
    }
