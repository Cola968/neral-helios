"""Small, dependency-free loss telemetry helpers for Neral-50M.

The functions in this module do not train or modify weights. They make checkpoint selection
more precise and expose cross-entropy in easier-to-read forms (perplexity and bits/token).
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def ce_to_perplexity(ce: float) -> float:
    ce = float(ce)
    if not math.isfinite(ce):
        return float("inf")
    return math.exp(ce)


def ce_to_bits(ce: float) -> float:
    ce = float(ce)
    if not math.isfinite(ce):
        return float("inf")
    return ce / math.log(2.0)


def validation_derived(row: dict) -> dict:
    macro = float(row["macro"])
    domains = {str(k): float(v) for k, v in row.get("domains", {}).items()}
    return {
        "macro_ce": macro,
        "macro_perplexity": ce_to_perplexity(macro),
        "macro_bits_per_token": ce_to_bits(macro),
        "domain_perplexity": {k: ce_to_perplexity(v) for k, v in domains.items()},
        "domain_bits_per_token": {k: ce_to_bits(v) for k, v in domains.items()},
    }


def should_select(score: float, best: float, min_delta: float, *, eligible: bool = True) -> bool:
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    score, best = float(score), float(best)
    if not eligible or not math.isfinite(score):
        return False
    return not math.isfinite(best) or score <= best - min_delta


def eval_due(step: int, normal_every: int, early_every: int, early_until: int) -> bool:
    if min(step, normal_every, early_every) <= 0 or early_until < 0:
        raise ValueError("invalid evaluation schedule")
    every = early_every if step <= early_until else normal_every
    return step % every == 0


def summarize_metrics(path: str | Path, window: int = 5) -> dict:
    path = Path(path)
    if window <= 1:
        raise ValueError("window must be >1")
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    evals = [r for r in rows if isinstance(r.get("validation"), dict) and "macro" in r["validation"]]
    if not evals:
        return {"evaluations": 0, "status": "no_validation"}
    values = [float(r["validation"]["macro"]) for r in evals]
    latest = values[-1]
    best = min(values)
    recent = values[-window:]
    slope = 0.0
    if len(recent) >= 2:
        n = len(recent)
        xm = (n - 1) / 2
        ym = sum(recent) / n
        denom = sum((i - xm) ** 2 for i in range(n))
        slope = sum((i - xm) * (y - ym) for i, y in enumerate(recent)) / max(denom, 1e-12)
    status = "improving" if slope < -1e-4 else "regressing" if slope > 1e-4 else "plateau"
    return {
        "evaluations": len(evals),
        "latest_ce": latest,
        "best_ce": best,
        "latest_perplexity": ce_to_perplexity(latest),
        "best_perplexity": ce_to_perplexity(best),
        "recent_ce_slope_per_eval": slope,
        "status": status,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(summarize_metrics(args.metrics, args.window), indent=2))
