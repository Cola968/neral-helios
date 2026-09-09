from __future__ import annotations

"""Frozen release gate for Neral 350M Frontier.

Input is a JSON scorecard produced by the frozen benchmark stack. Training and checkpoint
selection must never inspect benchmark examples. This gate is post-training only and is
intentionally difficult: 350M is the first model line intended for a public Helios release.
"""

import argparse
import json
import math
from pathlib import Path

# Hard model-quality gate. Missing, NaN or infinite values always fail closed.
MINIMUMS = {
    "nis2": 85.0,
    "rawmath_exact": 50.0,
    "targeted_math": 85.0,
    "code": 82.0,
    "de": 97.0,
    "reasoning": 94.0,
    "context": 100.0,
    "free_eos": 92.0,
    "utf8": 99.99,
}
MAXIMUMS = {
    "repetition": 1.5,
    "bpb4095": 1.25,
}

# Research targets beyond the Play Store model gate. They are not assumed achievable and
# must only be reported when actually measured by the frozen suite.
STRETCH_MINIMUMS = {
    "nis2": 88.0,
    "rawmath_exact": 60.0,
    "targeted_math": 92.0,
    "code": 88.0,
    "de": 99.0,
    "reasoning": 97.0,
    "context": 100.0,
    "free_eos": 97.0,
    "utf8": 99.99,
}
STRETCH_MAXIMUMS = {
    "repetition": 1.0,
    "bpb4095": 1.18,
}


def finite_number(value) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("non-finite metric")
    return x


def assess(metrics: dict) -> dict:
    required = [*MINIMUMS, *MAXIMUMS]
    missing = [k for k in required if k not in metrics]
    failures = [f"missing:{x}" for x in missing]
    clean: dict[str, float] = {}
    for key in required:
        if key in metrics:
            try:
                clean[key] = finite_number(metrics[key])
            except (TypeError, ValueError):
                failures.append(f"nonfinite:{key}")

    for key, threshold in MINIMUMS.items():
        if key in clean and clean[key] < threshold:
            failures.append(f"{key}<{threshold}")
    for key, threshold in MAXIMUMS.items():
        if key in clean and clean[key] > threshold:
            failures.append(f"{key}>{threshold}")

    stretch_hits = {}
    for key, threshold in STRETCH_MINIMUMS.items():
        stretch_hits[key] = key in clean and clean[key] >= threshold
    for key, threshold in STRETCH_MAXIMUMS.items():
        stretch_hits[key] = key in clean and clean[key] <= threshold

    passed = not failures
    all_stretch = passed and all(stretch_hits.values())
    return {
        "pass": passed,
        "release_candidate": passed,
        "frontier_stretch_complete": all_stretch,
        "failures": failures,
        "minimums": MINIMUMS,
        "maximums": MAXIMUMS,
        "stretch_minimums": STRETCH_MINIMUMS,
        "stretch_maximums": STRETCH_MAXIMUMS,
        "stretch_hits": stretch_hits,
        "metrics": metrics,
        "policy": (
            "No public Helios model promotion unless every hard frozen model-quality metric passes. "
            "NIS2 is necessary but not sufficient; math, code, German, reasoning, context, EOS, "
            "repetition, UTF-8 and BPB must pass simultaneously. Separate safety/privacy/mobile "
            "release gates are also required before Play Store publication."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    metrics = json.loads(Path(args.scorecard).read_text(encoding="utf8"))
    result = assess(metrics)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
