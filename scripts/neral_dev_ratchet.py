"""Development-only nonregression check for NERAL's fixed Hard36 diagnostic.

This cannot approve a release, prove model intelligence, or substitute for a
separately held-out evaluation. It only detects loss of measured correct cases
in a complete, pinned 36-case development summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess_hard36(summary: dict) -> dict:
    if type(summary) is not dict:
        raise ValueError("Expected a JSON object")
    if (summary.get("status") != "NERAL_TIR_HARD36_DEVELOPMENT_ONLY"
            or summary.get("cases") != 36
            or summary.get("independent_holdout_verified") is not False
            or summary.get("weight_improvement_proven") is not False):
        raise ValueError("Wrong development-only benchmark provenance")
    counts = summary.get("correct")
    if type(counts) is not dict:
        raise ValueError("Missing per-arm correct counts")
    baseline, candidate = counts.get("raw_qwen35"), counts.get("neral_high_router")
    if (type(baseline) is not int or baseline != 20
            or type(candidate) is not int or not 0 <= candidate <= 36):
        raise ValueError("Incompatible baseline or invalid candidate count")
    return {
        "scope": "hard36_development_only",
        "baseline_correct": baseline,
        "candidate_correct": candidate,
        "regression_cases": max(0, baseline - candidate),
        "development_gate_passed": candidate >= baseline,
        "promotion_allowed": False,
        "independent_holdout_verified": False,
        "explanation": (
            "No decline on the fixed development metric; this is not a release gate"
            if candidate >= baseline else
            "Candidate regressed versus raw Qwen3.5 on the fixed development metric"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path,
                        help="Actual scalar-summary.json from a completed GPU hard36 run")
    args = parser.parse_args()
    try:
        report = assess_hard36(json.loads(args.summary.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
