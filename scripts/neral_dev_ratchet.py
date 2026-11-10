"""Development-only paired nonregression diagnostic for NERAL's Hard36 runs.

Hard36 draws a new random 36-case set on every run. Compare Qwen and NERAL
within the SAME run, never against the raw 20/36 obtained in an earlier run.
This script cannot authorize deployment, verify a holdout or validate weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess_hard36(summary: dict, *, candidate_arm: str = "neral_high_router") -> dict:
    if candidate_arm not in ("neral_high", "neral_high_router"):
        raise ValueError("Unknown permitted HIGH comparison arm")
    if type(summary) is not dict:
        raise ValueError("Expected a JSON object")
    if (summary.get("status") != "NERAL_SAME_BASE_3ARM_DEVELOPMENT_DIAGNOSTIC"
            or summary.get("collector_status") != "NERAL_TIR_HARD36_DEVELOPMENT_ONLY"
            or summary.get("cases") != 36
            or summary.get("independent_holdout_verified") is not False
            or summary.get("weight_improvement_proven") is not False
            or summary.get("automatic_promotion_authorized") is not False):
        raise ValueError("Wrong development-only benchmark provenance")

    arms = summary.get("arms")
    paired = summary.get("paired")
    if type(arms) is not dict or type(paired) is not dict:
        raise ValueError("Missing paired scorer results")
    raw = arms.get("raw_qwen35")
    candidate = arms.get(candidate_arm)
    match = paired.get(candidate_arm + "_vs_raw_qwen35")
    if any(type(v) is not dict for v in (raw, candidate, match)):
        raise ValueError("Missing same-run baseline, candidate or paired result")

    baseline, result = raw.get("correct"), candidate.get("correct")
    wins, losses = match.get("wins"), match.get("losses")
    if (any(type(v) is not int or not 0 <= v <= 36
            for v in (baseline, result, wins, losses))
            or wins + losses > 36
            or result - baseline != wins - losses
            or losses > baseline or wins > 36 - baseline):
        raise ValueError("Invalid or inconsistent same-run paired counts")

    return {
        "scope": "hard36_development_only",
        "candidate_arm": candidate_arm,
        "baseline_correct": baseline,
        "candidate_correct": result,
        "paired_wins": wins,
        "paired_losses": losses,
        "development_gate_passed": result >= baseline,
        "promotion_allowed": False,
        "independent_holdout_verified": False,
        "explanation": (
            "No score decline on this SAME set of 36 development cases; not a release approval"
            if result >= baseline else
            "Candidate regressed against SAME-RUN raw Qwen on development cases"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path,
                        help="Actual scalar-summary.json from a completed Hard36 GPU run")
    parser.add_argument("--candidate-arm", choices=("neral_high", "neral_high_router"),
                        default="neral_high_router",
                        help="Candidate in the same original paired Hard36 report")
    args = parser.parse_args()
    try:
        report = assess_hard36(json.loads(args.summary.read_text(encoding="utf-8")),
                               candidate_arm=args.candidate_arm)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
