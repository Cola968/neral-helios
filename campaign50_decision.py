"""Final decision helper for the autonomous Neral-50M campaign.

A candidate only counts as having surpassed 20M Preview02 when all three independent
requirements hold:
1. the existing hard 8/9 promotion gate is ready for RC review,
2. NIS 1.0 is strictly higher than the identical-harness 20M score,
3. frozen v5 long-context evaluation shows no severe regression versus 20M.

This helper does not train, generate, call tools, or alter benchmarks. It only combines
already-written frozen reports so the runner can stop early when the target has genuinely
been demonstrated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from long_context_gate50 import decide as long_context_decide
from promotion_gate50 import evaluate as promotion_evaluate


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def decide(comparison: dict, baseline_v5: dict, candidate_v5: dict, candidate_name: str = "instruct") -> dict:
    promotion = promotion_evaluate(comparison, candidate_name=candidate_name)
    nis = comparison.get("nis", {})
    baseline_nis = nis.get("baseline20", {}).get("score")
    candidate_nis = nis.get(candidate_name, {}).get("score")
    nis_available = baseline_nis is not None and candidate_nis is not None
    nis_strict_win = bool(nis_available and float(candidate_nis) > float(baseline_nis))

    long_context = long_context_decide(baseline_v5, candidate_v5, mode="protect")
    winner = bool(
        promotion.get("ready_for_rc_review")
        and nis_strict_win
        and long_context.get("proceed")
    )

    reasons = []
    if not promotion.get("ready_for_rc_review"):
        reasons.append("hard_promotion_gate_not_passed")
    if not nis_available:
        reasons.append("nis_incomplete")
    elif not nis_strict_win:
        reasons.append("nis_not_above_20m")
    if not long_context.get("proceed"):
        reasons.append("v5_long_context_regression")

    return {
        "surpassed_20m": winner,
        "candidate": candidate_name,
        "rule": "8/9 hard promotion gate AND NIS1.0 > 20M AND v5 long-context protection pass",
        "promotion": promotion,
        "nis": {
            "baseline20": baseline_nis,
            "candidate": candidate_nis,
            "strict_win": nis_strict_win,
        },
        "long_context": long_context,
        "reasons": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("comparison")
    ap.add_argument("baseline_v5")
    ap.add_argument("candidate_v5")
    ap.add_argument("--candidate", default="instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = decide(load(args.comparison), load(args.baseline_v5), load(args.candidate_v5), args.candidate)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "surpassed_20m": result["surpassed_20m"],
        "strict_wins": result["promotion"].get("strict_wins"),
        "nis20": result["nis"]["baseline20"],
        "nis50": result["nis"]["candidate"],
        "v5_protect": result["long_context"].get("proceed"),
        "reasons": result["reasons"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
