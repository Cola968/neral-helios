"""Explicit I1 acceptance targets using only frozen pre-I1 model-only release suites.

Targeted-v1 is intentionally not used here because B3 was already frozen before that
supplemental suite was created. This gate therefore remains the contamination-audited release
criterion for I1 and strengthens math/code/German/non-regression requirements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate(comparison, before_v5, after_v5, promotion):
    before = comparison.get("stages", {}).get("baseline20", {})
    after = comparison.get("stages", {}).get("instruct", {})
    checks = {}

    def check(name, baseline_or_target, candidate, relation):
        checks[name] = {
            "baseline_or_target": baseline_or_target,
            "candidate": candidate,
            "pass": bool(finite(baseline_or_target) and finite(candidate) and relation(candidate, baseline_or_target)),
        }

    nis = comparison.get("nis", {})
    score = nis.get("instruct", {}).get("score")
    check("nis_at_least_55", 55, score, lambda a, b: a >= b)
    check("nis_beats_preview02", nis.get("baseline20", {}).get("score"), score, lambda a, b: a > b)

    # Free-generation health: readable, terminating, non-collapsed output.
    check("utf8_at_least_99_pct", 99, after.get("utf8_valid_pct"), lambda a, b: a >= b)
    check("free_eos_at_least_60_pct", 60, after.get("eos_pct"), lambda a, b: a >= b)
    check("free_repetition_below_5_pct", 5, after.get("repeated_trigram_pct"), lambda a, b: a < b)

    # Raw arithmetic must become a real capability rather than only preserving language loss.
    check("rawmath_at_least_60_pct", 60, after.get("rawmath_v1_exact_pct"), lambda a, b: a >= b)
    check("rawmath_transfer_ood_at_least_45_pct", 45, after.get("rawmath_transfer_ood_pct"), lambda a, b: a >= b)

    # Code/math/German on suites that were already frozen and screened before B3 data build.
    for key in ("v3_code_pct", "v4_code_pct"):
        check(key + "_held", before.get(key), after.get(key), lambda a, b: a >= b)
        check(key + "_at_least_50", 50, after.get(key), lambda a, b: a >= b)
    check("v4_math_held", before.get("v4_math_pct"), after.get("v4_math_pct"), lambda a, b: a >= b)
    check("v4_math_at_least_50", 50, after.get("v4_math_pct"), lambda a, b: a >= b)
    for key in ("v3_de_grammar_pct", "v4_de_grammar_pct", "v3_pair_overall_pct", "v4_pair_overall_pct", "context_recall_overall_pct"):
        check(key + "_held", before.get(key), after.get(key), lambda a, b: a >= b)

    # Long-context evidence must not be traded away for instruction following.
    for key in ("single_needle", "multi_needle"):
        check(key + "_held", before_v5.get(key, {}).get("accuracy"), after_v5.get(key, {}).get("accuracy"), lambda a, b: a >= b)
    for seq in ("2048", "3072", "4095"):
        key = "document_bounded_true_byte_bpb"
        b = before_v5.get(key, {}).get("macro_bits_per_byte", {}).get(seq)
        a = after_v5.get(key, {}).get("macro_bits_per_byte", {}).get(seq)
        check("bpb_" + seq + "_held", b, a, lambda a, b: 0 < a <= b)

    checks["v5_same_frozen_suite"] = {"pass": bool(before_v5.get("suite_sha256") and before_v5.get("suite_sha256") == after_v5.get("suite_sha256"))}
    checks["existing_promotion_gate"] = {"pass": promotion.get("ready_for_rc_review") is True}
    return {
        "campaign": "Neral-50M I1",
        "gate_version": 2,
        "targets_met": all(v["pass"] for v in checks.values()),
        "checks": checks,
        "policy": "Final frozen pre-I1 comparison only. No SOLVE/Helios credit. No weights deleted or promoted automatically.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    args = ap.parse_args()
    root = Path(args.reports)
    load = lambda p: json.loads((root / p).read_text(encoding="utf8"))
    result = evaluate(load("I1_20M_VS_50M.json"), load("baseline20/v5.json"), load("candidate50/v5.json"), load("I1_PROMOTION_GATE.json"))
    (root / "I1_TARGET_GATE.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
