"""Supplemental final-quality gate for Neral-50M I1.

This gate does not replace the frozen promotion gate. It adds explicit post-B3 checks for the
new Targeted-v1 diagnostic (German grammar, math, code and reasoning) and requires the legacy
promotion gate to pass. Because Targeted-v1 was frozen after B3 had already begun, it is
reported as supplemental evidence rather than retroactively treated as contamination-proof.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DOMAINS = ("de_grammar", "math", "code", "reasoning")
MIN_DOMAIN_PCT = 50.0
MIN_OVERALL_PCT = 55.0
MAX_DOMAIN_REGRESSION_PP = 5.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pct(report, domain=None):
    row = report.get("overall", {}) if domain is None else report.get("domains", {}).get(domain, {})
    value = row.get("accuracy")
    return 100.0 * float(value) if finite(value) else None


def evaluate(comparison, promotion, baseline_targeted, candidate_targeted):
    checks = {}

    def add(name, baseline, candidate, passed, note=None):
        checks[name] = {
            "baseline_or_target": baseline,
            "candidate": candidate,
            "pass": bool(passed),
        }
        if note:
            checks[name]["note"] = note

    same_suite = bool(
        baseline_targeted.get("suite_sha256")
        and baseline_targeted.get("suite_sha256") == candidate_targeted.get("suite_sha256")
        and baseline_targeted.get("suite_sha256") == comparison.get("targeted_suite_sha256")
    )
    add("targeted_same_suite", baseline_targeted.get("suite_sha256"), candidate_targeted.get("suite_sha256"), same_suite)

    base_overall = pct(baseline_targeted)
    cand_overall = pct(candidate_targeted)
    add("targeted_overall_at_least_55", MIN_OVERALL_PCT, cand_overall, finite(cand_overall) and cand_overall >= MIN_OVERALL_PCT)
    add("targeted_overall_nonregression", base_overall, cand_overall, finite(base_overall) and finite(cand_overall) and cand_overall >= base_overall)

    for domain in DOMAINS:
        b = pct(baseline_targeted, domain)
        c = pct(candidate_targeted, domain)
        add(f"targeted_{domain}_at_least_50", MIN_DOMAIN_PCT, c, finite(c) and c >= MIN_DOMAIN_PCT)
        add(
            f"targeted_{domain}_regression_guard",
            b,
            c,
            finite(b) and finite(c) and c >= b - MAX_DOMAIN_REGRESSION_PP,
            note="No more than 5 percentage points below Neral-20M on the supplemental domain.",
        )

    nis = comparison.get("nis", {}).get("instruct", {}).get("score")
    add("nis_at_least_55", 55.0, nis, finite(nis) and nis >= 55.0)
    add("legacy_promotion_gate", True, promotion.get("ready_for_rc_review"), promotion.get("ready_for_rc_review") is True)

    return {
        "campaign": "Neral-50M I1",
        "gate": "supplemental_quality_v1",
        "quality_targets_met": all(row["pass"] for row in checks.values()),
        "checks": checks,
        "policy": "Targeted-v1 is supplemental post-B3 evidence; frozen legacy promotion remains mandatory. No tools/SOLVE/Helios credit.",
        "targeted_suite_status": "supplemental_post_b3",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--promotion", required=True)
    ap.add_argument("--baseline-targeted", required=True)
    ap.add_argument("--candidate-targeted", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = evaluate(load(args.comparison), load(args.promotion), load(args.baseline_targeted), load(args.candidate_targeted))
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
