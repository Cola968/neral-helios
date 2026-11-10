"""Fail-closed NERAL development regression veto. Never grants release.

Read-only CPU diagnostic using scalar counts from the 72-case three-way suite.
A passed development screen is NOT an independent benchmark or a release grant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from scripts.neral_v6_fresh_dev_eval import (
    BASE_SHA, V5_SHA, V6_SHA, DOMAINS, build_cases, suite_fingerprint
)

VARIANTS = ("qwen_fewshot", "neral_v5", "neral_v6")
SHA = re.compile(r"[a-f0-9]{64}\Z")
SOURCE_ISOLATION = "exact_and_10gram_checked_against_v6_train_validation_only"
KNOWN_REGRESSED_ADAPTERS = frozenset({V6_SHA})


class RegressionRejected(ValueError):
    """Missing/invalid evidence or a candidate with known harmful regression."""


def _count(value: object, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise RegressionRejected("Missing or untrusted integer count")
    return value


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RegressionRejected(reason)


def audit_candidate_report(report: dict) -> dict:
    """Inspect only scalar scores, rejecting fabricated or incomplete contracts.

    Passing the screen permits *more independent evaluation*, never deployment.
    The 72-case development set has been inspected and is never release evidence.
    """
    from scripts.neral_v6_fresh_dev_eval import report_exposes_locked_holdout
    _require(not report_exposes_locked_holdout(report),
             "Locked holdout material is not an input to this decision")
    _require(isinstance(report, dict), "Missing candidate report")
    _require(type(report.get("schema")) is int and report["schema"] == 1,
             "Untrusted schema")
    _require(report.get("status") == "NERAL_V6_NEW_STRUCTURAL_DEV_72_THREE_WAY",
             "Unexpected benchmark report")
    _require(type(report.get("total")) is int and report["total"] == 72,
             "Expected exactly 72 paired development tasks")
    _require(type(report.get("task_families")) is int and report["task_families"] == 12,
             "Expected 12 development task families")
    _require(report.get("suite_sha256") == suite_fingerprint(build_cases()),
             "Benchmark suite does not match pinned generator")
    _require(report.get("source_isolation") == SOURCE_ISOLATION,
             "Source-isolation record absent or changed")
    _require(report.get("base_revision") == BASE_SHA, "Unexpected foundation revision")
    _require(report.get("v5_adapter_sha256") == V5_SHA,
             "Wrong v5 reference adapter")
    candidate_sha = report.get("v6_adapter_sha256")
    _require(isinstance(candidate_sha, str) and bool(SHA.fullmatch(candidate_sha)),
             "Missing or invalid candidate weight SHA")
    for field in ("release_validated", "weights_promoted", "independent_benchmark_proven"):
        _require(report.get(field) is False,
                 "Development report must not carry release or independence claims")

    scores = report.get("scores")
    _require(isinstance(scores, dict) and set(scores) == set(VARIANTS),
             "Every original model and the same baseline are mandatory")
    for variant in VARIANTS:
        s = scores[variant]
        _require(isinstance(s, dict), "Invalid per-model scores")
        correct = _count(s.get("correct"), maximum=72)
        _require(_count(s.get("total"), maximum=72) == 72,
                 "Incorrect per-model denominator")
        incomplete = _count(s.get("incomplete"), maximum=72)
        ungraded = _count(s.get("ungraded_completed"), maximum=72)
        _require(correct + incomplete + ungraded <= 72,
                 "Inconsistent completed/gradable model results")
        domains = s.get("per_domain")
        _require(isinstance(domains, dict) and set(domains) == set(DOMAINS),
                 "All three original domains are mandatory")
        domain_scores = [_count(domains[d], maximum=24) for d in DOMAINS]
        _require(sum(domain_scores) == correct, "Domain and total correct counts disagree")

    pairs = report.get("paired")
    _require(isinstance(pairs, dict) and set(pairs) == {
        "v5_vs_qwen", "v6_vs_qwen", "v6_vs_v5"
    }, "All paired model differences are mandatory")
    for label, candidate, base in (
        ("v5_vs_qwen", "neral_v5", "qwen_fewshot"),
        ("v6_vs_qwen", "neral_v6", "qwen_fewshot"),
        ("v6_vs_v5", "neral_v6", "neral_v5"),
    ):
        pair = pairs[label]
        _require(isinstance(pair, dict), "Missing paired comparison")
        improve = _count(pair.get("improvements"), maximum=72)
        regress = _count(pair.get("regressions"), maximum=72)
        _require(improve + regress <= 72, "Too many discordant model pairs")
        _require(improve <= scores[candidate]["correct"] and
                 regress <= scores[base]["correct"],
                 "Impossible paired contingency table")
        _require(improve - regress ==
                 scores[candidate]["correct"] - scores[base]["correct"],
                 "Inconsistent paired improvement and regression counts")

    failures: list[str] = []
    if candidate_sha in KNOWN_REGRESSED_ADAPTERS:
        failures.append("known_regressed_adapter_sha")
    candidate = scores["neral_v6"]
    for baseline in ("neral_v5", "qwen_fewshot"):
        ref = scores[baseline]
        if candidate["correct"] < ref["correct"]:
            failures.append(f"total_below_{baseline}")
        elif candidate["correct"] == ref["correct"]:
            failures.append(f"no_accuracy_gain_over_{baseline}")
        if candidate["incomplete"] > ref["incomplete"]:
            failures.append(f"more_incomplete_than_{baseline}")
        if candidate["ungraded_completed"] > ref["ungraded_completed"]:
            failures.append(f"ungraded_worse_than_{baseline}")
        for domain in DOMAINS:
            if candidate["per_domain"][domain] < ref["per_domain"][domain]:
                failures.append(f"domain_regression_{domain}_vs_{baseline}")

    return {
        "schema": 1,
        "status": ("DEVELOPMENT_REGRESSION_BLOCKED" if failures else
                   "DEVELOPMENT_SCREEN_PASSED_NONRELEASE"),
        "candidate_sha256": candidate_sha,
        "candidate_quarantined": bool(failures),
        "development_screen_passed": not failures,
        "reasons": failures,
        "baselines_required": list(VARIANTS[:-1]),
        "release_authorized": False,
        "weights_promoted": False,
        "independent_benchmark_proven": False,
        "limitations": ("Diagnostic veto on an already-consumed, researcher-authored 72-task "
                        "development suite. Reports require independently trusted provenance; "
                        "a passing result never permits adapter deployment. A fresh 600-case "
                        "registered release test plus external checks remains mandatory."),
    }


def require_screen_pass(report: dict) -> None:
    """Hard stop for workflows attempting to advance a regressed candidate."""
    result = audit_candidate_report(report)
    if result["candidate_quarantined"]:
        raise RegressionRejected("Candidate advancement denied: " +
                                 ", ".join(result["reasons"]))


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RegressionRejected("Duplicate benchmark JSON property")
        result[key] = value
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--report", type=Path, required=True,
                   help="Local scalar summary.json; raw answers are never uploaded")
    p.add_argument("--require-screen-pass", action="store_true",
                   help="Exit nonzero on regression; does NOT grant a release")
    a = p.parse_args(argv)
    if (a.report.is_symlink() or not a.report.is_file() or
            not 0 < a.report.stat().st_size <= 100_000):
        raise RegressionRejected("Missing, unsafe or oversize scalar report")
    report = json.loads(a.report.read_text("utf-8"), object_pairs_hook=_no_duplicate_keys)
    decision = audit_candidate_report(report)
    print(json.dumps(decision, sort_keys=True, allow_nan=False))
    if a.require_screen_pass and decision["candidate_quarantined"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
