"""NERAL evidence-first training loop: private-data-safe, read-only decisions.

This policy is applied to the scalar-only post-training v8 report. It detects
negative transfer, generation truncation and absent generalization gains.
It never trains, chooses weights, authorizes release, or inspects raw prompts.
A developer-authored test suite is not an independent blind holdout.
"""
from __future__ import annotations

from collections import Counter

STATUS="NERAL_V8_EVIDENCE_POLICY_RESEARCH_ONLY"
REPORT_STATUS="NERAL_V8_FRESH_DEV_TRANSFER_GATE_NOT_RELEASE"
ARMS=("qwen","v5","v7","v8")
_SUMMARY_FIELDS={
    "total","correct","wrong_final","unreadable_complete",
    "length_censored","non_eos_terminated","parseable_but_unfinished",
}
_BASE_CHECKS={
    "pilot","transfer_gain","transfer_significance",
    "zero_retention_regressions","zero_transfer_regressions",
    "retention_net","retention_by_family",
    "zero_regressions_vs_qwen","zero_regressions_vs_v5",
    "zero_regressions_vs_v7",
}
_TOP_FIELDS={
    "schema","status","sample","case_count","post_training_developer_suite",
    "independent_ood_proven","model_only_no_tools","matched_prompts",
    "greedy_deterministic","max_new_tokens","same_eos_policy",
    "raw_prompts_or_responses_uploaded","training_performed",
    "weights_promoted","release_validated","models","paired",
    "slices","gate","limitations",
}
_OPTIONAL_PROVENANCE={
    "v7_adapter_sha256","v8_adapter_sha256","v8_training_sha256",
    "foundation_revision","predecessor_v5_sha256",
}
_PAIR_ARMS={
    "v8_minus_qwen":("qwen","v8"),
    "v8_minus_v5":("v5","v8"),
    "v8_minus_v7":("v7","v8"),
    "v7_minus_v5":("v5","v7"),
}


class EvidenceError(ValueError):
    """Invalid, incomplete, or untrusted aggregate evidence (never echo data)."""


def _fail() -> None:
    raise EvidenceError("Incomplete or inconsistent evaluation evidence; fail closed.")


def _nonnegative(value, limit=10000) -> bool:
    return type(value) is int and 0<=value<=limit


def _counts(row: dict, expected: int) -> None:
    if type(row) is not dict or set(row)!=_SUMMARY_FIELDS:
        _fail()
    if row["total"]!=expected or any(
        not _nonnegative(row[k],expected) for k in _SUMMARY_FIELDS
    ):
        _fail()
    if (row["correct"]+row["wrong_final"]+row["unreadable_complete"]+
            row["length_censored"]+row["non_eos_terminated"]!=expected):
        _fail()
    if row["parseable_but_unfinished"] > (
        row["length_censored"]+row["non_eos_terminated"]
    ):
        _fail()


def _pair(row: dict, expected: int, baseline_correct: int,
          candidate_correct: int) -> None:
    if (type(row) is not dict or
        set(row)!={"improvements","regressions","net",
                   "exact_mcnemar_p_exploratory"}):
        _fail()
    better,worse=row["improvements"],row["regressions"]
    p=row["exact_mcnemar_p_exploratory"]
    if (not _nonnegative(better,expected) or
        not _nonnegative(worse,expected) or better+worse>expected or
        type(row["net"]) is not int or row["net"]!=better-worse or
        candidate_correct-baseline_correct!=better-worse or
        type(p) not in (int,float) or not 0<=p<=1):
        _fail()


def _validate(report: dict) -> tuple[str,int]:
    if (type(report) is not dict or
        not _TOP_FIELDS.issubset(report) or
        set(report)-_TOP_FIELDS-_OPTIONAL_PROVENANCE):
        _fail()
    sample=report["sample"]
    expected={"pilot":32,"full":128}.get(sample)
    if (expected is None or report["schema"]!=1 or
        report["status"]!=REPORT_STATUS or
        report["case_count"]!=expected or
        report["max_new_tokens"]!=512 or
        type(report["limitations"]) is not str or
        len(report["limitations"])>4096):
        _fail()
    required_true=(
        "post_training_developer_suite","model_only_no_tools",
        "matched_prompts","greedy_deterministic","same_eos_policy"
    )
    required_false=(
        "independent_ood_proven","raw_prompts_or_responses_uploaded",
        "training_performed","weights_promoted","release_validated"
    )
    if (any(report[k] is not True for k in required_true) or
        any(report[k] is not False for k in required_false)):
        _fail()
    models=report["models"]
    if type(models) is not dict or set(models)!=set(ARMS):
        _fail()
    for model in ARMS:
        _counts(models[model],expected)
    paired=report["paired"]
    if type(paired) is not dict or set(paired)!=set(_PAIR_ARMS):
        _fail()
    for key,(first,second) in _PAIR_ARMS.items():
        _pair(paired[key],expected,models[first]["correct"],
              models[second]["correct"])
    slices=report["slices"]
    if type(slices) is not dict or set(slices)!={"retention","transfer"}:
        _fail()
    for key in ("retention","transfer"):
        section=slices[key]
        if (type(section) is not dict or
            set(section)!={"models","paired_v8_minus_v5"} or
            type(section["models"]) is not dict or
            set(section["models"])!=set(ARMS)):
            _fail()
        for model in ARMS:
            _counts(section["models"][model],expected//2)
        _pair(section["paired_v8_minus_v5"],expected//2,
              section["models"]["v5"]["correct"],
              section["models"]["v8"]["correct"])
    for model in ARMS:
        if (models[model]["correct"] !=
            sum(slices[part]["models"][model]["correct"]
                for part in ("retention","transfer"))):
            _fail()

    gate=report["gate"]
    if (type(gate) is not dict or
        gate.get("schema")!=1 or
        gate.get("status")!="NERAL_V8_REVIEW_ONLY_REJECTION_GATE" or
        gate.get("release_validated") is not False or
        gate.get("adapter_promoted") is not False or
        gate.get("independent_ood_proven") is not False or
        type(gate.get("checks")) is not dict or
        set(gate["checks"])!=_BASE_CHECKS or
        any(type(v) is not bool for v in gate["checks"].values())):
        _fail()
    checks=gate["checks"]
    if (checks["pilot"]!=(sample=="full") or
        checks["zero_retention_regressions"]!=(
            gate.get("retention",{}).get("regressions")==0
        ) or
        checks["zero_transfer_regressions"]!=(
            gate.get("transfer",{}).get("regressions")==0
        ) or
        any(checks["zero_regressions_vs_"+name]!=(
                paired["v8_minus_"+name]["regressions"]==0
            ) for name in ("qwen","v5","v7")) or
        gate.get("eligible_for_review") is not all(checks.values()) or
        type(gate.get("failed_checks")) is not list or
        gate["failed_checks"]!=sorted(
            name for name,value in checks.items() if not value
        )):
        _fail()
    if (gate["retention"]["regressions"]!=
        slices["retention"]["paired_v8_minus_v5"]["regressions"] or
        gate["transfer"]["regressions"]!=
        slices["transfer"]["paired_v8_minus_v5"]["regressions"]):
        _fail()
    return sample,expected


def evaluate_experiment(report: dict) -> dict:
    """Derive a diagnostic next step, with no raw task or automatic promotion."""
    sample,n=_validate(report)
    paired=report["paired"]
    gate=report["gate"]
    v8=report["models"]["v8"]
    codes=[]
    if sample=="pilot":
        codes.append("PILOT_ONLY")
    for arm in ("qwen","v5","v7"):
        if paired["v8_minus_"+arm]["regressions"]:
            codes.append("REGRESSION_VS_"+arm.upper())
    if gate["retention"]["regressions"]:
        codes.append("RETENTION_REGRESSION")
    if gate["transfer"]["regressions"]:
        codes.append("TRANSFER_REGRESSION")
    if any(v8["length_censored"]>report["models"][a]["length_censored"]
           or v8["non_eos_terminated"]>report["models"][a]["non_eos_terminated"]
           for a in ("qwen","v5","v7")):
        codes.append("EOS_OR_DECODING_REGRESSION")
    if (not gate["checks"]["transfer_gain"] or
        not gate["checks"]["transfer_significance"]):
        codes.append("NO_TRANSFER_GAIN")
    if not gate["eligible_for_review"] and not codes:
        codes.append("QUALITY_GATE_FAILED")

    # A successful developer result can *only* progress to a second frozen,
    # independent holdout. It never authorizes training or shipping a LoRA.
    if "EOS_OR_DECODING_REGRESSION" in codes:
        action="FIX_COMPLETION_BUDGET_WITHOUT_TRAINING"
    elif any("REGRESSION" in c for c in codes):
        action="DIAGNOSE_NEGATIVE_TRANSFER_WITHOUT_HOLDOUT_REUSE"
    elif "NO_TRANSFER_GAIN" in codes:
        action="REVISE_TRAINING_OBJECTIVE_AND_SOURCE_DIVERSITY"
    elif sample=="pilot":
        action="FULL_MATCHED_COMPARE_WITH_UNCHANGED_WEIGHTS"
    elif codes:
        action="REVIEW_QUALITY_GATE_AND_REMEASURE"
    else:
        action="BLIND_HOLDOUT_WITH_FROZEN_WEIGHTS"
    result={
        "status":STATUS,
        "decision":("REJECT" if codes else "INDEPENDENT_REVIEW_REQUIRED"),
        "failure_codes":codes,
        "next_experiment":action,
        "case_count":n,
        "sample":sample,
        "candidate_total_correct":v8["correct"],
        "comparators":{
            arm:{
                "correct":report["models"][arm]["correct"],
                "v8_newly_correct":paired["v8_minus_"+arm]["improvements"],
                "v8_newly_wrong":paired["v8_minus_"+arm]["regressions"],
            }
            for arm in ("qwen","v5","v7")
        },
        "retention_newly_wrong_vs_v5":gate["retention"]["regressions"],
        "transfer_newly_wrong_vs_v5":gate["transfer"]["regressions"],
        "candidate_length_censored":v8["length_censored"],
        "candidate_non_eos_terminated":v8["non_eos_terminated"],
        "requires_new_blind_holdout":True,
        "independent_ood_proven":False,
        "holdout_cases_must_not_enter_training":True,
        "no_training_authorized":True,
        "release_allowed":False,
        "adapter_promoted":False,
        "baseline_preserved":True,
        "quality_claim_not_authorized":True,
    }
    return result
