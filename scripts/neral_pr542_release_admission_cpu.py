"""Pure scalar-only admission guard for PR542 prospective 100.

Original strict green check cannot authorize promotion until independent
raw-answer regrading with no semantic ambiguity. Never merges; no GPU/API.
"""
from __future__ import annotations
from collections import Counter

RUN="43709097701"
SOURCE="737413f03b347676108d507ff3fe7693975dd269"
SHA="74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
CATS=("mathematics","logic","programming","mixed")

def assess_release_evidence(strict:dict, secondary:dict|None)->dict:
    reasons=[]
    if (type(strict) is not dict or strict.get("github_actions_run_id")!=RUN
       or strict.get("source_commit")!=SOURCE
       or strict.get("existing_sealed_sha256")!=SHA
       or strict.get("case_count")!=100):
        reasons.append("invalid_original_receipt")
    if secondary is None or secondary.get("status") in (
        "original_raw_missing","original_raw_not_found_in_scanned_scope",
        "multiple_verified_raw_copies_needs_manual_identity_check"):
        reasons.append("raw_evidence_missing")
    elif type(secondary) is not dict:
        reasons.append("invalid_offline_receipt")
    else:
        if (secondary.get("status") not in (
                 "offline_cpu_regrade_complete","original_raw_recovered_readonly")
            or secondary.get("source_run")!=RUN
            or secondary.get("source_commit")!=SOURCE
            or secondary.get("sealed_sha256")!=SHA
            or secondary.get("case_count")!=100
            or secondary.get("gpu_calls")!=0
            or secondary.get("inference_calls")!=0
            or secondary.get("raw_outputs_uploaded") is not False):
            reasons.append("provenance_mismatch")
        cats=secondary.get("categories")
        consistent=(type(cats) is dict and set(cats)==set(CATS))
        if consistent:
            try:
                for cat in CATS:
                    row=cats[cat]
                    if (row["count"]!=25 or
                        not all(type(row[k]) is int and 0<=row[k]<=25
                                for k in ("qwen","neral","regressions"))):
                        consistent=False
                if consistent:
                    consistent=(
                       sum(cats[k]["qwen"] for k in CATS)==secondary["qwen_correct"]
                       and sum(cats[k]["neral"] for k in CATS)==secondary["neral_correct"]
                       and sum(cats[k]["regressions"] for k in CATS)==secondary["regression_count"]
                       and secondary["neral_correct"]-secondary["qwen_correct"]==secondary["net_gain"]
                    )
            except (KeyError,TypeError,ValueError):
                consistent=False
        if not consistent:reasons.append("invalid_category_accounting")
        count=secondary.get("regression_count")
        ids=secondary.get("regression_ids")
        if (type(count) is not int or count<0
            or type(ids) is not list or len(ids)!=count
            or len(set(ids))!=len(ids)):
            reasons.append("invalid_regression_evidence")
        elif count>0:
            reasons.append("corrected_regressions")
        if secondary.get("net_gain", -100)<8:
            reasons.append("insufficient_verified_net_gain")
        if (not isinstance(secondary.get("qwen_ambiguous_ids"),list)
            or not isinstance(secondary.get("neral_ambiguous_ids"),list)
            or secondary["qwen_ambiguous_ids"]
            or secondary["neral_ambiguous_ids"]):
            reasons.append("unresolved_semantic_ambiguity")
    return {
      "admission":"blocked" if reasons else "manual_review_only",
      "reasons":reasons,
      "automatic_merge_allowed":False,
      "source_run":RUN,
      "source_commit":SOURCE,
      "sealed_sha256":SHA,
      "policy":"Original strict green gate is never sufficient for a release decision",
    }
