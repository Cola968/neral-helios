"""Arctus350: privacy-minimized, *paired* native evaluation evidence.

This module NEVER runs a model or touches a checkpoint. It compares TWO
externally supplied, per-case boolean observation manifests that claim to
come from native checkpoint inference. Such claims are NOT verified here.
The output is a review recommendation only; there is NO model promotion.

A fixed case identity/category fingerprint reduces accidental split drift,
but does NOT prove prompt identity, lack of training contamination or that
generations really came from the claimed checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

EXPECTED_TOKENIZER_SHA256 = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
SCHEMA = "arctus350-native-case-observations-v1"
_TOP_LEVEL = frozenset({
    "schema", "model_family", "evaluation_source", "benchmark_id",
    "benchmark_sha256", "prompt_template_sha256", "scoring_version",
    "tokenizer_sha256", "checkpoint_sha256", "sampling", "cases"
})
_CASE_KEYS = frozenset({"id", "category", "correct", "eos"})
_SAMPLE_KEYS = frozenset({"temperature", "top_k", "max_new_tokens"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_CATEGORY = re.compile(r"[a-z][a-z0-9_]{0,35}\Z")
_MAX_INPUT_BYTES = 1024 * 1024


class EvidenceError(ValueError):
    """Evidence is missing, mismatched, unsafe or not comparable."""


def _unique_pairs(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise EvidenceError("duplicate JSON field")
        obj[key] = value
    return obj


def _reject_nonfinite(_literal):
    raise EvidenceError("non-finite JSON numbers are forbidden")


def read_run_json(text):
    """Strict JSON parse: duplicates, NaN, infinity and root lists fail closed."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise EvidenceError("input exceeds bounded JSON size")
    try:
        obj = json.loads(text, object_pairs_hook=_unique_pairs,
                         parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise EvidenceError("malformed JSON observations") from exc
    if not isinstance(obj, dict):
        raise EvidenceError("native run must be an object")
    return obj


def _sha(value, name):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise EvidenceError(f"{name}: lowercase SHA-256 required")
    return value


def _bounded_string(value, name, pattern):
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvidenceError(f"{name}: invalid string")
    return value


def _validate(run):
    if not isinstance(run, dict) or run.keys() != _TOP_LEVEL:
        raise EvidenceError("run fields missing or raw/private extra fields present")
    if run["schema"] != SCHEMA or run["model_family"] != "neral-arctus350":
        raise EvidenceError("only native Arctus350 observations are comparable")
    if run["evaluation_source"] != "native-checkpoint-readonly":
        raise EvidenceError("tool and external-model answers are not native accuracy")
    _bounded_string(run["benchmark_id"], "benchmark_id",
                    re.compile(r"[A-Za-z0-9_.:-]{4,110}\Z"))
    _bounded_string(run["scoring_version"], "scoring_version",
                    re.compile(r"[A-Za-z0-9_.:-]{4,80}\Z"))
    for key in ("benchmark_sha256", "prompt_template_sha256", "checkpoint_sha256"):
        _sha(run[key], key)
    if run["tokenizer_sha256"] != EXPECTED_TOKENIZER_SHA256:
        raise EvidenceError("wrong native 16K BPE tokenizer")
    settings = run["sampling"]
    if not isinstance(settings, dict) or settings.keys() != _SAMPLE_KEYS:
        raise EvidenceError("missing or unknown generation settings")
    temperature = settings["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not (0 <= temperature <= 2):
        raise EvidenceError("invalid temperature")
    if type(settings["top_k"]) is not int or not 0 <= settings["top_k"] <= 256:
        raise EvidenceError("invalid top_k")
    if type(settings["max_new_tokens"]) is not int or not 1 <= settings["max_new_tokens"] <= 2048:
        raise EvidenceError("invalid token budget")
    rows = run["cases"]
    if not isinstance(rows, list) or not 8 <= len(rows) <= 512:
        raise EvidenceError("8..512 case-level observations required")
    seen = set()
    categories = Counter()
    parsed = {}
    for row in rows:
        if not isinstance(row, dict) or row.keys() != _CASE_KEYS:
            raise EvidenceError("case shape invalid; raw generations/PII forbidden")
        case_id = _bounded_string(row["id"], "id", _ID)
        category = _bounded_string(row["category"], "category", _CATEGORY)
        if case_id in seen:
            raise EvidenceError("repeated case identity")
        if type(row["correct"]) is not bool or type(row["eos"]) is not bool:
            raise EvidenceError("scores must be measured booleans")
        seen.add(case_id)
        categories[category] += 1
        parsed[case_id] = row
    if any(count < 2 for count in categories.values()):
        raise EvidenceError("at least two cases per category needed for development comparison")
    # Fingerprint the ordered canonical pair identities, not the unordered JSON input.
    specification = [
        {"id": parsed[k]["id"], "category": parsed[k]["category"]}
        for k in sorted(parsed)
    ]
    actual = hashlib.sha256(
        json.dumps(specification,sort_keys=True,separators=(",",":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if actual != run["benchmark_sha256"]:
        raise EvidenceError("frozen case identities/categories do not match fingerprint")
    return parsed


def compare_native_runs(baseline, candidate):
    """Summarize matched native claims; review eligibility != measured intelligence."""
    prior = _validate(baseline)
    current = _validate(candidate)
    if baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"]:
        raise EvidenceError("two DIFFERENT checkpoint identities required")
    if prior.keys() != current.keys():
        raise EvidenceError("heldout case set changed")
    for name in (
        "model_family", "evaluation_source", "benchmark_id", "benchmark_sha256",
        "prompt_template_sha256", "tokenizer_sha256", "scoring_version", "sampling"
    ):
        if baseline[name] != candidate[name]:
            raise EvidenceError(f"not comparable: {name}")
    overall = Counter()
    by_category = {}
    for case_id in sorted(prior):
        old, new = prior[case_id], current[case_id]
        if old["category"] != new["category"]:
            raise EvidenceError("paired category drift")
        cohort = by_category.setdefault(old["category"], Counter())
        for c in (overall, cohort):
            c["total"] += 1
            c["baseline_correct"] += int(old["correct"])
            c["candidate_correct"] += int(new["correct"])
            c["baseline_eos"] += int(old["eos"])
            c["candidate_eos"] += int(new["eos"])
            c["gained"] += int(not old["correct"] and new["correct"])
            c["lost"] += int(old["correct"] and not new["correct"])
    regression = any(c["candidate_correct"] < c["baseline_correct"]
                     for c in by_category.values())
    eos_regression = any(c["candidate_eos"] < c["baseline_eos"]
                         for c in by_category.values())
    if regression:
        decision = "REJECT_CATEGORY_REGRESSION"
    elif eos_regression:
        decision = "REJECT_EOS_REGRESSION"
    elif overall["candidate_correct"] <= overall["baseline_correct"]:
        decision = "NO_MEASURED_GAIN"
    else:
        decision = "REVIEW_CANDIDATE"
    def summary(counter):
        return {
            **{k: int(counter[k]) for k in (
                "total", "baseline_correct", "candidate_correct",
                "baseline_eos", "candidate_eos", "gained", "lost"
            )},
            "delta_correct": int(counter["candidate_correct"] - counter["baseline_correct"]),
        }
    return {
        "schema": "arctus350-native-paired-comparison-v1",
        "benchmark_id": baseline["benchmark_id"],
        "benchmark_sha256": baseline["benchmark_sha256"],
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "case_count": len(prior),
        "overall": summary(overall),
        "categories": {k: summary(v) for k, v in sorted(by_category.items())},
        "decision": decision,
        "provenance_verified": False,
        "release_authorization": "none",
        "limitations": (
            "Reported observations and checkpoint ownership are not independently verified. "
            "Case fingerprint proves ID/category consistency only, not prompt identity, sealed "
            "holdout status, contamination freedom or actual GPU inference. "
            "A review recommendation is never permission to promote, train or publish."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.baseline.resolve() == args.candidate.resolve():
        parser.error("baseline and candidate must be different files")
    # Read only; never write, train, execute input code or load model weights.
    def load(path):
        if path.stat().st_size > _MAX_INPUT_BYTES:
            raise EvidenceError("input too large")
        return read_run_json(path.read_text(encoding="utf-8"))
    report = compare_native_runs(load(args.baseline), load(args.candidate))
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
