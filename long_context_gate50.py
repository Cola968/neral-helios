"""Fail-closed long-context review gate for Neral benchmark v5.

This gate is intentionally separate from NIS and the 8/9 promotion scorecard. It prevents a
later specialization stage from being accepted merely because short-context metrics improve
while the model loses the long-context capability trained in B3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _acc(report, key):
    value = report.get(key, {}).get("accuracy")
    return None if value is None else float(value)


def _bpb(report, seq):
    value = report.get("document_bounded_true_byte_bpb", {}).get("macro_bits_per_byte", {}).get(str(seq))
    return None if value is None else float(value)


def decide(before, after, mode="protect"):
    if before.get("suite_sha256") != after.get("suite_sha256"):
        return {"proceed": False, "reason": "suite_mismatch", "severe": ["suite_sha256_mismatch"], "improvements": []}
    missing = []
    b_single, a_single = _acc(before, "single_needle"), _acc(after, "single_needle")
    b_multi, a_multi = _acc(before, "multi_needle"), _acc(after, "multi_needle")
    if None in (b_single, a_single):
        missing.append("single_needle_accuracy")
    if None in (b_multi, a_multi):
        missing.append("multi_needle_accuracy")
    if missing:
        return {"proceed": False, "reason": "missing_metrics", "missing": missing, "severe": [], "improvements": []}

    severe = []
    if a_single < b_single - 0.20:
        severe.append("single_needle_regressed_gt_20pp")
    if a_multi < b_multi - 0.25:
        severe.append("multi_needle_regressed_gt_25pp")

    bpb_rows = {}
    improvements = []
    for seq in (2048, 3072, 4095):
        b, a = _bpb(before, seq), _bpb(after, seq)
        bpb_rows[str(seq)] = {"before": b, "after": a, "relative_change": None}
        if b is None or a is None or b <= 0:
            continue
        rel = (a - b) / b
        bpb_rows[str(seq)]["relative_change"] = rel
        if rel > 0.05:
            severe.append(f"bpb_{seq}_regressed_gt_5pct")
        if rel < -0.005:
            improvements.append(f"bpb_{seq}_improved")
    if a_single > b_single:
        improvements.append("single_needle_improved")
    if a_multi > b_multi:
        improvements.append("multi_needle_improved")

    if mode == "progress":
        proceed = not severe and bool(improvements)
        reason = "pass" if proceed else "no_long_context_progress_or_regression"
    else:
        proceed = not severe
        reason = "pass" if proceed else "long_context_regression"
    return {
        "proceed": proceed,
        "mode": mode,
        "reason": reason,
        "suite_sha256": before.get("suite_sha256"),
        "single_needle": {"before": b_single, "after": a_single, "delta": a_single - b_single},
        "multi_needle": {"before": b_multi, "after": a_multi, "delta": a_multi - b_multi},
        "bpb": bpb_rows,
        "improvements": improvements,
        "severe": severe,
        "note": "Separate long-context gate; not an intelligence score and not part of NIS.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--mode", choices=("protect", "progress"), default="protect")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = decide(_load(args.before), _load(args.after), args.mode)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["proceed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
