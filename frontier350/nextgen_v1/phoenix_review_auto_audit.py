"""CPU-only local structural audit of the 24-row PRIVATE Phoenix review queue.

This is NOT a substitute for examining content, license, factual correctness or
holdout leakage. It never outputs source text, IDs or private paths to stdout.
Creates exactly one new private report; never edits review decisions or weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from phoenix_gate import unique_object

PIN = "94053e24e34b309ced74cf0ecdf75db484b4168718a260039ae3b7931c579f81"
FIELDS = ("privacy_review", "attribution_review", "answer_quality_review", "semantic_holdout_overlap_review")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\d[ .-]?){10,14}(?!\w)")
ROLE = re.compile(r"(?im)^\s*(?:user|nutzer|assistant|system)\s*:")


def read_private(path: Path, root: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing, nonregular or symlinked input")
    target = path.resolve(strict=True)
    if not target.is_relative_to(root) or target.stat().st_size > limit:
        raise ValueError("private input outside root or size bound")
    return target.read_bytes()


def audit(queue_bytes: bytes, receipt_bytes: bytes, expected_sha: str = PIN) -> dict:
    if expected_sha != PIN or hashlib.sha256(queue_bytes).hexdigest() != PIN:
        raise ValueError("queue identity differs from original pin")
    receipt = json.loads(receipt_bytes, object_pairs_hook=unique_object)
    if not isinstance(receipt, dict) or (receipt.get("review_file_sha256") != PIN
            or receipt.get("rows") != 24 or receipt.get("all_human_reviews_pending") is not True
            or receipt.get("training_approved") is not False):
        raise ValueError("queue receipt not original pending sample")
    rows = [json.loads(line, object_pairs_hook=unique_object) for line in queue_bytes.splitlines() if line.strip()]
    if len(rows) != 24:
        raise ValueError("review population changed")
    ids = set()
    prompts, answers, buckets = set(), set(), Counter()
    flagged = []
    totals = Counter()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "split", "language", "user", "assistant", *FIELDS}:
            raise ValueError("review row schema changed")
        ident = row["id"]
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("invalid or duplicated review row id")
        ids.add(ident)
        if (row["split"], row["language"]) not in {("train", "de"), ("train", "en"), ("validation", "de"), ("validation", "en")}:
            raise ValueError("invalid review stratum")
        buckets[row["split"] + "_" + row["language"]] += 1
        if any(row[field] != "pending" for field in FIELDS):
            raise ValueError("review decisions already modified")
        question, answer = row["user"], row["assistant"]
        if not isinstance(question, str) or not isinstance(answer, str) or len(question) > 8192 or len(answer) > 8192:
            raise ValueError("invalid private text")
        qnorm, anorm = " ".join(question.casefold().split()), " ".join(answer.casefold().split())
        reasons = []
        if not qnorm or not anorm:
            reasons.append("empty_text")
        if qnorm in prompts:
            reasons.append("repeated_prompt")
        if anorm in answers:
            reasons.append("repeated_answer")
        prompts.add(qnorm)
        answers.add(anorm)
        if EMAIL.search(question + "\n" + answer):
            reasons.append("possible_email")
        if PHONE.search(question + "\n" + answer):
            reasons.append("possible_phone")
        if ROLE.search(answer):
            reasons.append("answer_role_marker")
        if len(question) > 512:
            reasons.append("long_prompt_similarity_incomplete")
        if len(answer) < 40 or len(answer) > 2500:
            reasons.append("answer_length_outlier")
        if len(qnorm) >= 35 and qnorm == anorm:
            reasons.append("answer_exact_prompt_echo")
        words = re.findall(r"\w+", answer.casefold())
        grams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
        if len(grams) >= 12 and (len(grams)-len(set(grams))) / len(grams) > 0.30:
            reasons.append("answer_repetition_hint")
        for reason in reasons:
            totals[reason] += 1
        if reasons:
            # IDs and flags stay on local D: ONLY. Never print these records.
            flagged.append({"id": ident, "flags": reasons})
    if dict(buckets) != {"train_de": 6, "train_en": 6, "validation_de": 6, "validation_en": 6}:
        raise ValueError("review population not stratified as frozen")
    return {"schema": 1, "status": "automated_review_hints_only_not_human_review",
            "queue_sha256": PIN, "queue_rows": 24, "strata": dict(sorted(buckets.items())),
            "flagged_rows": len(flagged), "flag_counts": dict(sorted(totals.items())),
            "private_flags": flagged, "human_content_review_completed": False,
            "source_license_verified": False, "factual_accuracy_verified": False,
            "holdout_semantic_overlap_verified": False, "training_approved": False,
            "training_started": False, "raw_text_uploaded": False}


def run(root: Path, queue: Path, receipt: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    if any((directory / ".git").exists() for directory in (root, *root.parents)):
        raise ValueError("private root cannot be inside git tree")
    if (output.exists() or output.is_symlink() or not output.parent.resolve(strict=True).is_relative_to(root)):
        raise ValueError("new in-root output required")
    if output.resolve() in {queue.resolve(), receipt.resolve()}:
        raise ValueError("report must not replace original queue or receipt")
    report = audit(read_private(queue, root, 256_000), read_private(receipt, root, 32_000), PIN)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {key: report[key] for key in ("status", "queue_rows", "flagged_rows", "flag_counts",
                                           "human_content_review_completed", "training_approved", "raw_text_uploaded")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.assets_root, args.queue, args.receipt, args.out), sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "automated_review_blocked", "training_approved": False,
                          "training_started": False, "raw_text_uploaded": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
