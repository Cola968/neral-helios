"""Validate exported PRIVATE review decisions against the pinned queue.

CPU-only, local files only. Never prints source text, row IDs, notes or paths.
A reviewed 24-row SAMPLE is never an approval for the other 1,176 candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from phoenix_gate import unique_object

FIELDS = (
    "privacy_review", "attribution_review", "answer_quality_review",
    "semantic_holdout_overlap_review",
)
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _read(path: Path, root: Path, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("private input must be a regular file, not a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.stat().st_size > max_bytes:
        raise ValueError("private input outside root or size limit")
    return resolved.read_bytes()


def _json(raw: bytes):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate(queue_bytes: bytes, receipt_bytes: bytes, decision_bytes: bytes,
             expected_queue_sha: str) -> dict:
    if not isinstance(expected_queue_sha, str) or not HEX.fullmatch(expected_queue_sha):
        raise ValueError("invalid pinned queue digest")
    if hashlib.sha256(queue_bytes).hexdigest() != expected_queue_sha:
        raise ValueError("original review queue digest mismatch")
    original = _json(receipt_bytes)
    if not isinstance(original, dict) or (original.get("review_file_sha256") != expected_queue_sha
        or original.get("rows") != 24 or original.get("all_human_reviews_pending") is not True
        or original.get("training_approved") is not False
        or original.get("training_started") is not False):
        raise ValueError("untrusted original queue receipt")
    queue = [_json(line) for line in queue_bytes.splitlines() if line.strip()]
    if len(queue) != 24:
        raise ValueError("review population changed")
    ids = set()
    for item in queue:
        if not isinstance(item, dict) or set(item) != {"id", "split", "language", "user", "assistant", *FIELDS}:
            raise ValueError("invalid source review row schema")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
            raise ValueError("duplicate or invalid source row")
        if any(item[field] != "pending" for field in FIELDS):
            raise ValueError("source review queue was modified")
        ids.add(item["id"])
    payload = _json(decision_bytes)
    if (not isinstance(payload, dict) or set(payload) != {"version", "queue_sha256", "training_approved", "rows"}
        or type(payload["version"]) is not int or payload["version"] != 1
        or payload["queue_sha256"] != expected_queue_sha or payload["training_approved"] is not False
        or not isinstance(payload["rows"], list) or len(payload["rows"]) != 24):
        raise ValueError("untrusted exported decision payload")
    seen = set()
    counts = {field: {"pass": 0, "reject": 0} for field in FIELDS}
    for row in payload["rows"]:
        if not isinstance(row, dict) or set(row) != {"id", "decisions", "note"}:
            raise ValueError("invalid exported review row")
        ident = row["id"]
        if not isinstance(ident, str) or ident not in ids or ident in seen:
            raise ValueError("unknown or duplicate review row")
        seen.add(ident)
        choices = row["decisions"]
        if not isinstance(choices, dict) or set(choices) != set(FIELDS):
            raise ValueError("missing, extra or incomplete review decisions")
        if not isinstance(row["note"], str) or len(row["note"]) > 1000:
            raise ValueError("invalid or oversized private note")
        for field in FIELDS:
            choice = choices[field]
            if choice not in ("pass", "reject") or not isinstance(choice, str):
                raise ValueError("incomplete or invalid decision")
            counts[field][choice] += 1
    if seen != ids:
        raise ValueError("review sample coverage mismatch")
    return {
        "schema": 1, "status": "sample_decisions_validated_not_training_approved",
        "queue_sha256": expected_queue_sha,
        "decisions_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "review_sample_rows": 24, "field_counts": counts,
        "all_sample_decisions_present": True,
        "entire_training_corpus_reviewed": False,
        "independent_holdout_checked": False,
        "training_approved": False, "training_started": False,
        "raw_text_uploaded": False,
    }


def check(root: Path, queue: Path, receipt: Path, decisions: Path, output: Path,
          expected_queue_sha: str) -> dict:
    root = root.resolve(strict=True)
    if any((part / ".git").exists() for part in (root, *root.parents)):
        raise ValueError("private assets must remain outside git")
    if output.exists() or output.is_symlink() or not output.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("refusing overwrite or out-of-root output")
    if len({queue.resolve(), receipt.resolve(), decisions.resolve(), output.resolve()}) != 4:
        raise ValueError("all paths must be distinct")
    report = validate(_read(queue, root, 256_000), _read(receipt, root, 32_000),
                      _read(decisions, root, 128_000), expected_queue_sha)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return {"status": report["status"], "review_sample_rows": 24,
            "all_sample_decisions_present": True, "training_approved": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queue-sha256", required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(check(args.assets_root, args.queue, args.receipt, args.decisions,
                               args.out, args.queue_sha256), sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "private_review_decisions_blocked", "training_approved": False,
                          "training_started": False, "raw_text_logged": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
