"""Non-approving, CPU-only triage of an immutable private Phoenix candidate pack.

Creates a SEPARATE filtered working copy and local-only quarantine. A heuristic
flag is not proof of an incorrect answer. Never edits the original, approves
training, reads checkpoints, or prints text, IDs or private paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from phoenix_candidate_pack import _repeat_ratio
from sft_contract import norm

SOURCE_REPORT_SHA = "41f6c69bc1f624f43cf9aadb309647535b20cf8f6b041ee57999bca3aa28c6ab"
REASONS = ("duplicate_answer", "high_trigram_repetition")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def private_input(path: Path, root: Path, cap: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing, symlinked or non-file input")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.stat().st_size > cap:
        raise ValueError("input outside private root or size limit")
    return resolved.read_bytes()


def parse_lines(raw: bytes, split: str) -> list[dict]:
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError("empty candidate line")
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"id", "source_id", "user", "assistant"}:
            raise ValueError("unexpected candidate row schema")
        if any(not isinstance(value, str) or not value for value in row.values()):
            raise ValueError("missing candidate field")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty {split} candidates")
    return rows


def triage(root: Path, source: Path, out: Path, expected_report_sha: str = SOURCE_REPORT_SHA) -> dict:
    root = root.resolve(strict=True)
    if any((part / ".git").exists() for part in (root, *root.parents)):
        raise ValueError("private output may not be in Git")
    source = source.resolve(strict=True)
    if not source.is_relative_to(root) or source.is_symlink() or not source.is_dir():
        raise ValueError("candidate source must be a private directory")
    if (out.exists() or out.is_symlink() or not out.parent.resolve(strict=True).is_relative_to(root)
            or out.resolve() == source or source.is_relative_to(out.resolve())
            or out.resolve().is_relative_to(source)):
        raise ValueError("new nonnested output required; no overwrite")
    if expected_report_sha != SOURCE_REPORT_SHA:
        raise ValueError("unrecognized original report")
    report_bytes = private_input(source / "candidate_report.json", root, 100_000)
    if digest(report_bytes) != SOURCE_REPORT_SHA:
        raise ValueError("original candidate report changed")
    report = json.loads(report_bytes)
    if (report.get("classification") != "private_candidate_only_not_training_pack"
            or report.get("training_approved") is not False
            or report.get("holdout_checked") is not False
            or report.get("audit", {}).get("rows") != 1200
            or report.get("audit", {}).get("flags") != {"duplicate_answer": 1, "high_trigram_repetition": 7}):
        raise ValueError("unexpected candidate provenance/status")
    raw = {split: private_input(source / f"{split}.jsonl", root, 4_000_000)
           for split in ("train", "validation")}
    for split in raw:
        if digest(raw[split]) != report.get(f"{split}_sha256"):
            raise ValueError("candidate split hash mismatch")
    splits = {split: parse_lines(raw[split], split) for split in raw}
    if (len(splits["train"]) != 1094 or len(splits["validation"]) != 106):
        raise ValueError("candidate population differs from original")
    clean = {split: [] for split in splits}
    quarantine = []
    reason_totals = Counter()
    seen_answers = set()
    seen_ids = set()
    for split in ("train", "validation"):
        for row in splits[split]:
            if row["id"] in seen_ids:
                raise ValueError("duplicate candidate ID")
            seen_ids.add(row["id"])
            answer_norm = norm(row["assistant"])
            reasons = []
            if answer_norm in seen_answers:
                reasons.append("duplicate_answer")
            seen_answers.add(answer_norm)
            if _repeat_ratio(row["assistant"]) > 0.30:
                reasons.append("high_trigram_repetition")
            if reasons:
                quarantine.append({"split": split, "row": row, "reasons": reasons})
                reason_totals.update(reasons)
            else:
                clean[split].append(row)
    if dict(reason_totals) != report["audit"]["flags"]:
        raise ValueError("flagging is not identical to pinned original audit")
    if not clean["train"] or not clean["validation"]:
        raise ValueError("triage emptied a split")
    # An output directory is created only after all source and flag checks succeed.
    out.mkdir(exist_ok=False)
    created = []
    try:
        hashes = {}
        for name, rows in (("train", clean["train"]), ("validation", clean["validation"]),
                           ("quarantine", quarantine)):
            path = out / f"{name}.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            hashes[name] = digest(path.read_bytes())
        manifest = {"schema": 1, "classification": "triaged_working_copy_not_training_pack",
                    "source_report_sha256": SOURCE_REPORT_SHA, "original_rows": 1200,
                    "train_rows": len(clean["train"]), "validation_rows": len(clean["validation"]),
                    "quarantine_rows": len(quarantine), "flag_counts": dict(sorted(reason_totals.items())),
                    "output_sha256": hashes, "holdout_checked": False,
                    "content_correctness_verified": False, "license_approved": False,
                    "privacy_approved": False, "human_review_completed": False,
                    "training_approved": False, "training_started": False,
                    "raw_text_uploaded": False,
                    "note": "Quarantine is a HEURISTIC review list, not proof of wrong answers; never authorize SFT from this copy."}
        path = out / "triage_manifest.json"
        with path.open("x", encoding="utf-8") as handle:
            created.append(path)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        for item in created:
            item.unlink(missing_ok=True)
        out.rmdir()
        raise
    return {"status": "local_triage_only_not_approved", "original_rows": 1200,
            "train_rows": len(clean["train"]), "validation_rows": len(clean["validation"]),
            "quarantine_rows": len(quarantine), "flag_counts": dict(sorted(reason_totals.items())),
            "manifest_sha256": digest((out / "triage_manifest.json").read_bytes()),
            "training_approved": False, "raw_text_uploaded": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets-root", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    try:
        print(json.dumps(triage(a.assets_root, a.candidate, a.out), sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "triage_blocked", "training_approved": False,
                          "training_started": False, "raw_text_uploaded": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
