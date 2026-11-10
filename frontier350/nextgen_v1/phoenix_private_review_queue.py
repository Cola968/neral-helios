"""Prepare a PRIVATE human-review sheet, not a training dataset or approval.

Explicit opt-in only. Original SHA-verified archive, conservative screened
candidate and tree split are read in memory; at most 32 samples with source text
are saved only to NEW files inside a non-git private asset root. No raw logs,
network, GPU, training, model weights, benchmark registry or score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# The upstream fetch_oasst2_sft50.py lives in the repository root. Standalone
# invocation must work from an unrelated working directory, not just unittest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phoenix_clean_sampler import choose_screened_diverse
from phoenix_oasst2_bridge import UPSTREAM_SHA256, load_archive, local_input, select_tree_disjoint


def private_destination(path: Path, root: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to overwrite a private review file")
    if not path.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("review destination must remain inside private asset root")
    return path


def build_review_rows(messages: list[dict], per_bucket: int = 6,
                      max_en: int = 800, max_de: int = 400) -> tuple[list[dict], dict]:
    """Deterministically sample both languages and BOTH tree-disjoint splits."""
    if not 2 <= per_bucket <= 8:
        raise ValueError("bounded per-bucket review size required")
    selected, identity = choose_screened_diverse(messages, max_en=max_en, max_de=max_de)
    splits, summary = select_tree_disjoint(selected, messages, frozen=set())
    language_by_id = {row["id"]: lang for lang in ("de", "en") for row in selected[lang]}
    if len(language_by_id) != sum(len(selected[lang]) for lang in selected):
        raise ValueError("ambiguous original language provenance")
    if not summary["split_disjoint"]:
        raise ValueError("conversation trees overlap")
    rows = []
    for split in ("train", "validation"):
        for lang in ("de", "en"):
            bucket = [row for row in splits[split] if language_by_id[row["id"]] == lang]
            if len(bucket) < per_bucket:
                raise ValueError("insufficient distinct rows in review stratum")
            bucket.sort(key=lambda item: (hashlib.sha256(("review-v1:" + item["id"]).encode()).hexdigest(), item["id"]))
            for original in bucket[:per_bucket]:
                rows.append({"id": original["id"], "split": split,
                             "language": lang, "user": original["user"],
                             "assistant": original["assistant"],
                             "privacy_review": "pending", "attribution_review": "pending",
                             "answer_quality_review": "pending",
                             "semantic_holdout_overlap_review": "pending"})
    return rows, {"status": "private_review_queue_only_not_approved",
                  "rows": len(rows), "per_language_and_split": per_bucket,
                  "source_sha256": UPSTREAM_SHA256,
                  "candidate_policy": identity["policy"],
                  "candidate_inventory_sha256": identity["selected_inventory_sha256"],
                  "tree_disjoint": True, "holdout_checked": False,
                  "all_human_reviews_pending": True, "training_approved": False,
                  "training_dataset_exported": False, "training_started": False,
                  "model_weights_modified": False, "raw_text_uploaded": False}


def prepare(root: Path, archive: Path, output: Path, receipt: Path,
            per_bucket: int = 6, max_en: int = 800, max_de: int = 400) -> dict:
    root = root.resolve(strict=True)
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("private root may not live in a git working tree")
    archive = local_input(archive, root)
    output = private_destination(output, root)
    receipt = private_destination(receipt, root)
    if output == receipt or output.suffix != ".jsonl" or receipt.suffix != ".json":
        raise ValueError("distinct JSONL review and JSON receipt required")
    rows, report = build_review_rows(load_archive(archive), per_bucket=per_bucket,
                                     max_en=max_en, max_de=max_de)
    created = []
    try:
        with output.open("x", encoding="utf-8") as handle:
            created.append(output)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["review_file_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        with receipt.open("x", encoding="utf-8") as handle:
            created.append(receipt)
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except Exception:
        for new in created:
            new.unlink(missing_ok=True)
        raise
    # Return only aggregate status, never the private paths, IDs or conversation text.
    return {"status": report["status"], "rows": report["rows"],
            "all_human_reviews_pending": True, "training_approved": False,
            "training_started": False, "raw_text_uploaded": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--assets-root", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--receipt", required=True, type=Path)
    ap.add_argument("--per-bucket", type=int, default=6)
    ap.add_argument("--max-en", type=int, default=800)
    ap.add_argument("--max-de", type=int, default=400)
    args = ap.parse_args()
    if not args.execute:
        ap.error("explicit --execute required for private text creation")
    try:
        report = prepare(args.assets_root, args.archive, args.out, args.receipt,
                         args.per_bucket, args.max_en, args.max_de)
    except Exception:
        print(json.dumps({"status": "private_review_queue_blocked", "training_approved": False,
                          "training_started": False, "raw_text_uploaded": False}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
