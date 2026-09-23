from __future__ import annotations

"""Build a DEVELOPMENT-ONLY BPE foundation refresh corpus for the current Neral 350M anchor.

This is deliberately separate from the historical 1B continuation corpora. It uses
only the existing immutable source lock and the exact 16,384-token Neral BPE.

The builder does NOT claim that documents are novel versus the historical 2.073B
training stream because a complete historical document-hash registry is unavailable.
It DOES guarantee exact document dedupe within this new build and document-disjoint
validation/train partitions by building validation first with one shared dedupe DB.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from build_frontier_data import SourceSpec
from build_frontier_bpe_data import (
    LocalBPE,
    PersistentSeen,
    StreamingShardWriter,
    build_source,
)
from source_lock350 import LOCK_PATH, locked_revisions

CURRENT_ANCHOR_TOKENS = 2_073_161_216
CURRENT_ANCHOR_SHA256 = "f7a73e283421975acc34e13afc38378be4eab11da34e7cb0cda883e7"
# Correct full SHA kept separately below to make accidental abbreviation impossible.
CURRENT_ANCHOR_SHA256_FULL = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
EXPECTED_TOKENIZER_SHA256 = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"

# Quality-first refresh: less generic web, more educational, German and mathematics.
# All sources already exist in frontier350/source_lock.json at immutable revisions.
REFRESH_SOURCES = (
    SourceSpec("fineweb_edu_en", "HuggingFaceFW/fineweb-edu", "sample-10BT",
               "train", "odc-by", "text", 0.40, "en", "education"),
    SourceSpec("fineweb_en", "HuggingFaceFW/fineweb", "sample-10BT",
               "train", "odc-by", "text", 0.10, "en", "general"),
    SourceSpec("fineweb2_de", "HuggingFaceFW/fineweb-2", "deu_Latn",
               "train", "odc-by", "text", 0.25, "de", "general"),
    SourceSpec("finemath", "HuggingFaceTB/finemath", "finemath-3plus",
               "train", "odc-by", "text", 0.25, "multi", "math"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quota(total: int, index: int) -> int:
    if index == len(REFRESH_SOURCES) - 1:
        return total - sum(int(total * s.share) for s in REFRESH_SOURCES[:-1])
    return int(total * REFRESH_SOURCES[index].share)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--valid-tokens", type=int, default=500_000)
    ap.add_argument("--shard-tokens", type=int, default=5_000_000)
    args = ap.parse_args()

    if not 1_000_000 <= args.train_tokens <= 100_000_000:
        raise SystemExit("development refresh train budget must be 1M..100M tokens")
    if not 100_000 <= args.valid_tokens <= min(5_000_000, args.train_tokens // 2):
        raise SystemExit("invalid development validation budget")
    if abs(sum(s.share for s in REFRESH_SOURCES) - 1.0) > 1e-12:
        raise RuntimeError("refresh shares must sum to one")

    root = args.root.absolute()
    if root.exists():
        raise RuntimeError("refresh output must be create-only")
    tokenizer = LocalBPE(args.tokenizer.resolve(strict=True))
    token_contract = tokenizer.contract()
    if token_contract["tokenizer_sha256"] != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("current Neral tokenizer SHA mismatch")

    revisions = locked_revisions(REFRESH_SOURCES)
    source_lock_sha = sha256(LOCK_PATH)
    root.mkdir(parents=True, exist_ok=False)
    train_root, valid_root = root / "train", root / "valid"
    train_writer = StreamingShardWriter(train_root, "train", args.shard_tokens)
    valid_writer = StreamingShardWriter(valid_root, "valid", min(args.shard_tokens, args.valid_tokens))
    seen = PersistentSeen(root / "dedupe.sqlite3")
    reports = []
    try:
        # Build validation first, then training through the SAME exact-dedupe database.
        for i, spec in enumerate(REFRESH_SOURCES):
            reports.append({
                "partition": "validation",
                **build_source(spec, revisions[spec.name], quota(args.valid_tokens, i),
                               valid_writer, seen, tokenizer),
            })
        for i, spec in enumerate(REFRESH_SOURCES):
            reports.append({
                "partition": "train",
                **build_source(spec, revisions[spec.name], quota(args.train_tokens, i),
                               train_writer, seen, tokenizer),
            })
        documents = seen.count()
    finally:
        seen.close()

    valid_files = valid_writer.finish()
    train_files = train_writer.finish()
    actual_train = sum(row["tokens"] for row in train_files)
    actual_valid = sum(row["tokens"] for row in valid_files)
    if actual_train != args.train_tokens or actual_valid != args.valid_tokens:
        raise RuntimeError("refresh token accounting mismatch")

    manifest = {
        "schema": 1,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining-refresh-development",
        "classification": "DEVELOPMENT_ONLY_NOT_PRODUCTION_APPROVED",
        "continuation_profile": "phoenix-quality-refresh-v1",
        "continuation_from_tokens": CURRENT_ANCHOR_TOKENS,
        "continuation_from_checkpoint_sha256": CURRENT_ANCHOR_SHA256_FULL,
        "new_train_tokens": actual_train,
        "validation_tokens": actual_valid,
        "target_cumulative_tokens_if_fully_consumed": CURRENT_ANCHOR_TOKENS + actual_train,
        "tokenizer": token_contract,
        "source_lock_sha256": source_lock_sha,
        "source_mix": {s.name: s.share for s in REFRESH_SOURCES},
        "resolved_revisions": revisions,
        "licenses": sorted({s.license for s in REFRESH_SOURCES}),
        "sources": reports,
        "train_files": train_files,
        "valid_files": valid_files,
        "deduplicated_documents": documents,
        "validation_built_first_shared_dedupe": True,
        "train_validation_exact_document_disjoint": True,
        "benchmark_content_intentionally_included": False,
        "historical_2b_document_overlap_checked": False,
        "historical_overlap_warning": (
            "A complete historical document-hash registry is unavailable; do not claim "
            "novelty versus the 2.073B-token anchor."
        ),
        "independent_holdout_checked": False,
        "human_source_review_complete": False,
        "production_eligible": False,
        "created_unix": time.time(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "event": "phoenix_refresh_corpus_built",
        "manifest_sha256": sha256(manifest_path),
        "continuation_from_tokens": CURRENT_ANCHOR_TOKENS,
        "train_tokens": actual_train,
        "validation_tokens": actual_valid,
        "target_cumulative_tokens_if_fully_consumed": CURRENT_ANCHOR_TOKENS + actual_train,
        "documents": documents,
        "source_mix": manifest["source_mix"],
        "resolved_revisions": revisions,
        "train_shards": len(train_files),
        "validation_shards": len(valid_files),
        "historical_overlap_checked": False,
        "human_source_review_complete": False,
        "production_eligible": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
