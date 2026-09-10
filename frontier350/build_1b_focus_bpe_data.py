from __future__ import annotations

"""Build a fresh, targeted continuation corpus for Arctus 500M -> 1B.

The continuation corpus uses only the already admitted and source-locked Frontier
sources. It seeds exact-deduplication from the existing 1.10B foundation corpus so
new shards are document-disjoint from the data already available to Arctus.

This is still foundation pretraining data: no chat/SFT examples, benchmark cases, or
foreign model weights are introduced here.
"""

import argparse
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

from build_frontier_bpe_data import (
    LocalBPE,
    PersistentSeen,
    StreamingShardWriter,
    build_source,
    sha256,
)
from build_frontier_data import DEFAULT_SOURCES
from source_lock350 import locked_revisions

FOCUS_PROFILE = "arctus-1b-focus-v1"
FOCUS_SHARES = {
    "fineweb_edu_en": 0.35,
    "fineweb_en": 0.10,
    "fineweb2_de": 0.20,
    "finemath": 0.35,
}


def focus_sources():
    names = {s.name for s in DEFAULT_SOURCES}
    if names != set(FOCUS_SHARES):
        raise RuntimeError("focus profile no longer matches locked source set")
    if abs(sum(FOCUS_SHARES.values()) - 1.0) > 1e-12:
        raise RuntimeError("focus shares must sum to 1")
    return [replace(s, share=FOCUS_SHARES[s.name]) for s in DEFAULT_SOURCES]


def quota(total: int, specs, index: int) -> int:
    if index == len(specs) - 1:
        return total - sum(int(total * s.share) for s in specs[:-1])
    return int(total * specs[index].share)


def seed_dedupe_database(source: Path, destination: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(f"seed dedupe DB missing: {source}")
    if destination.exists():
        raise RuntimeError(f"refusing existing dedupe DB: {destination}")
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
        row = dst.execute("SELECT COUNT(*) FROM seen").fetchone()
        return int(row[0])
    finally:
        dst.close()
        src.close()


def validate_seed_manifest(obj: dict, tokenizer: LocalBPE, revisions: dict[str, str]) -> None:
    if obj.get("schema") != 2 or obj.get("architecture") != "frontier350-v1":
        raise RuntimeError("seed corpus is not the locked Frontier BPE corpus")
    if obj.get("corpus_role") != "foundation-pretraining":
        raise RuntimeError("seed corpus has wrong role")
    if obj.get("benchmark_content_included") is not False:
        raise RuntimeError("seed corpus benchmark separation is not guaranteed")
    if obj.get("resolved_revisions") != revisions:
        raise RuntimeError("seed corpus revisions differ from source lock")
    if (obj.get("tokenizer") or {}).get("tokenizer_sha256") != tokenizer.contract()["tokenizer_sha256"]:
        raise RuntimeError("seed corpus tokenizer differs from current locked tokenizer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--seed-dedupe", required=True)
    ap.add_argument("--train-tokens", type=int, default=500_000_000)
    ap.add_argument("--valid-tokens", type=int, default=100_000)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000)
    args = ap.parse_args()

    if args.train_tokens != 500_000_000:
        raise SystemExit("focus-v1 is locked to exactly 500M new training tokens")
    if args.valid_tokens < 100_000 or args.valid_tokens > 5_000_000:
        raise SystemExit("focus validation must stay small and bounded")

    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty continuation corpus root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    tokenizer = LocalBPE(Path(args.tokenizer))
    revisions = locked_revisions(DEFAULT_SOURCES)
    seed_manifest_path = Path(args.seed_manifest)
    seed_obj = json.loads(seed_manifest_path.read_text(encoding="utf8"))
    validate_seed_manifest(seed_obj, tokenizer, revisions)

    prior_seen = seed_dedupe_database(Path(args.seed_dedupe), root / "dedupe.sqlite3")
    specs = focus_sources()
    train_writer = StreamingShardWriter(root / "train", "train", args.shard_tokens)
    valid_writer = StreamingShardWriter(root / "valid", "valid", min(args.shard_tokens, args.valid_tokens))
    seen = PersistentSeen(root / "dedupe.sqlite3")
    reports: list[dict] = []
    try:
        # Build a tiny continuation-local validation slice first. The actual 1B
        # comparison still uses the original frozen 50M validation partition.
        for i, spec in enumerate(specs):
            reports.append({
                "partition": "valid",
                **build_source(spec, revisions[spec.name], quota(args.valid_tokens, specs, i), valid_writer, seen, tokenizer),
            })
        for i, spec in enumerate(specs):
            reports.append({
                "partition": "train",
                **build_source(spec, revisions[spec.name], quota(args.train_tokens, specs, i), train_writer, seen, tokenizer),
            })
        total_seen = seen.count()
    finally:
        seen.close()

    train_files = train_writer.finish()
    valid_files = valid_writer.finish()
    target_tokens = args.train_tokens + args.valid_tokens
    manifest = {
        "schema": 2,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining",
        "continuation_profile": FOCUS_PROFILE,
        "continuation_from_tokens": 500_000_000,
        "tokenizer": tokenizer.contract(),
        "target_tokens": target_tokens,
        "train_tokens": sum(x["tokens"] for x in train_files),
        "validation_tokens": sum(x["tokens"] for x in valid_files),
        "benchmark_content_included": False,
        "licenses": sorted({s.license for s in specs}),
        "resolved_revisions": revisions,
        "focus_shares": FOCUS_SHARES,
        "sources": reports,
        "train_files": train_files,
        "valid_files": valid_files,
        "seed_manifest_sha256": sha256(seed_manifest_path),
        "seed_deduplicated_documents": prior_seen,
        "deduplicated_documents": total_seen,
        "new_deduplicated_documents": total_seen - prior_seen,
        "created_unix": time.time(),
    }
    if manifest["train_tokens"] != args.train_tokens:
        raise RuntimeError("continuation train token accounting mismatch")
    if manifest["validation_tokens"] != args.valid_tokens:
        raise RuntimeError("continuation validation token accounting mismatch")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "profile": FOCUS_PROFILE,
        "root": str(root),
        "train_tokens": manifest["train_tokens"],
        "validation_tokens": manifest["validation_tokens"],
        "prior_seen_documents": prior_seen,
        "new_documents": manifest["new_deduplicated_documents"],
        "focus_shares": FOCUS_SHARES,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
