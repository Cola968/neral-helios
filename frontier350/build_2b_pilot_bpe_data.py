from __future__ import annotations

"""Build a small fresh balanced corpus for Arctus 1B recovery LR pilots.

This is development-only data. It uses the same locked source revisions and tokenizer as
the 2B continuation, and seeds exact deduplication from the completed 500M->1B corpus.
No benchmark/SFT content or foreign model weights are admitted.
"""

import argparse
import json
import time
from pathlib import Path

from build_frontier_bpe_data import LocalBPE, PersistentSeen, StreamingShardWriter, build_source, sha256
from build_frontier_data import DEFAULT_SOURCES
from build_2b_focus_bpe_data import SHARES, quota, seed_db, specs, validate_previous_manifest
from source_lock350 import locked_revisions

PROFILE = "arctus-2b-balanced-pilot-v1"
START_TOKENS = 1_000_003_584


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--seed-dedupe", required=True)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--valid-tokens", type=int, default=200_000)
    ap.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = ap.parse_args()

    if not 10_000_000 <= args.train_tokens <= 100_000_000:
        raise SystemExit("recovery pilot train tokens must stay within [10M,100M]")
    if not 100_000 <= args.valid_tokens <= 1_000_000:
        raise SystemExit("recovery pilot validation must stay within [100k,1M]")
    if not 1_000_000 <= args.shard_tokens <= args.train_tokens:
        raise SystemExit("invalid pilot shard size")

    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty pilot corpus root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    tokenizer = LocalBPE(Path(args.tokenizer))
    revisions = locked_revisions(DEFAULT_SOURCES)
    seed_manifest_path = Path(args.seed_manifest)
    seed_obj = json.loads(seed_manifest_path.read_text(encoding="utf8"))
    validate_previous_manifest(seed_obj, tokenizer, revisions)
    prior_seen = seed_db(Path(args.seed_dedupe), root / "dedupe.sqlite3")

    sources = specs()
    train_writer = StreamingShardWriter(root / "train", "train", args.shard_tokens)
    valid_writer = StreamingShardWriter(root / "valid", "valid", min(args.shard_tokens, args.valid_tokens))
    seen = PersistentSeen(root / "dedupe.sqlite3")
    reports: list[dict] = []
    try:
        for i, spec in enumerate(sources):
            reports.append({
                "partition": "valid",
                **build_source(spec, revisions[spec.name], quota(args.valid_tokens, sources, i), valid_writer, seen, tokenizer),
            })
        for i, spec in enumerate(sources):
            reports.append({
                "partition": "train",
                **build_source(spec, revisions[spec.name], quota(args.train_tokens, sources, i), train_writer, seen, tokenizer),
            })
        total_seen = seen.count()
    finally:
        seen.close()

    train_files = train_writer.finish()
    valid_files = valid_writer.finish()
    manifest = {
        "schema": 2,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining",
        "continuation_profile": PROFILE,
        "continuation_from_tokens": START_TOKENS,
        "target_cumulative_tokens": START_TOKENS + args.train_tokens,
        "pilot_only": True,
        "tokenizer": tokenizer.contract(),
        "target_tokens": args.train_tokens + args.valid_tokens,
        "train_tokens": sum(x["tokens"] for x in train_files),
        "validation_tokens": sum(x["tokens"] for x in valid_files),
        "benchmark_content_included": False,
        "licenses": sorted({s.license for s in sources}),
        "resolved_revisions": revisions,
        "focus_shares": SHARES,
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
        raise RuntimeError("pilot train token accounting mismatch")
    if manifest["validation_tokens"] != args.valid_tokens:
        raise RuntimeError("pilot validation token accounting mismatch")
    if manifest["new_deduplicated_documents"] <= 0:
        raise RuntimeError("pilot contains no newly admitted documents")

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "profile": PROFILE,
        "train_tokens": manifest["train_tokens"],
        "validation_tokens": manifest["validation_tokens"],
        "prior_seen_documents": prior_seen,
        "new_documents": manifest["new_deduplicated_documents"],
        "focus_shares": SHARES,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
