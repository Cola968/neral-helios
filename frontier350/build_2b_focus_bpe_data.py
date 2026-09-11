from __future__ import annotations

"""Build the balanced Arctus 1B -> 2B foundation continuation corpus.

The builder keeps Neral's locked tokenizer/source revisions and seeds exact deduplication
from the completed 500M->1B continuation database, which itself was seeded from the
original foundation corpus. No benchmark, SFT, or foreign-model weights are introduced.
"""

import argparse
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

from build_frontier_bpe_data import LocalBPE, PersistentSeen, StreamingShardWriter, build_source, sha256
from build_frontier_data import DEFAULT_SOURCES
from source_lock350 import locked_revisions

PROFILE = "arctus-2b-balanced-v1"
SHARES = {
    "fineweb_edu_en": 0.35,
    "fineweb_en": 0.20,
    "fineweb2_de": 0.25,
    "finemath": 0.20,
}


def specs():
    names = {s.name for s in DEFAULT_SOURCES}
    if names != set(SHARES):
        raise RuntimeError("2B profile no longer matches locked source set")
    if abs(sum(SHARES.values()) - 1.0) > 1e-12:
        raise RuntimeError("2B shares must sum to 1")
    return [replace(s, share=SHARES[s.name]) for s in DEFAULT_SOURCES]


def quota(total: int, sources, index: int) -> int:
    if index == len(sources) - 1:
        return total - sum(int(total * s.share) for s in sources[:-1])
    return int(total * sources[index].share)


def seed_db(source: Path, destination: Path) -> int:
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
        dst.close(); src.close()


def validate_previous_manifest(obj: dict, tokenizer: LocalBPE, revisions: dict[str, str]) -> None:
    if obj.get("schema") != 2 or obj.get("architecture") != "frontier350-v1":
        raise RuntimeError("previous corpus is not locked Frontier BPE data")
    if obj.get("corpus_role") != "foundation-pretraining":
        raise RuntimeError("previous corpus has wrong role")
    if obj.get("benchmark_content_included") is not False:
        raise RuntimeError("benchmark separation is not guaranteed")
    if obj.get("resolved_revisions") != revisions:
        raise RuntimeError("source revisions differ from source lock")
    if (obj.get("tokenizer") or {}).get("tokenizer_sha256") != tokenizer.contract()["tokenizer_sha256"]:
        raise RuntimeError("tokenizer differs from locked tokenizer")
    if obj.get("continuation_profile") != "arctus-1b-focus-v1":
        raise RuntimeError("seed must be the admitted 500M->1B continuation corpus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--seed-dedupe", required=True)
    ap.add_argument("--train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--valid-tokens", type=int, default=200_000)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000)
    args = ap.parse_args()
    if args.train_tokens != 1_000_000_000:
        raise SystemExit("2B balanced profile is locked to exactly 1B new training tokens")
    if not 100_000 <= args.valid_tokens <= 5_000_000:
        raise SystemExit("continuation-local validation must stay bounded")

    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty continuation root: {root}")
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
    reports = []
    try:
        for i, spec in enumerate(sources):
            reports.append({"partition": "valid", **build_source(spec, revisions[spec.name], quota(args.valid_tokens, sources, i), valid_writer, seen, tokenizer)})
        for i, spec in enumerate(sources):
            reports.append({"partition": "train", **build_source(spec, revisions[spec.name], quota(args.train_tokens, sources, i), train_writer, seen, tokenizer)})
        total_seen = seen.count()
    finally:
        seen.close()

    train_files = train_writer.finish(); valid_files = valid_writer.finish()
    manifest = {
        "schema": 2,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining",
        "continuation_profile": PROFILE,
        "continuation_from_tokens": 1_000_003_584,
        "target_cumulative_tokens": 2_000_000_000,
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
        raise RuntimeError("2B continuation train token accounting mismatch")
    if manifest["validation_tokens"] != args.valid_tokens:
        raise RuntimeError("2B continuation validation token accounting mismatch")
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
