from __future__ import annotations

"""Build fresh development-only corpora that target Arctus 1B weaknesses.

The three profiles deliberately vary only the source mix. They keep Neral's immutable
frontier350-v1 architecture, native tokenizer, locked source revisions, and exact-document
deduplication. They seed dedupe from the already-built balanced recovery pilot so the
fixed recovery validation set is never reused for training.

This file does not admit new sources. In particular, OASST1 remains a later SFT candidate
and code corpora with unresolved/gated licensing remain excluded.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from build_frontier_bpe_data import LocalBPE, PersistentSeen, StreamingShardWriter, build_source, sha256
from build_frontier_data import DEFAULT_SOURCES
from build_2b_focus_bpe_data import quota, seed_db
from source_lock350 import locked_revisions

START_TOKENS = 1_000_003_584
SEED_PROFILE = "arctus-2b-balanced-pilot-v1"
PROFILE_PREFIX = "arctus-weakness-recovery-v1"

PROFILES: dict[str, dict[str, float]] = {
    # Control: current balanced recovery mix.
    "control": {
        "fineweb_edu_en": 0.35,
        "fineweb_en": 0.20,
        "fineweb2_de": 0.25,
        "finemath": 0.20,
    },
    # Math is the weakest foundation category; add a modest +5 pp FineMath dose while
    # keeping science/educational content fixed.
    "math_language": {
        "fineweb_edu_en": 0.35,
        "fineweb_en": 0.15,
        "fineweb2_de": 0.25,
        "finemath": 0.25,
    },
    # Bilingual/language are also weak; shift another 5 pp from generic English toward
    # German while retaining the math boost and the full educational share.
    "bilingual_math": {
        "fineweb_edu_en": 0.35,
        "fineweb_en": 0.10,
        "fineweb2_de": 0.30,
        "finemath": 0.25,
    },
}


def specs(profile: str):
    if profile not in PROFILES:
        raise RuntimeError(f"unknown profile: {profile}")
    shares = PROFILES[profile]
    names = {s.name for s in DEFAULT_SOURCES}
    if names != set(shares):
        raise RuntimeError("weakness profile no longer matches locked source set")
    if abs(sum(shares.values()) - 1.0) > 1e-12:
        raise RuntimeError("weakness shares must sum to 1")
    return [replace(s, share=shares[s.name]) for s in DEFAULT_SOURCES]


def validate_seed_manifest(obj: dict, tokenizer: LocalBPE, revisions: dict[str, str]) -> None:
    if obj.get("schema") != 2 or obj.get("architecture") != "frontier350-v1":
        raise RuntimeError("seed corpus is not locked Frontier BPE data")
    if obj.get("corpus_role") != "foundation-pretraining":
        raise RuntimeError("seed corpus has wrong role")
    if obj.get("benchmark_content_included") is not False:
        raise RuntimeError("benchmark separation is not guaranteed")
    if obj.get("resolved_revisions") != revisions:
        raise RuntimeError("seed source revisions differ from source lock")
    if (obj.get("tokenizer") or {}).get("tokenizer_sha256") != tokenizer.contract()["tokenizer_sha256"]:
        raise RuntimeError("seed tokenizer differs from locked tokenizer")
    if obj.get("continuation_profile") != SEED_PROFILE or obj.get("pilot_only") is not True:
        raise RuntimeError("seed must be the admitted balanced recovery pilot")
    if int(obj.get("train_tokens", 0)) != 20_000_000 or int(obj.get("validation_tokens", 0)) != 200_000:
        raise RuntimeError("unexpected balanced recovery pilot size")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--seed-dedupe", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--train-tokens", type=int, default=10_000_000)
    ap.add_argument("--valid-tokens", type=int, default=100_000)
    ap.add_argument("--shard-tokens", type=int, default=5_000_000)
    args = ap.parse_args()

    if args.train_tokens != 10_000_000:
        raise SystemExit("weakness recovery v1 is locked to exactly 10M training tokens")
    if args.valid_tokens != 100_000:
        raise SystemExit("weakness recovery v1 is locked to exactly 100k local validation tokens")
    if not 1_000_000 <= args.shard_tokens <= args.train_tokens:
        raise SystemExit("invalid shard size")

    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty weakness corpus root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    tokenizer = LocalBPE(Path(args.tokenizer))
    revisions = locked_revisions(DEFAULT_SOURCES)
    seed_manifest_path = Path(args.seed_manifest)
    seed_obj = json.loads(seed_manifest_path.read_text(encoding="utf8"))
    validate_seed_manifest(seed_obj, tokenizer, revisions)
    prior_seen = seed_db(Path(args.seed_dedupe), root / "dedupe.sqlite3")

    sources = specs(args.profile)
    shares = PROFILES[args.profile]
    train_writer = StreamingShardWriter(root / "train", "train", args.shard_tokens)
    valid_writer = StreamingShardWriter(root / "valid", "valid", args.valid_tokens)
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
        "continuation_profile": f"{PROFILE_PREFIX}-{args.profile}",
        "continuation_from_tokens": START_TOKENS,
        "target_cumulative_tokens": START_TOKENS + args.train_tokens,
        "pilot_only": True,
        "weakness_targets": ["math", "language", "bilingual"],
        "preservation_targets": ["code", "factual", "reasoning", "science"],
        "tokenizer": tokenizer.contract(),
        "target_tokens": args.train_tokens + args.valid_tokens,
        "train_tokens": sum(x["tokens"] for x in train_files),
        "validation_tokens": sum(x["tokens"] for x in valid_files),
        "benchmark_content_included": False,
        "licenses": sorted({s.license for s in sources}),
        "resolved_revisions": revisions,
        "focus_shares": shares,
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
        raise RuntimeError("weakness recovery train token accounting mismatch")
    if manifest["validation_tokens"] != args.valid_tokens:
        raise RuntimeError("weakness recovery validation token accounting mismatch")
    if manifest["new_deduplicated_documents"] <= 0:
        raise RuntimeError("weakness recovery corpus contains no new documents")

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "profile": manifest["continuation_profile"],
        "train_tokens": manifest["train_tokens"],
        "validation_tokens": manifest["validation_tokens"],
        "new_documents": manifest["new_deduplicated_documents"],
        "focus_shares": shares,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
