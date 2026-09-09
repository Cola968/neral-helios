from __future__ import annotations

"""Build a later, disjoint Neral pretraining window from approved Hugging Face sources.

This is the scale-out companion to build_mobile_data_pack_v2.py.  It deliberately skips a
cumulative amount of *quality-filtered unique UTF-8 bytes per source* before writing anything.
A 2% safety margin is added to the skip budget so adjacent independently built windows cannot
silently overlap because an earlier window ended on a document boundary.

The output remains candidate-only.  No model weights are downloaded and no source blocked by
hf_source_policy50.py can enter the corpus.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import HfApi

import build_mobile_data_pack as core
import build_mobile_data_pack_v2 as approved
from hf_source_policy50 import assert_training_allowed

SKIP_SAFETY = 1.02


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-bytes", type=int, required=True)
    ap.add_argument("--skip-base-bytes", type=int, required=True,
                    help="Cumulative global UTF-8 byte budget represented by earlier windows")
    ap.add_argument("--shard-tokens", type=int, default=32 * 1024 * 1024)
    ap.add_argument("--max-docs-per-source", type=int, default=2_500_000)
    args = ap.parse_args()
    if args.target_bytes < 32 * 1024 * 1024:
        raise ValueError("target window is too small")
    if args.skip_base_bytes < 0:
        raise ValueError("negative skip base")

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    benchmark_hashes = core.benchmark_shingles(Path(__file__).resolve().parent)
    api = HfApi()

    resolved = []
    for spec in approved.feeder.SOURCES:
        assert_training_allowed(spec.repo)
        info = api.dataset_info(spec.repo)
        revision = str(info.sha)
        config = approved.strict_resolve_config(spec.repo, revision, spec.config_preferences)
        resolved.append((spec, revision, config))

    dedup = core.DedupDB(root / "dedup.sqlite")
    train_writer = core.ShardWriter(root, "train", args.shard_tokens)
    val_writer = core.ShardWriter(root, "val", max(1_000_000, args.shard_tokens // 4))
    documents = (root / "documents.jsonl").open("w", encoding="utf8")
    source_reports: list[dict] = []
    total_accepted = 0
    total_skipped = 0
    started = time.time()

    try:
        for spec, revision, config in resolved:
            quota = int(args.target_bytes * spec.fraction)
            # A conservative margin avoids overlap with earlier builders that stop on doc edges.
            skip_target = int(args.skip_base_bytes * spec.fraction * SKIP_SAFETY)
            accepted_bytes = accepted_docs = skipped_bytes = skipped_docs = scanned = 0
            rejected = Counter()
            print(json.dumps({
                "source": spec.name, "revision": revision, "config": config,
                "skip_target_bytes": skip_target, "write_quota_bytes": quota,
            }), flush=True)

            for row in core.stream_source(spec, revision, config):
                scanned += 1
                if scanned > args.max_docs_per_source:
                    break
                raw = core.row_text(row)
                if raw is None:
                    rejected["missing_text"] += 1
                    continue
                text = core.normalize(raw)
                reason = core.quality_reason(text, benchmark_hashes)
                if reason:
                    rejected[reason] += 1
                    continue
                digest = core.digest_text(text)
                # Add skipped docs too: this keeps exact duplicates from leaking back into the
                # writable portion of the same source or a later source in this run.
                if not dedup.add(digest):
                    rejected["exact_duplicate"] += 1
                    continue
                utf8_bytes = len(text.encode("utf8"))

                if skipped_bytes < skip_target:
                    skipped_bytes += utf8_bytes
                    skipped_docs += 1
                    total_skipped += utf8_bytes
                    continue

                if accepted_bytes and accepted_bytes + utf8_bytes > quota * 1.01:
                    rejected["quota_tail"] += 1
                    continue

                ids = core.TOKENIZER.encode(text + "\n", bos=True, eos=True)
                split = "val" if int(digest[:8], 16) % 100 == 0 else "train"
                writer = val_writer if split == "val" else train_writer
                shard, begin, end = writer.add(ids)
                documents.write(json.dumps({
                    "id": digest,
                    "source": spec.name,
                    "repo": spec.repo,
                    "revision": revision,
                    "config": config,
                    "language": spec.language,
                    "split": split,
                    "shard": shard,
                    "start": begin,
                    "end": end,
                    "tokens": len(ids),
                    "utf8_bytes": utf8_bytes,
                }, ensure_ascii=False) + "\n")
                accepted_docs += 1
                accepted_bytes += utf8_bytes
                total_accepted += utf8_bytes
                if accepted_docs % 2000 == 0:
                    print(json.dumps({
                        "source": spec.name,
                        "accepted_docs": accepted_docs,
                        "accepted_bytes": accepted_bytes,
                        "skipped_bytes": skipped_bytes,
                        "scanned": scanned,
                    }), flush=True)
                if accepted_bytes >= quota:
                    break

            source_reports.append({
                **asdict(spec),
                "revision": revision,
                "config": config,
                "skip_target_bytes": skip_target,
                "skipped_filtered_unique_bytes": skipped_bytes,
                "skipped_filtered_unique_docs": skipped_docs,
                "quota_bytes": quota,
                "accepted_utf8_bytes": accepted_bytes,
                "accepted_docs": accepted_docs,
                "scanned_docs": scanned,
                "rejected": dict(rejected),
            })
            if skipped_bytes < skip_target:
                raise RuntimeError(f"{spec.name}: could not reach disjoint skip target")
            if accepted_bytes < quota * 0.90:
                raise RuntimeError(f"{spec.name}: only {accepted_bytes}/{quota} bytes after skip")
    finally:
        documents.close()
        dedup.close()
        train_writer.close()
        val_writer.close()

    for suffix in ("", "-wal", "-shm"):
        p = root / f"dedup.sqlite{suffix}"
        if p.exists():
            p.unlink()

    manifest = {
        "version": 2,
        "name": "Neral policy-safe disjoint mobile data window",
        "candidate_only": True,
        "automatic_training_admission": False,
        "foreign_model_weights_used": False,
        "tokenizer": {
            "type": "Neral deterministic byte tokenizer",
            "vocab_size": 260,
            "dtype": "uint16-le",
            "bos": 1,
            "eos": 2,
            "byte_offset": 4,
        },
        "target_utf8_bytes": args.target_bytes,
        "skip_base_utf8_bytes": args.skip_base_bytes,
        "skip_safety_factor": SKIP_SAFETY,
        "accepted_utf8_bytes": total_accepted,
        "skipped_filtered_unique_bytes": total_skipped,
        "train_tokens": train_writer.total_tokens,
        "val_tokens": val_writer.total_tokens,
        "total_tokens": train_writer.total_tokens + val_writer.total_tokens,
        "benchmark_shingles_loaded": len(benchmark_hashes),
        "sources": source_reports,
        "train_shards": train_writer.shards,
        "val_shards": val_writer.shards,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({
        "accepted_utf8_bytes": total_accepted,
        "total_tokens": manifest["total_tokens"],
        "train_shards": len(train_writer.shards),
        "val_shards": len(val_writer.shards),
        "manifest_sha256": sha256_file(manifest_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
