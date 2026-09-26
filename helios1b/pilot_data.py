from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .train_tokenizer_pinned import (
    SOURCES,
    content_hash,
    normalize,
    quality_ok,
    stream_rows,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_targets(total: int) -> dict[str, int]:
    result: dict[str, int] = {}
    assigned = 0
    for i, spec in enumerate(SOURCES):
        value = int(total * spec.share)
        if i == len(SOURCES) - 1:
            value = total - assigned
        assigned += value
        result[spec.name] = value
    return result


def append_limited(dst: list[int], ids: list[int], target: int) -> int:
    remaining = target - len(dst)
    if remaining <= 0:
        return 0
    take = min(remaining, len(ids))
    dst.extend(ids[:take])
    return take


def build(
    tokenizer_path: Path,
    out_dir: Path,
    train_tokens: int,
    val_tokens: int,
) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 32_768:
        raise ValueError("pilot requires exact Helios 32K tokenizer")
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("Helios tokenizer is missing <eos>")

    train_targets = source_targets(train_tokens)
    val_targets = source_targets(val_tokens)
    train: list[int] = []
    val: list[int] = []
    global_seen: set[str] = set()
    stats: dict[str, dict] = {}

    for spec in SOURCES:
        train_source: list[int] = []
        val_source: list[int] = []
        accepted_train_docs = 0
        accepted_val_docs = 0
        rejected = 0

        for row in stream_rows(spec):
            raw = row.get("text", "")
            text = normalize(raw if isinstance(raw, str) else str(raw or ""))
            if not quality_ok(text):
                rejected += 1
                continue
            digest = content_hash(text)
            if digest in global_seen:
                rejected += 1
                continue
            global_seen.add(digest)

            ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos]
            holdout = int(digest[:8], 16) % 10 == 0

            if holdout and len(val_source) < val_targets[spec.name]:
                if append_limited(
                    val_source, ids, val_targets[spec.name]
                ):
                    accepted_val_docs += 1
            elif len(train_source) < train_targets[spec.name]:
                if append_limited(
                    train_source, ids, train_targets[spec.name]
                ):
                    accepted_train_docs += 1
            elif len(val_source) < val_targets[spec.name]:
                if append_limited(
                    val_source, ids, val_targets[spec.name]
                ):
                    accepted_val_docs += 1

            if (
                len(train_source) >= train_targets[spec.name]
                and len(val_source) >= val_targets[spec.name]
            ):
                break

        if len(train_source) != train_targets[spec.name]:
            raise RuntimeError(
                f"{spec.name}: train shortfall "
                f"{len(train_source)}/{train_targets[spec.name]}"
            )
        if len(val_source) != val_targets[spec.name]:
            raise RuntimeError(
                f"{spec.name}: val shortfall "
                f"{len(val_source)}/{val_targets[spec.name]}"
            )

        train.extend(train_source)
        val.extend(val_source)
        stats[spec.name] = {
            "train_tokens": len(train_source),
            "val_tokens": len(val_source),
            "train_documents": accepted_train_docs,
            "val_documents": accepted_val_docs,
            "rejected_or_duplicate_rows": rejected,
            "revision": spec.revision,
            "share": spec.share,
        }

    if len(train) != train_tokens or len(val) != val_tokens:
        raise RuntimeError("exact pilot token targets were not met")
    if max(train + val) >= 32_768:
        raise RuntimeError("pilot token id exceeds Helios vocabulary")

    out_dir.mkdir(parents=True, exist_ok=False)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    np.asarray(train, dtype="<u2").tofile(train_path)
    np.asarray(val, dtype="<u2").tofile(val_path)

    manifest = {
        "schema": 1,
        "dataset": "helios1b-realdata-pilot-v1",
        "tokenizer_sha256": sha256(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": len(train),
        "val_tokens": len(val),
        "dtype": "uint16",
        "train_sha256": sha256(train_path),
        "val_sha256": sha256(val_path),
        "unique_documents_seen": len(global_seen),
        "split_policy": (
            "Exact-content SHA256 dedupe globally; hash-modulo holdout; "
            "source-proportional exact token quotas; no train/val overlap."
        ),
        "sources": stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tokens", type=int, default=1_100_000)
    ap.add_argument("--val-tokens", type=int, default=100_000)
    args = ap.parse_args()
    if args.train_tokens < 1_000_000 or args.val_tokens < 50_000:
        raise SystemExit("pilot data budget too small")
    build(
        Path(args.tokenizer),
        Path(args.out),
        args.train_tokens,
        args.val_tokens,
    )


if __name__ == "__main__":
    main()
