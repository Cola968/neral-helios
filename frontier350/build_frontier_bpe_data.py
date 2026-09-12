from __future__ import annotations

"""Production corpus builder for the locked Neral 350M Frontier BPE tokenizer.

Unlike the exploratory byte builder, this path is intended for release-candidate corpora:
- source revisions come only from source_lock.json
- tokenizer file digest is recorded in the manifest
- documents are filtered and exact-deduplicated on disk
- uint16 token shards are streamed directly to disk instead of accumulating huge Python lists
- validation is built first and shares the dedupe database, guaranteeing document disjointness
"""

import argparse
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from build_frontier_data import DEFAULT_SOURCES, content_hash, extract_text, normalize, quality_ok, stream_rows
from source_lock350 import locked_revisions

SPECIALS = ["<|bos|>", "<|eos|>", "<|sep|>", "<|pad|>"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PersistentSeen:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
        self.pending = 0

    def add_if_new(self, digest: str) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (digest,))
        fresh = cur.rowcount == 1
        if fresh:
            self.pending += 1
            if self.pending >= 5000:
                self.db.commit(); self.pending = 0
        return fresh

    def count(self) -> int:
        self.db.commit()
        return int(self.db.execute("SELECT COUNT(*) FROM seen").fetchone()[0])

    def close(self) -> None:
        self.db.commit(); self.db.close()


class StreamingShardWriter:
    def __init__(self, root: Path, prefix: str, shard_tokens: int):
        if shard_tokens <= 0:
            raise ValueError("positive shard size required")
        self.root = root
        self.prefix = prefix
        self.shard_tokens = int(shard_tokens)
        self.index = 0
        self.current_tokens = 0
        self.current_path: Path | None = None
        self.handle = None
        self.files: list[dict] = []
        root.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        if self.handle is not None:
            return
        self.current_path = self.root / f"{self.prefix}_{self.index:05d}.bin"
        if self.current_path.exists():
            raise RuntimeError(f"refusing existing shard: {self.current_path}")
        self.handle = self.current_path.open("wb")
        self.current_tokens = 0

    def _close(self) -> None:
        if self.handle is None or self.current_path is None:
            return
        self.handle.flush(); self.handle.close()
        path = self.current_path
        if path.stat().st_size != self.current_tokens * 2:
            raise RuntimeError(f"uint16 byte size mismatch: {path}")
        self.files.append({"path": path.name, "tokens": self.current_tokens, "sha256": sha256(path)})
        self.index += 1
        self.current_path = None
        self.handle = None
        self.current_tokens = 0

    def add(self, ids: list[int]) -> None:
        pos = 0
        while pos < len(ids):
            self._open()
            room = self.shard_tokens - self.current_tokens
            take = min(room, len(ids) - pos)
            arr = np.asarray(ids[pos:pos + take], dtype=np.uint16)
            arr.tofile(self.handle)
            self.current_tokens += take
            pos += take
            if self.current_tokens >= self.shard_tokens:
                self._close()

    def finish(self) -> list[dict]:
        self._close()
        return list(self.files)


class LocalBPE:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()
        if self.vocab_size != 16_384:
            raise RuntimeError(f"locked Frontier BPE must have 16384 tokens, got {self.vocab_size}")
        self.special_ids = {token: self.tokenizer.token_to_id(token) for token in SPECIALS}
        if any(v is None for v in self.special_ids.values()):
            raise RuntimeError(f"missing BPE special token: {self.special_ids}")
        if len(set(self.special_ids.values())) != len(SPECIALS):
            raise RuntimeError("special token IDs are not unique")

    def encode_document(self, text: str) -> list[int]:
        body = self.tokenizer.encode(text, add_special_tokens=False).ids
        return [int(self.special_ids["<|bos|>"]), *body, int(self.special_ids["<|eos|>"])]

    def contract(self) -> dict:
        return {
            "type": "byte-bpe-v1",
            "vocab_size": self.vocab_size,
            "special_token_ids": self.special_ids,
            "tokenizer_sha256": sha256(self.path),
        }


def quota(total: int, index: int) -> int:
    if index == len(DEFAULT_SOURCES) - 1:
        return total - sum(int(total * s.share) for s in DEFAULT_SOURCES[:-1])
    return int(total * DEFAULT_SOURCES[index].share)


def build_source(spec, revision: str, target_tokens: int, writer: StreamingShardWriter, seen: PersistentSeen, tokenizer: LocalBPE) -> dict:
    if target_tokens <= 0:
        raise ValueError("positive source quota required")
    raw_rows = accepted = rejected = emitted = 0
    truncated_documents = 0
    reasons: dict[str, int] = {}
    start = time.time()
    for row in stream_rows(spec, revision):
        raw_rows += 1
        text = normalize(extract_text(spec, row))
        ok, reason = quality_ok(text)
        if not ok:
            rejected += 1; reasons[reason] = reasons.get(reason, 0) + 1; continue
        if not seen.add_if_new(content_hash(text)):
            rejected += 1; reasons["duplicate"] = reasons.get("duplicate", 0) + 1; continue
        ids = tokenizer.encode_document(text)
        remaining = target_tokens - emitted
        if len(ids) > remaining:
            # A budget boundary is not a real document ending. Preserve the actual
            # prefix, even for a 1/2-token remainder; never invent an EOS target.
            ids = ids[:remaining]
            truncated_documents += 1
        writer.add(ids)
        emitted += len(ids); accepted += 1
        if emitted >= target_tokens:
            break
        if raw_rows % 10_000 == 0:
            print(json.dumps({"source": spec.name, "rows": raw_rows, "accepted": accepted, "tokens": emitted, "target": target_tokens}), flush=True)
    if emitted != target_tokens:
        raise RuntimeError(f"source {spec.name} token quota mismatch {emitted}/{target_tokens}")
    return {
        "source": asdict(spec), "revision": revision, "raw_rows": raw_rows,
        "accepted_rows": accepted, "rejected_rows": rejected, "rejection_reasons": reasons,
        "emitted_tokens": emitted, "elapsed_s": time.time() - start,
        "budget_truncated_documents": truncated_documents,
        "synthetic_eos_targets": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--valid-tokens", type=int, required=True)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    args = ap.parse_args()
    if args.target_tokens <= args.valid_tokens or args.valid_tokens < 100_000:
        raise SystemExit("invalid train/validation token budget")

    root = Path(args.root)
    train_root, valid_root = root / "train", root / "valid"
    if root.exists() and any(root.rglob("*.bin")):
        raise RuntimeError("refusing to mix production shards with existing corpus")
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = LocalBPE(Path(args.tokenizer))
    revisions = locked_revisions(DEFAULT_SOURCES)
    seen = PersistentSeen(root / "dedupe.sqlite3")
    train_writer = StreamingShardWriter(train_root, "train", args.shard_tokens)
    valid_writer = StreamingShardWriter(valid_root, "valid", min(args.shard_tokens, args.valid_tokens))
    reports: list[dict] = []
    try:
        for i, spec in enumerate(DEFAULT_SOURCES):
            reports.append({"partition": "valid", **build_source(spec, revisions[spec.name], quota(args.valid_tokens, i), valid_writer, seen, tokenizer)})
        train_tokens = args.target_tokens - args.valid_tokens
        for i, spec in enumerate(DEFAULT_SOURCES):
            reports.append({"partition": "train", **build_source(spec, revisions[spec.name], quota(train_tokens, i), train_writer, seen, tokenizer)})
        document_count = seen.count()
    finally:
        seen.close()

    train_files, valid_files = train_writer.finish(), valid_writer.finish()
    manifest = {
        "schema": 2,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining",
        "tokenizer": tokenizer.contract(),
        "target_tokens": args.target_tokens,
        "train_tokens": sum(x["tokens"] for x in train_files),
        "validation_tokens": sum(x["tokens"] for x in valid_files),
        "benchmark_content_included": False,
        "licenses": sorted({s.license for s in DEFAULT_SOURCES}),
        "resolved_revisions": revisions,
        "sources": reports,
        "train_files": train_files,
        "valid_files": valid_files,
        "deduplicated_documents": document_count,
        "created_unix": time.time(),
    }
    if manifest["train_tokens"] + manifest["validation_tokens"] != args.target_tokens:
        raise RuntimeError("final BPE token accounting mismatch")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"root": str(root), "train_tokens": manifest["train_tokens"], "validation_tokens": manifest["validation_tokens"], "documents": document_count, "tokenizer": manifest["tokenizer"]}), flush=True)


if __name__ == "__main__":
    main()
