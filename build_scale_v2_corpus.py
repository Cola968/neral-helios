from __future__ import annotations

"""Streaming Hugging Face -> local Neral byte-token shard packer.

This tool is intentionally data-plane only. It never loads model weights and never uploads raw
corpora. It is designed for multi-billion-byte-token preparation where keeping every source row
in RAM would be unacceptable.

Example:
  python build_scale_v2_corpus.py \
    --dataset HuggingFaceFW/fineweb-2 --config <de-config> --split train \
    --language de --out D:/neral_data/fineweb2_de --target-tokens 1000000000

The exact config is explicit rather than guessed; dataset revisions/configs must first pass the
project's fail-closed admission/revision audit.
"""

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

from tokenizer import ByteTokenizer


CONTACT = re.compile(r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:api[_ -]?key|password|secret|token)\s*[:=])", re.I)
REPEATED = re.compile(r"(.)\1{20,}", re.S)
WS = re.compile(r"[ \t]+")


def clean_text(value) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, "not_text"
    text = unicodedata.normalize("NFC", value).replace("\x00", "")
    text = "\n".join(WS.sub(" ", line).strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line)
    if len(text) < 300:
        return None, "too_short"
    if len(text) > 2_000_000:
        return None, "too_long"
    if CONTACT.search(text):
        return None, "contact_or_secret_pattern"
    if REPEATED.search(text):
        return None, "pathological_repetition"
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    if printable / max(1, len(text)) < 0.98:
        return None, "controls"
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(1, len(text)) < 0.20:
        return None, "low_alpha_ratio"
    lines = text.splitlines()
    if len(lines) >= 8:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.55:
            return None, "repeated_lines"
    return text.strip(), None


def normalized_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf8")).hexdigest()


class ExactDedup:
    """Disk-backed exact normalized-hash index; bounded RAM, resume-safe."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
        self.pending = 0

    def add_if_new(self, digest: str) -> bool:
        before = self.db.total_changes
        self.db.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (digest,))
        fresh = self.db.total_changes > before
        if fresh:
            self.pending += 1
            if self.pending >= 5000:
                self.db.commit()
                self.pending = 0
        return fresh

    def close(self):
        self.db.commit()
        self.db.close()


class ShardWriter:
    def __init__(self, root: Path, shard_tokens: int):
        self.root = root
        self.shard_tokens = int(shard_tokens)
        self.index = 0
        self.count = 0
        self.total = 0
        self.handle = None
        self.path = None
        self._open()

    def _open(self):
        if self.handle:
            self.handle.close()
        self.path = self.root / f"tokens-{self.index:05d}.bin"
        self.handle = self.path.open("ab")
        self.count = self.path.stat().st_size // 2 if self.path.exists() else 0

    def append(self, ids: list[int]) -> tuple[str, int, int]:
        if self.count and self.count + len(ids) > self.shard_tokens:
            self.index += 1
            self._open()
        start = self.count
        values = array("H", ids)
        if values.itemsize != 2:
            raise RuntimeError("uint16 array required")
        if os.sys.byteorder != "little":
            values.byteswap()
        values.tofile(self.handle)
        self.count += len(ids)
        self.total += len(ids)
        return self.path.name, start, self.count

    def close(self):
        if self.handle:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", required=True, help="Immutable audited Hub revision")
    ap.add_argument("--language", required=True, choices=("de", "en", "code", "math"))
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--shard-tokens", type=int, default=250_000_000)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()

    if args.target_tokens <= 0 or args.shard_tokens <= 0:
        raise ValueError("positive token budgets required")
    if len(args.revision) < 12:
        raise ValueError("revision must be an immutable audited commit, not a floating branch")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-data.txt / datasets before corpus preparation") from exc

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    docs_path = out / "documents.jsonl"
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a completed corpus manifest")

    tok = ByteTokenizer()
    dedup = ExactDedup(out / "exact_dedup.sqlite3")
    writer = ShardWriter(out, args.shard_tokens)
    rejected: dict[str, int] = {}
    accepted = 0
    source_rows = 0
    source_bytes = 0
    written_tokens = 0
    started = time.time()

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )

    try:
        with docs_path.open("a", encoding="utf8") as docs:
            for row in dataset:
                source_rows += 1
                if args.max_docs and source_rows > args.max_docs:
                    break
                raw = row.get(args.text_field)
                text, reason = clean_text(raw)
                if text is None:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                source_bytes += len(text.encode("utf8"))
                digest = normalized_hash(text)
                if not dedup.add_if_new(digest):
                    rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
                    continue
                ids = tok.encode(text, bos=True, eos=True)
                if len(ids) < 302:
                    rejected["short_after_tokenize"] = rejected.get("short_after_tokenize", 0) + 1
                    continue
                if written_tokens + len(ids) > args.target_tokens:
                    # Keep document boundaries intact; do not truncate the final document.
                    break
                shard, start, end = writer.append(ids)
                docs.write(json.dumps({
                    "source": args.dataset,
                    "config": args.config,
                    "revision": args.revision,
                    "split": args.split,
                    "language": args.language,
                    "sha256_normalized": digest,
                    "shard": shard,
                    "start": start,
                    "end": end,
                    "tokens": len(ids),
                }, ensure_ascii=False) + "\n")
                accepted += 1
                written_tokens += len(ids)
                if accepted % 5000 == 0:
                    docs.flush()
                    print(json.dumps({
                        "accepted_docs": accepted,
                        "source_rows": source_rows,
                        "tokens": written_tokens,
                        "target_tokens": args.target_tokens,
                        "elapsed_s": time.time() - started,
                    }), flush=True)
                if written_tokens >= args.target_tokens:
                    break
    finally:
        writer.close()
        dedup.close()

    shards = []
    for p in sorted(out.glob("tokens-*.bin")):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        shards.append({"file": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()})

    manifest = {
        "schema": 1,
        "format": "Neral Scale V2 document-bounded uint16 byte-token shards",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "revision": args.revision,
        "language": args.language,
        "text_field": args.text_field,
        "target_tokens": args.target_tokens,
        "written_tokens": written_tokens,
        "accepted_docs": accepted,
        "source_rows_examined": source_rows,
        "accepted_source_utf8_bytes": source_bytes,
        "rejected": rejected,
        "tokenizer": {"type": "ByteTokenizer", "vocab_size": tok.vocab_size, "bos": tok.bos_id, "eos": tok.eos_id},
        "document_boundary_crossing": False,
        "exact_dedup": "normalized casefold/whitespace SHA-256; disk-backed SQLite",
        "near_dedup": "required as separate admission pass before final multi-source training mix",
        "shards": shards,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
