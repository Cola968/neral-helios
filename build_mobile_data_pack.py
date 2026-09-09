from __future__ import annotations

"""Build a reproducible, pretokenized Neral candidate corpus on a GitHub-hosted CPU.

This is deliberately a *candidate* pack, not an automatically promoted training corpus.
It resolves immutable Hugging Face revisions, streams only public data, applies deterministic
quality/dedup/contamination guards, and writes Neral's exact uint16 byte-token IDs. No model
weights are downloaded or used.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

from tokenizer import ByteTokenizer

TOKENIZER = ByteTokenizer()

SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)", re.I
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){8,16}(?!\d)")
URL_RE = re.compile(r"https?://\S+", re.I)
WORD_RE = re.compile(r"[\wÄÖÜäöüß]+", re.UNICODE)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    language: str
    fraction: float
    config_preferences: tuple[str, ...]
    license: str


SOURCES = (
    SourceSpec("fineweb2_de", "HuggingFaceFW/fineweb-2", "de", 0.40, ("deu_Latn", "deu"), "ODC-By-1.0"),
    SourceSpec("fineweb2_en", "HuggingFaceFW/fineweb-2", "en", 0.20, ("eng_Latn", "eng"), "ODC-By-1.0"),
    SourceSpec("fineweb_edu", "HuggingFaceFW/fineweb-edu", "en", 0.25, ("sample-10BT", "default"), "ODC-By-1.0"),
    SourceSpec("finemath", "HuggingFaceTB/finemath", "en", 0.15, ("finemath-3plus", "finemath-4plus", "default"), "ODC-By-1.0"),
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def digest_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def benchmark_shingles(root: Path, width: int = 8) -> set[str]:
    """Collect exact multi-word phrases from local frozen eval files for contamination screening."""
    patterns = ("*benchmark*.json", "*benchmark*.jsonl", "*nis2*.json", "*nis2*.jsonl", "*rawmath*.jsonl", "*raw_math*.jsonl")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.rglob(pattern))
    fields = {"question", "prompt", "instruction", "input", "text", "problem"}
    strings: list[str] = []

    def walk(value, key: str | None = None):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k).lower())
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and (key in fields if key else False):
            strings.append(value)

    for path in sorted(paths):
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf8", errors="ignore").splitlines():
                    if line.strip():
                        walk(json.loads(line))
            else:
                walk(json.loads(path.read_text(encoding="utf8", errors="ignore")))
        except Exception:
            continue
    shingles: set[str] = set()
    for text in strings:
        words = [w.lower() for w in WORD_RE.findall(normalize(text))]
        if len(words) < width:
            continue
        # First, middle and last phrase are enough to catch verbatim benchmark copies cheaply.
        starts = {0, max(0, (len(words) - width) // 2), len(words) - width}
        for start in starts:
            phrase = " ".join(words[start:start + width])
            shingles.add(hashlib.blake2s(phrase.encode("utf8"), digest_size=8).hexdigest())
    return shingles


def hits_benchmark(text: str, shingles: set[str], width: int = 8) -> bool:
    if not shingles:
        return False
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < width:
        return False
    for i in range(0, len(words) - width + 1, 2):
        phrase = " ".join(words[i:i + width])
        h = hashlib.blake2s(phrase.encode("utf8"), digest_size=8).hexdigest()
        if h in shingles:
            return True
    return False


def quality_reason(text: str, benchmark_hashes: set[str]) -> str | None:
    raw = text.encode("utf8")
    if len(raw) < 256:
        return "too_short"
    if len(raw) > 80_000:
        return "too_long"
    if "\x00" in text:
        return "nul"
    if SECRET_RE.search(text):
        return "secret"
    if len(EMAIL_RE.findall(text)) > 2 or len(PHONE_RE.findall(text)) > 2:
        return "pii_dense"
    if len(URL_RE.findall(text)) > 20:
        return "url_dense"
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text))
    if printable < 0.96:
        return "nonprintable"
    alnum = sum(ch.isalnum() for ch in text) / max(1, len(text))
    if alnum < 0.25:
        return "low_alnum"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        repeated = len(lines) - len(set(lines))
        if repeated / len(lines) > 0.45:
            return "repeated_lines"
    words = WORD_RE.findall(text)
    if len(words) >= 80:
        common = Counter(w.lower() for w in words).most_common(1)[0][1]
        if common / len(words) > 0.18:
            return "word_loop"
    if hits_benchmark(text, benchmark_hashes):
        return "benchmark_phrase"
    return None


def resolve_config(repo: str, revision: str, preferences: tuple[str, ...]) -> str:
    names = get_dataset_config_names(repo, revision=revision)
    if not names:
        raise RuntimeError(f"no configs for {repo}@{revision}")
    for pref in preferences:
        if pref in names:
            return pref
    lowered = {name.lower(): name for name in names}
    for pref in preferences:
        for low, original in lowered.items():
            if pref.lower() in low:
                return original
    # FineWeb-Edu/FineMath evolve config labels; record exact fallback in the manifest.
    return names[0]


class DedupDB:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")
        self.pending = 0

    def add(self, h: str) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO seen(h) VALUES (?)", (h,))
        added = cur.rowcount == 1
        self.pending += 1
        if self.pending >= 1000:
            self.db.commit(); self.pending = 0
        return added

    def close(self):
        self.db.commit(); self.db.close()


class ShardWriter:
    def __init__(self, root: Path, split: str, max_tokens: int):
        self.root = root; self.split = split; self.max_tokens = int(max_tokens)
        self.index = -1; self.tokens = 0; self.total_tokens = 0; self.handle = None; self.path = None
        self.shards: list[dict] = []
        self._open()

    def _open(self):
        if self.handle is not None:
            self._close_current()
        self.index += 1; self.tokens = 0
        self.path = self.root / f"{self.split}_{self.index:03d}.bin"
        self.handle = self.path.open("wb")

    def _close_current(self):
        self.handle.flush(); self.handle.close()
        size = self.path.stat().st_size
        h = hashlib.sha256()
        with self.path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        self.shards.append({"file": self.path.name, "tokens": self.tokens, "bytes": size, "sha256": h.hexdigest()})

    def add(self, ids: list[int]) -> tuple[str, int, int]:
        if self.tokens and self.tokens + len(ids) > self.max_tokens:
            self._open()
        start = self.tokens
        arr = np.asarray(ids, dtype="<u2")
        self.handle.write(arr.tobytes(order="C"))
        self.tokens += len(ids); self.total_tokens += len(ids)
        return self.path.name, start, self.tokens

    def close(self):
        if self.handle is not None:
            self._close_current(); self.handle = None


def row_text(row: dict) -> str | None:
    for key in ("text", "content", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def stream_source(spec: SourceSpec, revision: str, config: str) -> Iterable[dict]:
    ds = load_dataset(spec.repo, config, split="train", streaming=True, revision=revision)
    yield from ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mobile_data_pack")
    ap.add_argument("--target-bytes", type=int, default=128 * 1024 * 1024)
    ap.add_argument("--shard-tokens", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--max-docs-per-source", type=int, default=500_000)
    args = ap.parse_args()
    if args.target_bytes < 8 * 1024 * 1024 or args.shard_tokens < 1_000_000:
        raise ValueError("budgets are too small")

    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    benchmark_hashes = benchmark_shingles(Path(__file__).resolve().parent)
    api = HfApi()
    resolved = []
    for spec in SOURCES:
        info = api.dataset_info(spec.repo)
        revision = str(info.sha)
        config = resolve_config(spec.repo, revision, spec.config_preferences)
        resolved.append((spec, revision, config))

    dedup = DedupDB(root / "dedup.sqlite")
    train_writer = ShardWriter(root, "train", args.shard_tokens)
    val_writer = ShardWriter(root, "val", max(1_000_000, args.shard_tokens // 4))
    docs = (root / "documents.jsonl").open("w", encoding="utf8")
    source_reports = []
    total_accepted_bytes = 0
    start_wall = time.time()

    try:
        for spec, revision, config in resolved:
            quota = int(args.target_bytes * spec.fraction)
            accepted_bytes = accepted_docs = scanned = 0
            rejected = Counter()
            print(json.dumps({"source": spec.name, "repo": spec.repo, "revision": revision, "config": config, "quota_bytes": quota}), flush=True)
            for row in stream_source(spec, revision, config):
                scanned += 1
                if scanned > args.max_docs_per_source:
                    break
                raw = row_text(row)
                if raw is None:
                    rejected["missing_text"] += 1; continue
                text = normalize(raw)
                reason = quality_reason(text, benchmark_hashes)
                if reason:
                    rejected[reason] += 1; continue
                h = digest_text(text)
                if not dedup.add(h):
                    rejected["exact_duplicate"] += 1; continue
                b = len(text.encode("utf8"))
                # Do not wildly overshoot a source quota with one unusually large document.
                if accepted_bytes and accepted_bytes + b > quota * 1.01:
                    rejected["quota_tail"] += 1; continue
                ids = TOKENIZER.encode(text + "\n", bos=True, eos=True)
                split = "val" if int(h[:8], 16) % 100 == 0 else "train"
                writer = val_writer if split == "val" else train_writer
                shard, begin, end = writer.add(ids)
                docs.write(json.dumps({
                    "id": h, "source": spec.name, "repo": spec.repo, "revision": revision,
                    "config": config, "language": spec.language, "split": split, "shard": shard,
                    "start": begin, "end": end, "tokens": len(ids), "utf8_bytes": b,
                }, ensure_ascii=False) + "\n")
                accepted_docs += 1; accepted_bytes += b; total_accepted_bytes += b
                if accepted_docs % 1000 == 0:
                    print(json.dumps({"source": spec.name, "accepted_docs": accepted_docs, "accepted_bytes": accepted_bytes, "scanned": scanned}), flush=True)
                if accepted_bytes >= quota:
                    break
            report = {
                **asdict(spec), "revision": revision, "config": config, "quota_bytes": quota,
                "scanned_docs": scanned, "accepted_docs": accepted_docs,
                "accepted_utf8_bytes": accepted_bytes, "rejected": dict(rejected),
            }
            source_reports.append(report)
            if accepted_bytes < quota * 0.90:
                raise RuntimeError(f"source {spec.name} reached only {accepted_bytes}/{quota} bytes")
    finally:
        docs.close(); dedup.close(); train_writer.close(); val_writer.close()

    # Dedup DB is implementation state, not part of the portable training artifact.
    for suffix in ("", "-wal", "-shm"):
        p = root / f"dedup.sqlite{suffix}"
        if p.exists(): p.unlink()

    manifest = {
        "version": 1,
        "name": "Neral mobile feeder candidate pack",
        "candidate_only": True,
        "automatic_training_admission": False,
        "tokenizer": {"type": "Neral deterministic byte tokenizer", "vocab_size": 260, "dtype": "uint16-le", "bos": 1, "eos": 2, "byte_offset": 4},
        "requested_utf8_bytes": args.target_bytes,
        "accepted_utf8_bytes": total_accepted_bytes,
        "train_tokens": train_writer.total_tokens,
        "val_tokens": val_writer.total_tokens,
        "total_tokens": train_writer.total_tokens + val_writer.total_tokens,
        "shard_tokens": args.shard_tokens,
        "benchmark_shingles_loaded": len(benchmark_hashes),
        "filters": ["NFKC", "secret", "PII-density", "URL-density", "printability", "alnum", "repeated-lines", "word-loop", "exact-dedup", "benchmark-8-word-shingles"],
        "sources": source_reports,
        "train_shards": train_writer.shards,
        "val_shards": val_writer.shards,
        "elapsed_seconds": time.time() - start_wall,
        "foreign_model_weights_used": False,
        "notes": "FineWeb-Edu/FineMath source curation may itself use classifier annotations; Neral consumes only dataset text, never their model weights. Full project contamination/license gate remains required before promotion.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    (root / "ATTRIBUTION.txt").write_text(
        "HuggingFaceFW/fineweb-2 — ODC-By 1.0\n"
        "HuggingFaceFW/fineweb-edu — ODC-By 1.0\n"
        "HuggingFaceTB/finemath — ODC-By 1.0\n"
        "Exact immutable revisions and configs are recorded in manifest.json.\n",
        encoding="utf8",
    )
    print(json.dumps({"complete": True, "accepted_bytes": total_accepted_bytes, "tokens": manifest["total_tokens"], "train_shards": len(train_writer.shards), "val_shards": len(val_writer.shards)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
