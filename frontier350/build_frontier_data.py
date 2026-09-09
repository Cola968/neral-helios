from __future__ import annotations

"""Build audited uint16 foundation shards for Neral 350M Frontier.

Every Hugging Face source is resolved to an immutable commit SHA before streaming. The
builder applies conservative UTF-8/PII/spam filtering, persistent document deduplication,
exact token accounting and per-shard SHA256 hashes. Dialogue/SFT data is deliberately
excluded from this 10B foundation corpus and handled by a separate post-training pipeline.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from tokenizer350 import TOKENIZER, contract

ALLOWED_LICENSES = {"odc-by", "apache-2.0", "cc-by-4.0", "cc-by-sa-4.0", "mit", "bsd-3-clause"}
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ()\-]{7,}\d)(?!\d)")
SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.+=]{12,}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo: str
    config: str | None
    split: str
    license: str
    text_field: str
    share: float
    language: str
    kind: str


# 10B foundation target: language/knowledge + education + math, with a meaningful German slice.
# OASST2 and future preference/code instruction sets are post-training sources, not repeated
# hundreds of millions of times in foundation pretraining.
DEFAULT_SOURCES = [
    SourceSpec("fineweb_edu_en", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "odc-by", "text", 0.52, "en", "education"),
    SourceSpec("fineweb_en", "HuggingFaceFW/fineweb", "sample-10BT", "train", "odc-by", "text", 0.20, "en", "general"),
    SourceSpec("fineweb2_de", "HuggingFaceFW/fineweb-2", "deu_Latn", "train", "odc-by", "text", 0.13, "de", "general"),
    SourceSpec("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train", "odc-by", "text", 0.15, "multi", "math"),
]


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def quality_ok(text: str) -> tuple[bool, str]:
    if len(text) < 160:
        return False, "too_short"
    if len(text) > 500_000:
        return False, "too_long"
    if EMAIL_RE.search(text):
        return False, "email"
    if PHONE_RE.search(text):
        return False, "phone"
    if SECRET_RE.search(text):
        return False, "secret_pattern"
    if text.count("�") / max(1, len(text)) > 0.001:
        return False, "utf8_replacement"
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text))
    if printable < 0.985:
        return False, "nonprintable"
    lines = [x for x in text.split("\n") if x]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return False, "line_repetition"
    words = text.split()
    if len(words) >= 80:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.08:
            return False, "word_repetition"
    return True, "ok"


def content_hash(text: str) -> str:
    compact = " ".join(text.lower().split())
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PersistentSeen:
    """Disk-backed exact dedupe so a multi-billion-token build does not need a huge Python set."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
        self.pending = 0

    def add_if_new(self, digest: str) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (digest,))
        is_new = cur.rowcount == 1
        if is_new:
            self.pending += 1
            if self.pending >= 5000:
                self.db.commit()
                self.pending = 0
        return is_new

    def count(self) -> int:
        self.db.commit()
        return int(self.db.execute("SELECT COUNT(*) FROM seen").fetchone()[0])

    def close(self):
        self.db.commit()
        self.db.close()


def resolve_revision(repo: str) -> str:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo_id=repo, revision="main")
    sha = getattr(info, "sha", None)
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"could not resolve immutable revision for {repo}: {sha!r}")
    return sha


def stream_rows(spec: SourceSpec, revision: str) -> Iterable[dict]:
    from datasets import load_dataset
    kwargs = {"path": spec.repo, "split": spec.split, "streaming": True, "revision": revision}
    if spec.config:
        kwargs["name"] = spec.config
    return load_dataset(**kwargs)


def extract_text(spec: SourceSpec, row: dict) -> str:
    value = row.get(spec.text_field, "")
    return value if isinstance(value, str) else str(value or "")


class ShardWriter:
    def __init__(self, root: Path, prefix: str, shard_tokens: int):
        self.root = root
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.buffer: list[int] = []
        self.index = 0
        self.files: list[dict] = []
        root.mkdir(parents=True, exist_ok=True)

    def add(self, ids: list[int]):
        self.buffer.extend(ids)
        while len(self.buffer) >= self.shard_tokens:
            self._flush_exact(self.shard_tokens)

    def _flush_exact(self, n: int):
        chunk = np.asarray(self.buffer[:n], dtype=np.uint16)
        del self.buffer[:n]
        path = self.root / f"{self.prefix}_{self.index:05d}.bin"
        chunk.tofile(path)
        self.files.append({"path": path.name, "tokens": int(len(chunk)), "sha256": file_sha256(path)})
        self.index += 1

    def finish(self):
        if self.buffer:
            self._flush_exact(len(self.buffer))
        return self.files


def build_source(spec: SourceSpec, revision: str, target_tokens: int, writer: ShardWriter, seen: PersistentSeen) -> dict:
    accepted = rejected = raw_rows = 0
    rejection_reasons: dict[str, int] = {}
    emitted = 0
    start = time.time()
    for row in stream_rows(spec, revision):
        raw_rows += 1
        text = normalize(extract_text(spec, row))
        ok, reason = quality_ok(text)
        if not ok:
            rejected += 1
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        if not seen.add_if_new(content_hash(text)):
            rejected += 1
            rejection_reasons["duplicate"] = rejection_reasons.get("duplicate", 0) + 1
            continue
        ids = TOKENIZER.encode_document(text)
        remaining = target_tokens - emitted
        if len(ids) > remaining:
            ids = ids[:remaining]
        if ids:
            writer.add(ids)
            emitted += len(ids)
            accepted += 1
        if emitted >= target_tokens:
            break
        if raw_rows % 10_000 == 0:
            print(json.dumps({"source": spec.name, "rows": raw_rows, "accepted": accepted, "emitted_tokens": emitted, "target_tokens": target_tokens}), flush=True)
    if emitted < target_tokens:
        raise RuntimeError(f"source {spec.name} exhausted at {emitted:,}/{target_tokens:,} tokens")
    return {
        "source": asdict(spec), "revision": revision, "raw_rows": raw_rows,
        "accepted_rows": accepted, "rejected_rows": rejected,
        "rejection_reasons": rejection_reasons, "emitted_tokens": emitted,
        "elapsed_s": time.time() - start,
    }


def quota(total: int, spec: SourceSpec, index: int) -> int:
    if index == len(DEFAULT_SOURCES) - 1:
        return total - sum(int(total * s.share) for s in DEFAULT_SOURCES[:-1])
    return int(total * spec.share)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target-tokens", type=int, default=50_000_000,
                    help="total foundation tokens including validation; pilot first")
    ap.add_argument("--valid-tokens", type=int, default=5_000_000)
    ap.add_argument("--shard-tokens", type=int, default=25_000_000)
    ap.add_argument("--seed-note", default="frontier350-v1 deterministic source order")
    args = ap.parse_args()
    if args.target_tokens < 1_000_000 or args.valid_tokens < 100_000:
        raise SystemExit("corpus pilot too small for meaningful validation")
    if args.valid_tokens >= args.target_tokens:
        raise SystemExit("validation tokens must be smaller than total")
    if abs(sum(s.share for s in DEFAULT_SOURCES) - 1.0) > 1e-9:
        raise RuntimeError("source shares must sum to 1")
    for spec in DEFAULT_SOURCES:
        if spec.license not in ALLOWED_LICENSES:
            raise RuntimeError(f"license not admitted: {spec}")

    root = Path(args.root)
    train_root, valid_root = root / "train", root / "valid"
    root.mkdir(parents=True, exist_ok=True)
    if any(train_root.glob("*.bin")) or any(valid_root.glob("*.bin")):
        raise RuntimeError("refusing to mix a new build with existing token shards")

    resolved = {spec.name: resolve_revision(spec.repo) for spec in DEFAULT_SOURCES}
    source_reports = []
    train_writer = ShardWriter(train_root, "train", args.shard_tokens)
    valid_writer = ShardWriter(valid_root, "valid", min(args.shard_tokens, args.valid_tokens))
    seen = PersistentSeen(root / "dedupe.sqlite3")
    try:
        for i, spec in enumerate(DEFAULT_SOURCES):
            q = quota(args.valid_tokens, spec, i)
            if q:
                source_reports.append({"partition": "valid", **build_source(spec, resolved[spec.name], q, valid_writer, seen)})
        train_target = args.target_tokens - args.valid_tokens
        for i, spec in enumerate(DEFAULT_SOURCES):
            q = quota(train_target, spec, i)
            if q:
                source_reports.append({"partition": "train", **build_source(spec, resolved[spec.name], q, train_writer, seen)})
        dedup_count = seen.count()
    finally:
        seen.close()

    valid_files = valid_writer.finish()
    train_files = train_writer.finish()
    manifest = {
        "schema": 1,
        "architecture": "frontier350-v1",
        "corpus_role": "foundation-pretraining",
        "tokenizer": contract(),
        "target_tokens": args.target_tokens,
        "train_tokens": sum(x["tokens"] for x in train_files),
        "validation_tokens": sum(x["tokens"] for x in valid_files),
        "seed_note": args.seed_note,
        "benchmark_content_included": False,
        "licenses": sorted({s.license for s in DEFAULT_SOURCES}),
        "resolved_revisions": resolved,
        "sources": source_reports,
        "train_files": train_files,
        "valid_files": valid_files,
        "deduplicated_documents": dedup_count,
        "created_unix": time.time(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "root": str(root), "train_tokens": manifest["train_tokens"],
        "validation_tokens": manifest["validation_tokens"], "documents": dedup_count,
        "revisions": resolved,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
