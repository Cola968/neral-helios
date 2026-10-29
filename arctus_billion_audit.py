"""Offline, conservative, disk-backed near-duplicate and split-leakage audit.

For PR #205's staged JSONL data, NOT for model weights or tokenized .bin
shards. No downloading, cloud jobs, training, modifying the source corpus, or
automatic certification. Uses five 13/13/13/13/12-bit SimHash LSH bands: a Hamming distance
of <=4 is guaranteed to share a band, although SimHash itself is a heuristic.
Higher-distance near duplicates and benchmark *semantic* contamination are NOT
excluded by this test and require separate manual review.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ROW_BYTES = 4 * 1024 * 1024
MAX_CANDIDATES_PER_BUCKET = 4096
SHARD = re.compile(r"(train|valid)-[0-9]{6}\.jsonl")
DIGEST = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
WORDS = re.compile(r"\w+", re.UNICODE)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            d.update(data)
    return d.hexdigest()


def normalize_for_dedup(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def simhash64(text: str) -> int:
    """Word 3-shingle SimHash with deterministic 64-bit BLAKE2b features."""
    words = WORDS.findall(normalize_for_dedup(text))
    if not words:
        return 0
    if len(words) < 3:
        grams = [tuple(words)]
    else:
        grams = (tuple(words[i:i + 3]) for i in range(len(words) - 2))
    freqs = collections.Counter(grams)
    weights = [0] * 64
    for shingle, count in freqs.items():
        feature = " ".join(shingle).encode("utf8")
        bits = int.from_bytes(hashlib.blake2b(feature, digest_size=8).digest(), "little")
        for bit in range(64):
            weights[bit] += count if (bits >> bit) & 1 else -count
    return sum(1 << i for i, score in enumerate(weights) if score > 0)


def hamming64(a: int, b: int) -> int:
    if not (0 <= a < 2**64 and 0 <= b < 2**64):
        raise ValueError("SimHash must be 64 bit unsigned")
    return (a ^ b).bit_count()


def _validated_manifest(root: Path) -> tuple[dict, list, dict]:
    path = root / "manifest.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Missing, linked or oversized staged corpus manifest")
    manifest = json.loads(path.read_text(encoding="utf8"))
    if (manifest.get("version") != 1 or
            manifest.get("purpose") != "Arctus350 native pretraining DATA only" or
            manifest.get("tokenizer_kind") != "NERAL-16K-BPE" or
            manifest.get("training_started") is not False or
            manifest.get("trained_tokens") != 0):
        raise ValueError("Only the original untrained Arctus staging schema may be audited")
    if manifest.get("data_is_training_ready") is not False:
        raise ValueError("Audit cannot certify previously promoted staging data")
    specs = manifest.get("sources")
    if not isinstance(specs, list) or not specs:
        raise ValueError("No pinned staged source records")
    sources = {}
    for s in specs:
        if not isinstance(s, dict):
            raise ValueError("Invalid source record")
        sid = s.get("source_id")
        if (not isinstance(sid, str) or sid in sources or
                s.get("license") != "ODC-By-1.0" or
                s.get("language") not in ("en", "de") or
                s.get("domain") not in ("general", "education") or
                not isinstance(s.get("revision"), str) or
                not REVISION.fullmatch(s["revision"])):
            raise ValueError("Invalid or unapproved pinned source record")
        sources[sid] = s

    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("No complete JSONL shards")
    seen = set()
    for s in entries:
        if not isinstance(s, dict):
            raise ValueError("Invalid shard manifest entry")
        name = s.get("file")
        if (not isinstance(name, str) or
                not SHARD.fullmatch(name) or name in seen or
                s.get("split") != name.split("-")[0] or
                not isinstance(s.get("sha256"), str) or
                not DIGEST.fullmatch(s["sha256"]) or
                type(s.get("bytes")) is not int or s["bytes"] <= 0 or
                type(s.get("tokens")) is not int or s["tokens"] < 1 or
                type(s.get("documents")) is not int or s["documents"] < 1):
            raise ValueError("Unsafe or incomplete shard record")
        seen.add(name)
        file = root / name
        if not file.is_file() or file.is_symlink():
            raise ValueError("Shard missing or symlinked")
        if file.stat().st_size != s["bytes"]:
            raise ValueError("Shard byte count differs from manifest")
        if sha256_file(file) != s["sha256"]:
            raise ValueError("Shard SHA256 mismatch")
    present = {p.name for p in root.glob("train-*.jsonl")} | {
        p.name for p in root.glob("valid-*.jsonl")}
    if seen != present:
        raise ValueError("Missing or unexpected train/valid JSONL shard")
    return manifest, sorted(entries, key=lambda x: x["file"]), sources


def _rows(root: Path, entries, sources):
    for shard in entries:
        total_tokens = 0
        total_docs = 0
        split = shard["split"]
        with (root / shard["file"]).open("rb") as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    raise ValueError("Blank staged row")
                if len(line) > MAX_ROW_BYTES:
                    raise ValueError("Oversized staged document row")
                try:
                    row = json.loads(line.decode("utf8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("Malformed JSONL staged document") from exc
                if not isinstance(row, dict):
                    raise ValueError("Staged document must be a record")
                source = sources.get(row.get("source_id"))
                if source is None:
                    raise ValueError("Unknown staged source ID")
                if any(row.get(k) != source.get(k) for k in ("revision", "license", "language", "domain")):
                    raise ValueError("Staged row source provenance mismatch")
                text = row.get("text")
                digest = row.get("normalized_sha256")
                tok = row.get("tokens")
                if (not isinstance(text, str) or
                        not text.strip() or
                        not isinstance(digest, str) or not DIGEST.fullmatch(digest) or
                        type(tok) is not int or tok < 3):
                    raise ValueError("Invalid staged document text/hash/token count")
                if hashlib.sha256(normalize_for_dedup(text).encode("utf8")).hexdigest() != digest:
                    raise ValueError("Staged normalized document digest mismatch")
                total_docs += 1
                total_tokens += tok
                yield split, digest, text
        if total_docs != shard["documents"] or total_tokens != shard["tokens"]:
            raise ValueError("Shard document/token accounting mismatch")


def audit_corpus(root: Path, *, max_candidates_per_bucket: int = MAX_CANDIDATES_PER_BUCKET) -> dict:
    """Audit source data in read-only mode; temp SQLite database is discarded."""
    source = Path(root)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Staged corpus directory missing or symlinked")
    if type(max_candidates_per_bucket) is not int or not 1 <= max_candidates_per_bucket <= 100_000:
        raise ValueError("Invalid candidate bucket limit")
    manifest, entries, sources = _validated_manifest(source)
    counters = collections.Counter()
    with tempfile.TemporaryDirectory(prefix="arctus-audit-") as scratch:
        with sqlite3.connect(Path(scratch) / "audit.sqlite3") as db:
            db.executescript("""
                CREATE TABLE docs(
                    id INTEGER PRIMARY KEY,
                    digest TEXT NOT NULL,
                    hash64 TEXT NOT NULL,
                    split TEXT NOT NULL
                );
                CREATE INDEX docs_digest ON docs(digest);
                CREATE TABLE bands(bucket INTEGER NOT NULL, doc_id INTEGER NOT NULL);
                CREATE INDEX bands_bucket ON bands(bucket);
            """)
            for split, digest, text in _rows(source, entries, sources):
                counters["documents_checked"] += 1
                old = db.execute(
                    "SELECT split FROM docs WHERE digest=? LIMIT 1", (digest,)
                ).fetchone()
                if old is not None:
                    counters["exact_duplicate_pairs"] += 1
                    if split != old[0]:
                        counters["cross_split_exact_duplicate_pairs"] += 1
                    continue
                sim = simhash64(text)
                nearby = set()
                for band in range(5):
                    width = 12 if band == 4 else 13
                    key = (band << 13) | ((sim >> (band * 13)) & ((1 << width) - 1))
                    candidates = db.execute(
                        "SELECT d.id, d.hash64, d.split FROM bands b "
                        "JOIN docs d ON d.id=b.doc_id WHERE b.bucket=? "
                        "LIMIT ?", (key, max_candidates_per_bucket + 1)
                    ).fetchall()
                    if len(candidates) > max_candidates_per_bucket:
                        raise ValueError("Near-duplicate LSH bucket saturated; audit incomplete")
                    nearby.update(tuple(x) for x in candidates)
                for _id, other_hash, other_split in nearby:
                    if hamming64(sim, int(other_hash, 16)) <= 4:
                        if split != other_split:
                            counters["cross_split_near_duplicate_pairs"] += 1
                        else:
                            counters["within_split_near_duplicate_pairs"] += 1
                cursor = db.execute(
                    "INSERT INTO docs(digest,hash64,split) VALUES (?,?,?)",
                    (digest, f"{sim:016x}", split))
                row_id = cursor.lastrowid
                db.executemany(
                    "INSERT INTO bands(bucket,doc_id) VALUES (?,?)",
                    (((band << 13) | ((sim >> (band * 13)) & ((1 << (12 if band == 4 else 13)) - 1)), row_id)
                     for band in range(5))
                )
                if counters["documents_checked"] % 1000 == 0:
                    db.commit()
            db.commit()
    return {
        "audit_schema": 1,
        "staged_manifest_sha256": sha256_file(source / "manifest.json"),
        "integrity_verified": True,
        "documents_checked": counters["documents_checked"],
        "exact_duplicate_pairs": counters["exact_duplicate_pairs"],
        "cross_split_exact_duplicate_pairs": counters["cross_split_exact_duplicate_pairs"],
        "cross_split_near_duplicate_pairs": counters["cross_split_near_duplicate_pairs"],
        "within_split_near_duplicate_pairs": counters["within_split_near_duplicate_pairs"],
        "near_duplicate_method": "64-bit word-trigram SimHash <=4 bits, five 13/13/13/13/12-bit LSH bands",
        "near_duplicate_coverage": "conservative heuristic; no semantic or historical near-duplicate guarantee",
        "held_out_contamination_eliminated": False,
        "needs_manual_license_and_holdout_review": True,
        "training_ready": False,
        "trained_tokens": 0,
        "model_weights_changed": False,
        "rented_gpu": False,
        "download_performed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="existing staged JSONL directory; read only")
    parser.add_argument("--report", type=Path, help="optional NEW report path, never overwrites")
    args = parser.parse_args(argv)
    if args.corpus is None:
        print(json.dumps({
            "state": "READ_ONLY_PLAN",
            "requires_explicit_corpus_path": True,
            "gpu_rented": False,
            "training_started": False,
            "download_performed": False,
        }, indent=2))
        return 0
    result = audit_corpus(args.corpus)
    if args.report is not None:
        with args.report.open("x", encoding="utf8") as dest:
            json.dump(result, dest, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
