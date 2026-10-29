"""Bounded-memory, explicit opt-in staging of billion-scale *training data*, not weights.

No network or filesystem writes in default --plan. --execute --allow-download explicitly
streams approved/pinned Hugging Face datasets to local JSONL shards, without training.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

from hf_source_policy50 import assert_training_allowed
from prepare_pretrain50_b2 import (
    benchmark_contamination, clean_text, normalize_for_dedup, split_for, sha256_file
)
from bootstrap_b3_sources import SPECS

REVISION = re.compile(r"[0-9a-f]{40}")
MAX_DOCUMENT_BYTES = 2_000_000
DEFAULT_SHARD_BYTES = 64 * 1024 * 1024


def default_sources():
    """Re-use the project's exact existing B3 data mix and source configurations."""
    return [
        {"id": s["id"], "repo": s["repo"], "config": s["config"],
         "requested_revision": s["revision"], "language": s["language"],
         "domain": s["domain"], "fraction": s["target_fraction"],
         "license": s["license"]}
        for s in SPECS
    ]


def _quotas(sources, total):
    if type(total) is not int or total < 1:
        raise ValueError("target_train_tokens must be a positive integer")
    if not sources:
        raise ValueError("sources cannot be empty")
    if not math.isclose(sum(s["fraction"] for s in sources), 1.0, abs_tol=1e-8):
        raise ValueError("source fractions must sum to 1")
    out = [int(total * s["fraction"]) for s in sources]
    out[-1] += total - sum(out)
    return out


def plan(target_train_tokens=1_000_000_000):
    sources = default_sources()
    quotas = _quotas(sources, target_train_tokens)
    return {
        "purpose": "Arctus350 own-model pretraining data; NOT model training",
        "target_train_tokens": target_train_tokens,
        "actual_staged_tokens": 0,
        "trained_tokens": 0,
        "requires_explicit_download": True,
        "requires_existing_original_tokenizer": True,
        "sources": [
            {"dataset_id": s["repo"], "config": s["config"],
             "requested_revision": s["requested_revision"],
             "language": s["language"], "license": s["license"],
             "target_train_tokens": q}
            for s, q in zip(sources, quotas)
        ],
        "note": "Source refs must be resolved to immutable commit SHAs before any streaming. "
                "Source token quotas are goals, not data already staged.",
    }


def validate_sources(sources):
    ids = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid source entry")
        sid = source.get("id")
        if not isinstance(sid, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,79}", sid) or sid in ids:
            raise ValueError("invalid or duplicated source id")
        ids.add(sid)
        repo = source.get("repo")
        policy = assert_training_allowed(repo)
        if policy.role != "pretrain":
            raise ValueError("Only approved human/web pretrain corpora are permitted")
        if source.get("license") != policy.license or source.get("license") != "ODC-By-1.0":
            raise ValueError("Dataset license differs from reviewed policy")
        if source.get("language") not in {"de", "en"}:
            raise ValueError("unsupported language")
        if source.get("domain") not in {"general", "education"}:
            raise ValueError("unsupported source domain")
        if not isinstance(source.get("config"), str) or not source["config"]:
            raise ValueError("missing source config")
        revision = source.get("revision")
        if not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise ValueError("Dataset revision must be an immutable 40-hex SHA")
        frac = source.get("fraction")
        if type(frac) not in (int, float) or not 0 < frac <= 1 or not math.isfinite(frac):
            raise ValueError("invalid source share")


class ShardWriter:
    """Disk-bounded JSONL shards; only complete shards appear in the manifest."""

    def __init__(self, root, max_bytes):
        if type(max_bytes) is not int or max_bytes < 256:
            raise ValueError("shard size must be >=256 bytes")
        self.root, self.max_bytes = root, max_bytes
        self.open_files = {}
        self.shards = []
        self.serial = collections.Counter()

    def write(self, split, row):
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf8")
        if len(payload) > MAX_DOCUMENT_BYTES * 2:
            raise ValueError("serialized document exceeds permitted size")
        current = self.open_files.get(split)
        if current and current["bytes"] and current["bytes"] + len(payload) > self.max_bytes:
            self.seal(split)
            current = None
        if current is None:
            num = self.serial[split]
            self.serial[split] += 1
            name = f"{split}-{num:06d}.jsonl"
            partial = self.root / (name + ".partial")
            current = {"file": name, "path": partial,
                       "handle": partial.open("wb"),
                       "bytes": 0, "tokens": 0, "documents": 0, "split": split}
            self.open_files[split] = current
        current["handle"].write(payload)
        current["bytes"] += len(payload)
        current["tokens"] += row["tokens"]
        current["documents"] += 1

    def seal(self, split):
        current = self.open_files.pop(split, None)
        if current is None:
            return
        current["handle"].flush()
        current["handle"].close()
        final_path = self.root / current["file"]
        current["path"].replace(final_path)
        self.shards.append({
            "file": current["file"], "split": split,
            "bytes": current["bytes"], "documents": current["documents"],
            "tokens": current["tokens"], "sha256": sha256_file(final_path),
        })

    def finish(self):
        for split in list(self.open_files):
            self.seal(split)
        return sorted(self.shards, key=lambda x: x["file"])


def build_from_streams(*, sources, output: Path, target_train_tokens: int,
                       tokenizer, stream_factory, contamination_phrases=(),
                       max_shard_bytes=DEFAULT_SHARD_BYTES):
    """Run an already-resolved source stream; inject fake streams for offline tests.

    Writes to a never-before-used output directory. Incomplete runs have no manifest
    and are therefore not training-ready. SQLite dedup never holds all hashes in RAM.
    """
    validate_sources(sources)
    quotas = _quotas(sources, target_train_tokens)
    out = Path(output)
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing corpus directory")
    out.mkdir(parents=True, exist_ok=False)
    writer = ShardWriter(out, max_shard_bytes)
    rejected = collections.Counter()
    counts = collections.Counter()
    source_reports = []
    normalized_phrases = tuple(x for x in (
        normalize_for_dedup(s) for s in contamination_phrases
    ) if len(x) >= 10)
    db_path = out / "dedup.sqlite3"
    try:
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY, source_row TEXT UNIQUE)")
            for spec, quota in zip(sources, quotas):
                accepted = 0
                documents = 0
                if quota <= 0:
                    source_reports.append({"source_id": spec["id"], "dataset_id": spec["repo"],
                                           "revision": spec["revision"], "target_train_tokens": 0,
                                           "actual_train_tokens": 0, "documents": 0,
                                           "license": spec["license"], "config": spec["config"]})
                    continue
                for index, original in enumerate(stream_factory(spec)):
                    if not isinstance(original, dict):
                        rejected["invalid_row"] += 1
                        continue
                    raw = original.get("text")
                    if not isinstance(raw, str):
                        rejected["not_text"] += 1
                        continue
                    try:
                        raw_size = len(raw.encode("utf8"))
                    except UnicodeError:
                        rejected["encoding_or_control"] += 1
                        continue
                    if raw_size > MAX_DOCUMENT_BYTES:
                        rejected["document_too_large"] += 1
                        continue
                    text, reason = clean_text(raw, spec["domain"])
                    if reason is not None:
                        rejected[reason] += 1
                        continue
                    normalized = normalize_for_dedup(text)
                    if any(fragment in normalized for fragment in normalized_phrases):
                        rejected["frozen_benchmark_contamination"] += 1
                        continue
                    digest = hashlib.sha256(normalized.encode("utf8")).hexdigest()
                    rid = original.get("id")
                    rid = str(rid) if isinstance(rid, (str, int)) and str(rid).strip() else digest
                    if len(rid) > 256:
                        rid = hashlib.sha256(rid.encode("utf8")).hexdigest()
                    source_row = hashlib.sha256((spec["id"] + "\0" + rid).encode("utf8")).hexdigest()
                    try:
                        db.execute("INSERT INTO seen(digest,source_row) VALUES (?,?)", (digest, source_row))
                    except sqlite3.IntegrityError:
                        rejected["duplicate"] += 1
                        continue
                    tokenized = tokenizer.encode(text)
                    tokens = len(tokenized.ids) + 2  # BOS/EOS are added by a later packer
                    if tokens < 3:
                        rejected["empty_tokenization"] += 1
                        continue
                    split = split_for(spec["id"], rid)
                    candidate_url = original.get("url")
                    url = candidate_url.split("?")[0].split("#")[0] if (
                        isinstance(candidate_url, str) and len(candidate_url) <= 2048
                        and candidate_url.startswith("https://")
                    ) else None
                    record = {
                        "id": rid, "source_id": spec["id"], "language": spec["language"],
                        "domain": spec["domain"], "revision": spec["revision"],
                        "license": spec["license"], "url": url,
                        "normalized_sha256": digest, "tokens": tokens, "text": text,
                    }
                    writer.write(split, record)
                    counts[split] += tokens
                    documents += 1
                    if split == "train":
                        accepted += tokens
                        if accepted >= quota:
                            break
                db.commit()
                if accepted < quota:
                    raise ValueError(f"{spec['id']}: train token quota not reached ({accepted}/{quota})")
                source_reports.append({
                    "source_id": spec["id"], "dataset_id": spec["repo"],
                    "revision": spec["revision"], "license": spec["license"],
                    "config": spec["config"], "language": spec["language"],
                    "domain": spec["domain"], "target_train_tokens": quota,
                    "actual_train_tokens": accepted, "documents": documents,
                })
        shards = writer.finish()
        manifest = {
            "version": 1, "purpose": "Arctus350 native pretraining DATA only",
            "tokenizer_kind": "NERAL-16K-BPE",
            "token_count_unit": "actual BPE encode length plus 2 boundary tokens",
            "target_train_tokens": target_train_tokens,
            "actual_train_tokens": counts["train"],
            "actual_staged_tokens": sum(counts.values()),
            "tokens_by_split": dict(counts),
            "trained_tokens": 0,
            "training_started": False,
            "data_is_training_ready": False,  # manual near-dup/license/holdout audit still required
            "dedup": "disk-backed exact normalized-text SHA256",
            "contamination": "frozen literal phrases only; not a full semantic audit",
            "sources": source_reports,
            "rejected": dict(rejected),
            "shards": shards,
        }
        temp = out / "manifest.json.partial"
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
        temp.replace(out / "manifest.json")
        return manifest
    finally:
        for split in list(writer.open_files):
            writer.seal(split)


def _tokenizer(path: Path):
    from frontier350.chat350_contract import EXPECTED_TOKENIZER_SHA256
    import secrets
    from tokenizers import Tokenizer
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("The user's own local 16K BPE tokenizer is missing or oversized")
    digest = sha256_file(path)
    if not secrets.compare_digest(digest, EXPECTED_TOKENIZER_SHA256):
        raise ValueError("Original Arctus350 tokenizer digest mismatch")
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size() != 16_384:
        raise ValueError("Original Arctus350 vocabulary mismatch")
    return tok


def _pin_and_stream():
    from huggingface_hub import HfApi
    from datasets import load_dataset
    api = HfApi()
    resolved = []
    for row in default_sources():
        spec = dict(row)
        revision = api.dataset_info(spec["repo"], revision=spec["requested_revision"]).sha
        if not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise ValueError(f"Cannot pin immutable dataset revision: {spec['repo']}")
        spec["revision"] = revision
        resolved.append(spec)
    def streams(spec):
        return load_dataset(spec["repo"], spec["config"], split="train",
                            streaming=True, revision=spec["revision"])
    return resolved, streams


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--execute", action="store_true", help="explicitly prepare and download corpus")
    ap.add_argument("--allow-download", action="store_true", help="second explicit network consent")
    ap.add_argument("--tokenizer", type=Path, help="local original NERAL-16K BPE tokenizer.json")
    ap.add_argument("--out", type=Path, help="new, empty corpus destination")
    ap.add_argument("--max-shard-mib", type=int, default=64)
    args = ap.parse_args(argv)
    report = plan(args.target_train_tokens)
    if not args.execute:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if not args.allow_download or args.tokenizer is None or args.out is None:
        ap.error("--execute requires --allow-download, --tokenizer and --out")
    if not 1 <= args.max_shard_mib <= 512:
        ap.error("--max-shard-mib must be 1..512")
    tokenizer = _tokenizer(args.tokenizer)
    specs, streams = _pin_and_stream()
    frozen, frozen_file_hashes = benchmark_contamination(Path(__file__).resolve().parent)
    output = build_from_streams(
        sources=specs, output=args.out, target_train_tokens=args.target_train_tokens,
        tokenizer=tokenizer, stream_factory=streams,
        contamination_phrases=frozen, max_shard_bytes=args.max_shard_mib * 1024 * 1024)
    # Keep benchmark provenance explicit; never claim universal leakage detection.
    path = args.out / "contamination_sources.json"
    path.write_text(json.dumps(frozen_file_hashes, indent=2), encoding="utf8")
    print(json.dumps({"manifest": str(args.out / "manifest.json"),
                      "actual_train_tokens": output["actual_train_tokens"],
                      "training_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
