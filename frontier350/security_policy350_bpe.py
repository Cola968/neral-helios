from __future__ import annotations

"""Production admission gate for Neral 350M Frontier BPE corpora."""

import hashlib
import json
import re
from pathlib import Path

from build_frontier_data import DEFAULT_SOURCES
from source_lock350 import locked_revisions

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_LICENSES = {"odc-by", "apache-2.0", "cc-by-4.0", "cc-by-sa-4.0", "mit", "bsd-3-clause"}
FORBIDDEN = ("nis2", "benchmark", "holdout", "rawmath_eval", "scorecard", "release_gate")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_shards(root: Path, subdir: str, items: list[dict]) -> int:
    if not items:
        raise ValueError(f"no {subdir} shards")
    total = 0
    base = (root / subdir).resolve()
    names = [str(item.get("path", "")) for item in items]
    if len(names) != len(set(names)):
        raise ValueError("duplicate shard entries")
    if set(names) != {p.name for p in base.glob("*.bin")}:
        raise ValueError("manifest does not cover exactly the loaded shards")
    for item in items:
        name = str(item.get("path", ""))
        digest = str(item.get("sha256", "")).lower()
        tokens = int(item.get("tokens", -1))
        if not name.endswith(".bin") or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"unsafe shard name: {name}")
        if any(word in name.lower() for word in FORBIDDEN):
            raise ValueError(f"benchmark-like shard name: {name}")
        if not HEX64.fullmatch(digest) or tokens <= 1:
            raise ValueError(f"invalid shard metadata: {name}")
        path = (base / name).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"missing/path-escape shard: {name}")
        if path.stat().st_size != tokens * 2:
            raise ValueError(f"uint16 size mismatch: {name}")
        if sha256(path) != digest:
            raise ValueError(f"hash mismatch: {name}")
        total += tokens
    return total


def validate_manifest(manifest_path: Path, root: Path, tokenizer_path: Path) -> dict:
    root = root.resolve()
    obj = json.loads(manifest_path.read_text(encoding="utf8"))
    if obj.get("schema") != 2 or obj.get("architecture") != "frontier350-v1":
        raise ValueError("wrong BPE manifest schema/architecture")
    if obj.get("corpus_role") != "foundation-pretraining":
        raise ValueError("wrong corpus role")
    if obj.get("benchmark_content_included") is not False:
        raise ValueError("benchmark separation not guaranteed")

    tokenizer = obj.get("tokenizer") or {}
    if tokenizer.get("type") != "byte-bpe-v1" or int(tokenizer.get("vocab_size", -1)) != 16_384:
        raise ValueError("wrong BPE tokenizer contract")
    expected_digest = str(tokenizer.get("tokenizer_sha256", "")).lower()
    if not HEX64.fullmatch(expected_digest) or not tokenizer_path.is_file() or sha256(tokenizer_path) != expected_digest:
        raise ValueError("local tokenizer digest mismatch")
    special = tokenizer.get("special_token_ids") or {}
    if set(special) != {"<|bos|>", "<|eos|>", "<|sep|>", "<|pad|>"}:
        raise ValueError("incomplete BPE special-token map")
    values = [int(v) for v in special.values()]
    if len(set(values)) != 4 or min(values) < 0 or max(values) >= 16_384:
        raise ValueError("invalid BPE special token IDs")
    from build_frontier_bpe_data import LocalBPE
    if LocalBPE(tokenizer_path).contract() != tokenizer:
        raise ValueError("manifest differs from actual tokenizer special-token contract")

    expected_revisions = locked_revisions(DEFAULT_SOURCES)
    if obj.get("resolved_revisions") != expected_revisions:
        raise ValueError("source revisions differ from immutable source lock")
    licenses = obj.get("licenses")
    if not isinstance(licenses, list) or not licenses or any(str(x).lower() not in ALLOWED_LICENSES for x in licenses):
        raise ValueError("unapproved/missing licenses")

    reports = obj.get("sources")
    if not isinstance(reports, list) or not reports:
        raise ValueError("source reports missing")
    for report in reports:
        spec = report.get("source") or {}
        name = str(spec.get("name", ""))
        repo = str(spec.get("repo", ""))
        if name not in expected_revisions or report.get("revision") != expected_revisions[name]:
            raise ValueError(f"source revision mismatch: {name}")
        from dataclasses import asdict
        locked = asdict(next(s for s in DEFAULT_SOURCES if s.name == name))
        if any(spec.get(k) != v for k, v in locked.items() if k != "share"):
            raise ValueError(f"source identity/license differs from lock: {name}")
        if any(word in (name + " " + repo).lower() for word in FORBIDDEN):
            raise ValueError(f"benchmark-like source: {name}")
        if int(report.get("emitted_tokens", 0)) <= 0:
            raise ValueError(f"empty source: {name}")

    train = _check_shards(root, "train", obj.get("train_files") or [])
    valid = _check_shards(root, "valid", obj.get("valid_files") or [])
    if train != int(obj.get("train_tokens", -1)) or valid != int(obj.get("validation_tokens", -1)):
        raise ValueError("partition token accounting mismatch")
    if train + valid != int(obj.get("target_tokens", -1)):
        raise ValueError("total token accounting mismatch")
    if train <= valid:
        raise ValueError("validation partition unexpectedly large")

    return {
        "ok": True,
        "schema": 2,
        "tokenizer_sha256": expected_digest,
        "train_tokens": train,
        "validation_tokens": valid,
        "sources": expected_revisions,
        "licenses": licenses,
    }
