from __future__ import annotations

"""Fail-closed security/data policy for Neral 350M Frontier.

The gate validates immutable source revisions, admitted licenses, tokenizer identity,
benchmark separation, shard hashes and token accounting before expensive training begins.
It also prevents accidental checkpoint/corpus uploads in GitHub report artifacts.
"""

import hashlib
import json
import re
from pathlib import Path

FORBIDDEN_NAME_PATTERNS = (
    "nis2", "benchmark", "holdout", "rawmath_eval", "scorecard", "release_gate"
)
ALLOWED_LICENSE_MARKERS = {
    "apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc-by-sa-4.0", "odc-by", "public-domain"
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_file(root: Path, subdir: str, item: dict) -> tuple[Path, int]:
    name = str(item.get("path", "")).replace("\\", "/").strip()
    expected = str(item.get("sha256", "")).lower().strip()
    tokens = int(item.get("tokens", -1))
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"unsafe shard name: {name!r}")
    if not name.endswith(".bin"):
        raise ValueError(f"unexpected shard extension: {name}")
    if any(x in name.lower() for x in FORBIDDEN_NAME_PATTERNS):
        raise ValueError(f"benchmark-like shard forbidden: {name}")
    if not HEX64.fullmatch(expected):
        raise ValueError(f"missing shard sha256: {name}")
    if tokens <= 1:
        raise ValueError(f"invalid token count: {name}:{tokens}")
    p = (root / subdir / name).resolve()
    expected_parent = (root / subdir).resolve()
    if not p.is_relative_to(expected_parent):
        raise ValueError(f"path escape: {name}")
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.stat().st_size != tokens * 2:
        raise ValueError(f"uint16 byte-size/token mismatch: {name}")
    actual = sha256(p)
    if actual != expected:
        raise ValueError(f"hash mismatch: {name}")
    return p, tokens


def validate_manifest(path: Path, root: Path) -> dict:
    root = root.resolve()
    obj = json.loads(path.read_text(encoding="utf8"))
    if obj.get("schema") != 1:
        raise ValueError("unsupported manifest schema")
    if obj.get("architecture") != "frontier350-v1":
        raise ValueError("wrong architecture in manifest")
    if obj.get("benchmark_content_included") is not False:
        raise ValueError("benchmark separation not explicitly guaranteed")

    tok = obj.get("tokenizer") or {}
    expected_tok = {
        "type": "utf8-byte-v1", "byte_ids": [0, 255],
        "bos_id": 256, "eos_id": 257, "sep_id": 258, "pad_id": 259, "vocab_size": 260,
    }
    if tok != expected_tok:
        raise ValueError(f"tokenizer contract mismatch: {tok}")

    revisions = obj.get("resolved_revisions")
    if not isinstance(revisions, dict) or not revisions:
        raise ValueError("resolved revisions missing")
    for source_id, rev in revisions.items():
        if not source_id or not HEX40.fullmatch(str(rev).lower()):
            raise ValueError(f"unpinned revision: {source_id}:{rev}")

    licenses = obj.get("licenses")
    if not isinstance(licenses, list) or not licenses:
        raise ValueError("licenses missing")
    for lic in licenses:
        if str(lic).lower() not in ALLOWED_LICENSE_MARKERS:
            raise ValueError(f"unapproved license: {lic}")

    sources = obj.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source reports missing")
    for report in sources:
        spec = report.get("source") or {}
        sid = str(spec.get("name", ""))
        repo = str(spec.get("repo", ""))
        lic = str(spec.get("license", "")).lower()
        rev = str(report.get("revision", "")).lower()
        if not sid or not repo:
            raise ValueError("malformed source report")
        if any(x in (sid + " " + repo).lower() for x in FORBIDDEN_NAME_PATTERNS):
            raise ValueError(f"benchmark-like source forbidden: {sid}")
        if lic not in ALLOWED_LICENSE_MARKERS:
            raise ValueError(f"unapproved source license: {sid}:{lic}")
        if not HEX40.fullmatch(rev):
            raise ValueError(f"unpinned source revision: {sid}")
        if revisions.get(sid) != rev:
            raise ValueError(f"revision map mismatch: {sid}")
        emitted = int(report.get("emitted_tokens", 0))
        if emitted <= 0:
            raise ValueError(f"empty admitted source: {sid}")

    train_items = obj.get("train_files")
    valid_items = obj.get("valid_files")
    if not isinstance(train_items, list) or not train_items:
        raise ValueError("train shards missing")
    if not isinstance(valid_items, list) or not valid_items:
        raise ValueError("validation shards missing")

    train_tokens = 0
    valid_tokens = 0
    for item in train_items:
        _, n = _safe_file(root, "train", item)
        train_tokens += n
    for item in valid_items:
        _, n = _safe_file(root, "valid", item)
        valid_tokens += n

    if train_tokens != int(obj.get("train_tokens", -1)):
        raise ValueError("train token accounting mismatch")
    if valid_tokens != int(obj.get("validation_tokens", -1)):
        raise ValueError("validation token accounting mismatch")
    if train_tokens + valid_tokens != int(obj.get("target_tokens", -1)):
        raise ValueError("total token accounting mismatch")
    if train_tokens <= valid_tokens:
        raise ValueError("validation reserve unexpectedly large")

    return {
        "ok": True,
        "architecture": obj["architecture"],
        "sources": len(revisions),
        "train_shards": len(train_items),
        "valid_shards": len(valid_items),
        "train_tokens": train_tokens,
        "validation_tokens": valid_tokens,
        "licenses": licenses,
    }


def reject_sensitive_uploads(root: Path) -> None:
    bad_ext = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".parquet", ".gz", ".zst"}
    bad = []
    for p in root.rglob("*"):
        if p.is_file() and (
            p.suffix.lower() in bad_ext or "token" in p.name.lower() or "secret" in p.name.lower()
        ):
            bad.append(str(p))
    if bad:
        raise RuntimeError("sensitive artifacts present in upload/report directory: " + ", ".join(bad[:10]))
