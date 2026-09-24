"""Fail-closed admission check for a built Phoenix 350M foundation corpus.

This validator is read-only. It checks shard bytes/hashes, tokenizer identity, source-policy
alignment and a separately produced independent contamination-screen report. It does NOT
create or approve that report and does not start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hf_source_policy50 import SOURCES

SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_RESERVED = {
    "openai/gsm8k",
    "google-research-datasets/mbpp",
    "deepmind/code_contests",
    "google/IFEval",
    "truthfulqa/truthful_qa",
    "google/boolq",
    "allenai/ai2_arc",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON object required: {path}")
    return obj


def _validate_shards(entries, root: Path, role: str) -> tuple[int, set[str]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{role}: nonempty shard manifest required")
    listed = set()
    digests = set()
    tokens = 0
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"path", "tokens", "sha256"}:
            raise ValueError(f"{role}: invalid shard row")
        name = row["path"]
        if not isinstance(name, str) or not name.endswith(".bin") or "/" in name or "\\" in name:
            raise ValueError(f"{role}: unsafe shard path {name!r}")
        if name in listed:
            raise ValueError(f"{role}: duplicate shard path")
        listed.add(name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{role}: shard missing/symlinked: {name}")
        count = int(row["tokens"])
        if count <= 0 or path.stat().st_size != count * 2:
            raise ValueError(f"{role}: uint16 size/token mismatch: {name}")
        digest = str(row["sha256"]).lower()
        if not SHA256.fullmatch(digest) or sha256_file(path) != digest:
            raise ValueError(f"{role}: SHA-256 mismatch: {name}")
        if digest in digests:
            raise ValueError(f"{role}: duplicate shard bytes")
        digests.add(digest)
        tokens += count
    actual = {p.name for p in root.glob("*.bin") if p.is_file() and not p.is_symlink()}
    if actual != listed:
        raise ValueError(f"{role}: manifest/directory shard set mismatch")
    return tokens, digests


def validate(manifest_path: Path, train_root: Path, valid_root: Path,
             tokenizer_sha256: str, benchmark_screen_path: Path) -> dict:
    manifest = read_json(manifest_path)
    if manifest.get("schema") != 2 or manifest.get("architecture") != "frontier350-v1":
        raise ValueError("production Frontier schema 2 manifest required")
    if manifest.get("corpus_role") != "foundation-pretraining":
        raise ValueError("foundation-pretraining corpus role required")

    expected_tok = tokenizer_sha256.lower()
    if not SHA256.fullmatch(expected_tok):
        raise ValueError("expected tokenizer SHA-256 must be lowercase 64-hex")
    tok = manifest.get("tokenizer")
    if not isinstance(tok, dict) or tok.get("tokenizer_sha256") != expected_tok:
        raise ValueError("tokenizer identity mismatch")
    if tok.get("vocab_size") != 16384:
        raise ValueError("unexpected tokenizer vocabulary")

    train_tokens, train_hashes = _validate_shards(manifest.get("train_files"), train_root, "train")
    valid_tokens, valid_hashes = _validate_shards(manifest.get("valid_files"), valid_root, "valid")
    if train_hashes & valid_hashes:
        raise ValueError("identical train/validation shard bytes")
    if train_tokens != int(manifest.get("train_tokens", -1)):
        raise ValueError("train token count mismatch")
    if valid_tokens != int(manifest.get("validation_tokens", -1)):
        raise ValueError("validation token count mismatch")
    if train_tokens + valid_tokens != int(manifest.get("target_tokens", -1)):
        raise ValueError("total token accounting mismatch")

    source_policy = manifest.get("source_policy")
    if not isinstance(source_policy, dict) or not source_policy:
        raise ValueError("builder source-policy receipt missing")
    for name, receipt in source_policy.items():
        if not isinstance(receipt, dict):
            raise ValueError(f"{name}: invalid policy receipt")
        dataset_id = str(receipt.get("dataset_id", ""))
        policy = SOURCES.get(dataset_id)
        if policy is None or not policy.allow_training or policy.role != "pretrain":
            raise ValueError(f"{name}: source is not currently admitted pretraining data: {dataset_id}")
        if receipt.get("role") != policy.role or receipt.get("license") != policy.license:
            raise ValueError(f"{name}: source-policy receipt drift")

    revisions = manifest.get("resolved_revisions")
    if not isinstance(revisions, dict) or set(revisions) != set(source_policy):
        raise ValueError("resolved revision/source-policy key mismatch")
    for name, revision in revisions.items():
        if not HEX40.fullmatch(str(revision).lower()):
            raise ValueError(f"{name}: immutable 40-hex HF revision required")

    # The builder itself must not pretend it performed the independent screen.
    if manifest.get("benchmark_content_included") is not None:
        raise ValueError("builder benchmark_content_included must remain unknown/null")
    if manifest.get("benchmark_screening_status") != "NOT_PERFORMED_BY_THIS_BUILDER_REQUIRES_SEPARATE_AUDIT":
        raise ValueError("unexpected builder benchmark-screening status")

    screen = read_json(benchmark_screen_path)
    if screen.get("schema") != 1 or screen.get("classification") != "independent_foundation_contamination_screen":
        raise ValueError("independent contamination-screen schema required")
    if screen.get("corpus_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("benchmark screen is not bound to this exact corpus manifest")
    if screen.get("approved") is not True or screen.get("independent_review_confirmed") is not True:
        raise ValueError("independent benchmark screen has not approved this corpus")
    for key in ("exact_prompt_matches", "near_duplicate_matches", "answer_leak_matches"):
        if screen.get(key) != 0:
            raise ValueError(f"benchmark contamination screen did not clear: {key}")
    if set(map(str, screen.get("reserved_eval_dataset_ids", []))) != EXPECTED_RESERVED:
        raise ValueError("benchmark screen did not cover the full reserved public-eval registry")

    return {
        "schema": 1,
        "classification": "foundation_corpus_admission_pass",
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "benchmark_screen_sha256": sha256_file(benchmark_screen_path),
        "train_tokens": train_tokens,
        "validation_tokens": valid_tokens,
        "train_shards": len(train_hashes),
        "valid_shards": len(valid_hashes),
        "source_families": sorted(r["dataset_id"] for r in source_policy.values()),
        "tokenizer_sha256": expected_tok,
        "training_started": False,
        "quality_claim": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--valid-root", type=Path, required=True)
    p.add_argument("--tokenizer-sha256", required=True)
    p.add_argument("--benchmark-screen", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(validate(
        args.manifest, args.train_root, args.valid_root,
        args.tokenizer_sha256, args.benchmark_screen,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
