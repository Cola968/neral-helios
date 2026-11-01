"""Arctus350 BPE corpus METADATA-ONLY inventory, read-only and stdlib-only.

This inspection is explicitly NOT a SHA-256 shard audit, benchmark leakage
check or a training approval. It never serializes paths or source text into
reports, downloads datasets, reads BPE file contents or loads model weights.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from frontier350.chat350_contract import EXPECTED_TOKENIZER_SHA256

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SHARDS = 2000
MAX_TOKENS_PER_SHARD = 10**11
SHARD_NAME = re.compile(r"[A-Za-z0-9_-]{1,110}\.bin")
SHA256 = re.compile(r"[a-f0-9]{64}")


class ProbeError(ValueError):
    """No private file path or source text is included in error messages."""


def _empty(*, present=False, issue=None):
    return {
        "present":present,
        "format_supported":False,
        "metadata_compatible":False,
        "train_shards":0,
        "valid_shards":0,
        "train_tokens_declared":0,
        "valid_tokens_declared":0,
        "sha256_verified":False,
        "corpus_provenance_verified":False,
        "training_ready":False,
        "issue":issue,
    }


def inspect_corpus(root: Path) -> dict:
    """Inspect up to 2K shard file *sizes*; NEVER open .bin shard contents."""
    path=Path(root)
    if not path.is_dir() or path.is_symlink():
        return _empty(issue="corpus_missing_or_link")
    manifest=path/"manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        return _empty(present=True,issue="manifest_missing_or_link")
    try:
        if manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise ProbeError("manifest_invalid")
        data=json.loads(manifest.read_text(encoding="utf8"))
        if not isinstance(data,dict):
            raise ProbeError("manifest_invalid")
        if (data.get("schema") != 2 or
                data.get("architecture") != "frontier350-v1"):
            raise ProbeError("format_mismatch")
        if (data.get("benchmark_content_included") is not False or
                not isinstance(data.get("tokenizer"),dict) or
                data["tokenizer"].get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256):
            raise ProbeError("identity_mismatch")

        result=_empty(present=True)
        result["format_supported"]=True
        for part,key in (("train","train_files"),("valid","valid_files")):
            entries=data.get(key)
            if not isinstance(entries,list) or not 1 <= len(entries) <= MAX_SHARDS:
                raise ProbeError("inventory_invalid")
            directory=path/part
            if not directory.is_dir() or directory.is_symlink():
                raise ProbeError("shard_directory_invalid")
            expected=set()
            n_tokens=0
            for entry in entries:
                if not isinstance(entry,dict):
                    raise ProbeError("inventory_invalid")
                rel=entry.get("path")
                tokens=entry.get("tokens")
                sha=entry.get("sha256")
                if (not isinstance(rel,str) or not SHARD_NAME.fullmatch(rel) or
                        rel in expected or type(tokens) is not int or
                        not 513 <= tokens <= MAX_TOKENS_PER_SHARD or
                        not isinstance(sha,str) or not SHA256.fullmatch(sha)):
                    raise ProbeError("inventory_invalid")
                expected.add(rel)
                shard=directory/rel
                if shard.is_symlink() or not shard.is_file():
                    raise ProbeError("shard_missing_or_link")
                if shard.stat().st_size != 2*tokens:
                    raise ProbeError("shard_size_mismatch")
                n_tokens+=tokens
            actual={f.name for f in directory.glob("*.bin")}
            if actual != expected:
                raise ProbeError("inventory_mismatch")
            result[f"{part}_shards"]=len(expected)
            result[f"{part}_tokens_declared"]=n_tokens

        result["metadata_compatible"]=True
        result["issue"]=None
        return result
    except ProbeError as exc:
        # Even diagnostic reports must not accidentally disclose filenames.
        return _empty(present=True,issue=str(exc))
    except (OSError,UnicodeError,ValueError,OverflowError):
        return _empty(present=True,issue="manifest_invalid")


def inspect_candidates(candidates: list[Path]) -> dict:
    if not isinstance(candidates,(tuple,list)) or not 1 <= len(candidates) <= 5:
        raise ValueError("Choose between 1 and 5 explicit candidate locations")
    data=[]
    for idx,root in enumerate(candidates,1):
        line=inspect_corpus(root)
        line["candidate_index"]=idx
        data.append(line)
    return {
        "schema":1,
        "inspection_kind":"read_only_metadata_without_shard_hashes",
        "candidates":data,
        "compatible_candidates":sum(int(v["metadata_compatible"]) for v in data),
        "sha256_verified":False,
        "corpus_provenance_verified":False,
        "private_data_uploaded":False,
        "training_started":False,
        "training_ready":False,
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate",action="append",type=Path,required=True,
                    help="Explicit local corpus directory (not printed in reports)")
    args=ap.parse_args(argv)
    report=inspect_candidates(args.candidate)
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
