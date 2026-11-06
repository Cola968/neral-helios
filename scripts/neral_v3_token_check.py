"""Verified local 512-token and completion-mask preflight for NERAL v3.

Loads the already-installed tokenizer only. Never runs inference or training,
downloads a model, sends prompts to an API or uploads a local corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


def preflight(trainer: Path, base: Path, dataset: Path, review: Path, max_length: int = 512) -> dict:
    if max_length != 512:
        raise ValueError("Only the approved 512-token profile is accepted")
    trainer, base = trainer.resolve(), base.resolve()
    if not trainer.is_dir() or trainer.is_symlink() or not base.is_dir() or base.is_symlink():
        raise ValueError("Pinned local trainer and model folders required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    sys.path.insert(0, str(trainer))
    for name, filename in (
        ("model_catalog", "model_catalog.py"),
        ("qwen_local_runtime", "qwen_local_runtime.py"),
        ("scripts.neral_qlora_pilot", "scripts/neral_qlora_pilot.py"),
    ):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (trainer / filename).resolve():
            raise ValueError("Unpinned tokenizer/trainer Python import refused")
    from model_catalog import validate_hf_snapshot
    from scripts.neral_qlora_pilot import verify_review, read_examples, tokenize_examples
    snap = validate_hf_snapshot(base, verify_hashes=True)
    if snap["base_model"] != "Qwen/Qwen2.5-Coder-7B-Instruct":
        raise ValueError("Unexpected local foundation model")
    verify_review(dataset, review)
    from transformers import Qwen2TokenizerFast
    tok = Qwen2TokenizerFast.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False, token=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = read_examples(dataset)
    if len(raw) != 842:
        raise ValueError("Exactly 842 reviewed v3 records are required")
    sequences = tokenize_examples(raw, tok, max_length)
    if len(sequences) != 842:
        raise ValueError("Tokenizer skipped or duplicated records")
    lengths = []
    for row in sequences:
        ids = row["input_ids"]
        mask = row["completion_mask"]
        if len(ids) != len(mask) or not ids or len(ids) > max_length:
            raise ValueError("Token/mask mismatch or overflow")
        if not set(mask) <= {0, 1} or mask.count(1) <= 0:
            raise ValueError("Missing completion-only supervision")
        first = mask.index(1)
        if any(mask[:first]) or any(x == 0 for x in mask[first:]):
            raise ValueError("Invalid completion-only boundary")
        lengths.append(len(ids))
    return {
        "schema": 1, "status": "NERAL_V3_512_TOKEN_GATE_PASS",
        "samples": len(sequences), "max_sequence_tokens": max(lengths),
        "max_allowed_tokens": max_length, "overflow": 0,
        "completion_only_masks_valid": True,
        "training_performed": False, "promoted": False,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }


def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False)
    for arg in ("trainer", "base", "dataset", "review", "report"):
        p.add_argument("--" + arg, required=True, type=Path)
    args = p.parse_args()
    if args.report.exists() or args.report.is_symlink():
        raise ValueError("Token report must not overwrite existing output")
    result = preflight(args.trainer, args.base, args.dataset, args.review)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
