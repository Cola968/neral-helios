"""Read-only, strictly checked Arctus350 continuation metadata probe.

Use on an already trusted local PC after verifying the original SHA.
Never saves, exports, uploads, resumes or trains model weights. Only bounded
booleans and token counts are printed; actual tensors stay on the local host.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier350.chat350_contract import EXPECTED_CONFIG, EXPECTED_TOKENIZER_SHA256
from frontier350.vast4090_guard import make_plan, sha256_file, SHA_PATTERN


def inspect_metadata(meta: dict) -> dict:
    """Check whether native continuation metadata is structurally compatible.

    This is not evidence that the BPE corpus or future GPU training is safe.
    """
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    model = isinstance(meta.get("model"), dict) and bool(meta["model"])
    optimizer = isinstance(meta.get("optimizer"), dict) and bool(meta["optimizer"])
    rng = all(meta.get(field) is not None
              for field in ("torch_rng", "python_rng", "numpy_rng"))
    processed = meta.get("processed_tokens")
    target = meta.get("target_tokens")
    schedule = (type(processed) is int and type(target) is int and
                target > processed >= 0)
    reasons = []
    if not model:
        reasons.append("missing_model_state")
    if not optimizer:
        reasons.append("missing_optimizer_state")
    if not rng:
        reasons.append("incomplete_rng_state")
    if meta.get("config") != EXPECTED_CONFIG:
        reasons.append("architecture_mismatch")
    if meta.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256:
        reasons.append("tokenizer_identity_mismatch")
    if not schedule:
        reasons.append("missing_or_exhausted_training_schedule")
    try:
        make_plan(meta, extra_tokens=8192)
    except ValueError:
        reasons.append("native_resume_guard_rejected_metadata")
    return {
        "metadata_schema": 1,
        "has_model": bool(model),
        "has_optimizer": bool(optimizer),
        "rng_complete": bool(rng),
        "schedule_has_remaining_tokens": bool(schedule),
        "processed_tokens": processed if type(processed) is int else None,
        "target_tokens": target if type(target) is int else None,
        "step": meta.get("step") if type(meta.get("step")) is int else None,
        "resume_checkpoint_metadata_compatible": not reasons,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "training_started": False,
        "corpus_verified": False,
    }


def inspect_file(path: Path, expected_sha256: str, *, checkpoint_loader=None) -> dict:
    file = Path(path)
    if file.is_symlink() or not file.is_file():
        raise ValueError("Checkpoint must be a real local file, not a link")
    if not isinstance(expected_sha256, str) or not SHA_PATTERN.fullmatch(expected_sha256):
        raise ValueError("Expected checkpoint SHA256 must be 64 lowercase hex digits")
    if sha256_file(file) != expected_sha256:
        raise ValueError("Original checkpoint SHA256 mismatch: no deserialization")
    if checkpoint_loader is None:
        from frontier350.secure_checkpoint350 import load_checkpoint
        checkpoint_loader = load_checkpoint
    meta = checkpoint_loader(file)
    result = inspect_metadata(meta)
    result.update({
        "checkpoint_sha256_verified": True,
        "private_checkpoint_uploaded": False,
        "original_checkpoint_modified": False,
        "metadata_only_no_real_training": True,
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    result = inspect_file(args.checkpoint, args.expected_sha256)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    # An incomplete original checkpoint is a valid diagnostic, not a
    # command to auto-run a new training schedule.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
