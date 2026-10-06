"""Read-only, research-only comparison of two trusted V7 recovery snapshots.

This compares all serialized fields, not merely weight checksums or counters.
A match proves equality of these TWO artifacts only. It does not prove that
Neral350M ran, that source data was licensed, that frozen weights match, that
GPU kernels are numerically equivalent, or that a checkpoint can be promoted.

WARNING: V7 embeds NumPy RNG state and requires torch.load(weights_only=False).
Only use trusted, independently pinned LOCAL artifacts, never downloaded or
user-submitted untrusted pickle files. No network/GPU/training/write operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED = (
    "schema", "kind", "anchor_sha256", "source_processed_tokens",
    "processed_tokens", "new_tokens", "updates", "semantic_contract",
    "masters", "optimizer", "sampler_rng", "torch_rng", "cuda_rng",
    "numpy_rng", "python_rng", "batch_chain", "source_tokens",
)


def _validate(state: object) -> str | None:
    if not isinstance(state, dict):
        return "state is not a dictionary"
    for key in _REQUIRED:
        if key not in state:
            return f"missing {key}"
    if state["schema"] != 7 or state["kind"] != "neral350-v7-exact-recovery":
        return "incorrect recovery schema or kind"
    if not isinstance(state["anchor_sha256"], str) or not _HEX64.fullmatch(state["anchor_sha256"]):
        return "invalid anchor SHA-256"
    if not isinstance(state["batch_chain"], str) or not _HEX64.fullmatch(state["batch_chain"]):
        return "invalid input hash chain"
    for key in ("source_processed_tokens", "processed_tokens", "new_tokens", "updates"):
        if type(state[key]) is not int or state[key] < 0:
            return f"invalid {key}"
    if not state["new_tokens"] or not state["updates"]:
        return "no recorded training progress"
    if state["processed_tokens"] != state["source_processed_tokens"] + state["new_tokens"]:
        return "inconsistent processed-token counters"
    if not isinstance(state["source_tokens"], dict) or not state["source_tokens"]:
        return "missing per-source token counters"
    if any(type(v) is not int or v < 0 for v in state["source_tokens"].values()):
        return "invalid per-source token counter"
    if sum(state["source_tokens"].values()) != state["new_tokens"]:
        return "per-source token totals disagree with new_tokens"
    if not isinstance(state["masters"], dict) or not state["masters"]:
        return "missing trainable master weights"
    if not isinstance(state["optimizer"], dict) or not state["optimizer"]:
        return "missing optimizer state"
    if not isinstance(state["semantic_contract"], dict) or not state["semantic_contract"]:
        return "missing semantic contract"
    return None


def _first_difference(left: object, right: object, path: str = "state") -> str | None:
    if type(left) is not type(right):
        return f"{path}: value types differ"
    if torch.is_tensor(left):
        if left.dtype != right.dtype or left.shape != right.shape:
            return f"{path}: tensor dtype/shape differs"
        return None if torch.equal(left, right) else f"{path}: tensor values differ"
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            return f"{path}: array dtype/shape differs"
        return None if np.array_equal(left, right) else f"{path}: array values differ"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: mapping keys differ"
        for key in left:
            mismatch = _first_difference(left[key], right[key], f"{path}[{key!r}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{path}: sequence lengths differ"
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = _first_difference(a, b, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    return None if left == right else f"{path}: scalar values differ"


def compare_states(left: object, right: object) -> tuple[bool, str]:
    """Compare all fields, including optimizer, masters, four RNGs and data chain."""
    for label, value in (("left", left), ("right", right)):
        issue = _validate(value)
        if issue:
            return False, f"{label}: {issue}"
    mismatch = _first_difference(left, right)
    return (False, mismatch) if mismatch else (True, "all serialized fields equal")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_file(path: Path, expected_sha: str) -> None:
    if not _HEX64.fullmatch(expected_sha):
        raise ValueError("expected file checksum must be lowercase 64-character SHA-256")
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("input must be an existing nonempty ordinary local file, not a symlink")
    if _sha256(path) != expected_sha:
        raise ValueError("checkpoint bytes differ from independently pinned SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--uninterrupted-sha256", required=True)
    parser.add_argument("--resumed-sha256", required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--acknowledge-trusted-local-pickle", action="store_true", required=True,
                        help="Required: confirms that BOTH local torch.load pickle files are trusted")
    args = parser.parse_args()
    result = {"schema": 1, "status": "rejected", "exact_recovery_state_equal": False,
              "eligible_for_production": False, "real_gpu_training_verified": False}
    try:
        if args.uninterrupted.resolve() == args.resumed.resolve():
            raise ValueError("A/B inputs must be two distinct files")
        if not _HEX64.fullmatch(args.anchor_sha256):
            raise ValueError("expected anchor SHA must be lowercase 64-character SHA-256")
        _trusted_file(args.uninterrupted, args.uninterrupted_sha256)
        _trusted_file(args.resumed, args.resumed_sha256)
        # V7's NumPy/Python RNG format is not compatible with weights_only=True.
        # Digest-pinned bytes MUST originate from separately trusted local files.
        a = torch.load(args.uninterrupted, map_location="cpu", weights_only=False)
        b = torch.load(args.resumed, map_location="cpu", weights_only=False)
        if _sha256(args.uninterrupted) != args.uninterrupted_sha256 or _sha256(args.resumed) != args.resumed_sha256:
            raise ValueError("a checkpoint changed during the audit")
        for label, value in (("uninterrupted", a), ("resumed", b)):
            if not isinstance(value, dict) or value.get("anchor_sha256") != args.anchor_sha256:
                raise ValueError(f"{label} anchor mismatch")
        same, reason = compare_states(a, b)
        result.update(status="match" if same else "mismatch",
                      exact_recovery_state_equal=same, reason=reason,
                      anchor_sha256=args.anchor_sha256,
                      new_tokens=a.get("new_tokens"),
                      uninterrupted_sha256=args.uninterrupted_sha256,
                      resumed_sha256=args.resumed_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0 if same else 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
