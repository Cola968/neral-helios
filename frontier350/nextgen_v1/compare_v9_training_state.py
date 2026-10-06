"""Compare the COMPLETE V9 continuation state of two trusted local experiment snapshots.

This is an experiment diagnostic, not a production gate. Only load trusted local
checkpoints: the legacy checkpoint format includes Python objects and requires
pickle deserialization. Do not download/deserialize third-party checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

SHA = re.compile(r"[0-9a-f]{64}\Z")
FIELDS = ("contract", "masters", "optimizer", "sampler_rng", "torch_rng",
          "cuda_rng", "new_tokens", "updates", "batch_chain", "source_tokens")


def first_difference(left, right, path="state"):
    if type(left) is not type(right):
        return f"{path}: type differs"
    if torch.is_tensor(left):
        if left.dtype != right.dtype or left.shape != right.shape:
            return f"{path}: tensor shape/dtype differs"
        return None if torch.equal(left, right) else f"{path}: tensor values differ"
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            return f"{path}: array shape/dtype differs"
        return None if np.array_equal(left, right) else f"{path}: array values differ"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: mapping keys differ"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}[{key!r}]")
            if difference:
                return difference
        return None
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{path}: sequence length differs"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else f"{path}: scalar differs"


def validate(ck, anchor_sha, expected_updates, expected_tokens):
    if not isinstance(ck, dict) or not isinstance(ck.get("v9_training_state"), dict):
        raise ValueError("V9 checkpoint state missing")
    state = ck["v9_training_state"]
    if any(key not in state for key in FIELDS):
        raise ValueError("incomplete V9 state")
    if not isinstance(state["contract"], dict) or state["contract"].get("anchor_sha256") != anchor_sha:
        raise ValueError("source anchor mismatch")
    if state["updates"] != expected_updates or state["new_tokens"] != expected_tokens:
        raise ValueError("unexpected progress counters")
    if ck.get("processed_tokens") != ck["v9_training_state"]["new_tokens"] + ck.get("v9_provenance", {}).get("source_processed_tokens", ck.get("processed_tokens") - state["new_tokens"]):
        raise ValueError("processed token arithmetic invalid")
    if not isinstance(state["batch_chain"], str) or not SHA.fullmatch(state["batch_chain"]):
        raise ValueError("invalid sample hash chain")
    if not isinstance(state["source_tokens"], dict) or sum(state["source_tokens"].values()) != expected_tokens:
        raise ValueError("source token counters do not add up")
    if not isinstance(state["masters"], dict) or not state["masters"] or not isinstance(state["optimizer"], dict):
        raise ValueError("missing trainable masters or optimizer")
    if ck.get("v9_provenance", {}).get("anchor_sha256") != anchor_sha:
        raise ValueError("provenance anchor mismatch")
    if ck.get("v9_provenance", {}).get("frozen_weights_unchanged") is not True:
        raise ValueError("no frozen-weight preservation assertion")
    return state


def hash_file(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("snapshot must be a nonempty regular file, not symlink")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--sha-a", required=True)
    parser.add_argument("--sha-b", required=True)
    parser.add_argument("--anchor-sha", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--acknowledge-trusted-local-pickle", action="store_true", required=True)
    args = parser.parse_args()
    result = {"schema": 1, "status": "rejected", "eligible_for_production": False,
              "real_training_quality_improved": False, "end_to_end_44k_verified": False}
    try:
        if args.arm_a.resolve() == args.arm_b.resolve():
            raise ValueError("two distinct output files required")
        for sha in (args.sha_a, args.sha_b, args.anchor_sha):
            if not SHA.fullmatch(sha):
                raise ValueError("invalid pinned SHA")
        if args.updates <= 0 or args.tokens <= 0:
            raise ValueError("positive progress expected")
        if hash_file(args.arm_a) != args.sha_a or hash_file(args.arm_b) != args.sha_b:
            raise ValueError("snapshot digest mismatch")
        # Only our own freshly generated local artifacts are permitted here.
        a = torch.load(args.arm_a, map_location="cpu", weights_only=False)
        b = torch.load(args.arm_b, map_location="cpu", weights_only=False)
        if hash_file(args.arm_a) != args.sha_a or hash_file(args.arm_b) != args.sha_b:
            raise ValueError("snapshot changed during inspection")
        sa = validate(a, args.anchor_sha, args.updates, args.tokens)
        sb = validate(b, args.anchor_sha, args.updates, args.tokens)
        difference = first_difference(sa, sb)
        if difference is None:
            difference = first_difference(a["model"], b["model"], "full_model")
        if difference is None:
            difference = first_difference(a["processed_tokens"], b["processed_tokens"], "processed_tokens")
        result.update(status="match" if difference is None else "mismatch",
                      state_and_all_model_weights_equal=difference is None,
                      reason=difference or "full V9 state and every model tensor equal",
                      anchor_sha256=args.anchor_sha, updates=args.updates,
                      training_tokens=args.tokens, sha_a=args.sha_a, sha_b=args.sha_b)
        print(json.dumps(result, sort_keys=True))
        return 0 if difference is None else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
