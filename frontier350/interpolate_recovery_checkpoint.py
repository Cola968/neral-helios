from __future__ import annotations

"""Interpolate a continuation candidate back toward its last accepted anchor.

The output is an eval/resume-capable checkpoint whose model weights are
    anchor + alpha * (candidate - anchor)
for every floating model tensor. Optimizer moments must already be reset/empty on the
candidate; they remain empty in the aligned checkpoint. Validation identity is cleared
so a future continuation re-measures its starting validation loss.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from model350 import FRONTIER_350
from recovery_identity import file_sha
from train350 import atomic_save


def interpolate(anchor_path: Path, candidate_path: Path, out_path: Path, alpha: float) -> dict:
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    for name, ck in (("anchor", anchor), ("candidate", candidate)):
        if ck.get("config") != FRONTIER_350.to_dict():
            raise RuntimeError(f"{name} architecture mismatch")
        if ck.get("tokenizer_sha256") != anchor.get("tokenizer_sha256"):
            raise RuntimeError(f"{name} tokenizer mismatch")
        if "model" not in ck:
            raise RuntimeError(f"{name} has no model state")
    if set(anchor["model"]) != set(candidate["model"]):
        raise RuntimeError("model state keys differ")
    if int(candidate.get("processed_tokens", 0)) < int(anchor.get("processed_tokens", 0)):
        raise RuntimeError("candidate predates anchor")
    opt = candidate.get("optimizer")
    if not isinstance(opt, dict) or opt.get("state"):
        raise RuntimeError("candidate optimizer must be reset before alignment")

    merged = {}
    with torch.no_grad():
        for key, base in anchor["model"].items():
            cand = candidate["model"][key]
            if base.shape != cand.shape or base.dtype != cand.dtype:
                raise RuntimeError(f"tensor contract mismatch: {key}")
            if torch.is_floating_point(base):
                value = torch.lerp(base.float(), cand.float(), alpha).to(dtype=cand.dtype)
            else:
                if not torch.equal(base, cand):
                    raise RuntimeError(f"non-floating state changed: {key}")
                value = cand.clone()
            merged[key] = value

    result_ck = dict(candidate)
    # A projected/interpolated evaluation snapshot must never resume stale FP32
    # master weights or Adam moments from the unaligned V9 candidate.
    result_ck.pop('v9_training_state', None)
    result_ck["model"] = merged
    result_ck["best_validation"] = float("inf")
    result_ck["validation_identity"] = ""
    result_ck["parameter_alignment"] = {
        "schema": 1,
        "method": "linear_anchor_interpolation",
        "alpha_candidate": alpha,
        "alpha_anchor": 1.0 - alpha,
        "anchor_checkpoint_sha256": file_sha(anchor_path),
        "candidate_checkpoint_sha256": file_sha(candidate_path),
        "anchor_processed_tokens": int(anchor.get("processed_tokens", 0)),
        "candidate_processed_tokens": int(candidate.get("processed_tokens", 0)),
    }
    atomic_save(result_ck, out_path)
    return {
        "ok": True,
        "alpha": alpha,
        "anchor_tokens": int(anchor.get("processed_tokens", 0)),
        "candidate_tokens": int(candidate.get("processed_tokens", 0)),
        "out_sha256": file_sha(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()
    print(
        json.dumps(
            interpolate(Path(args.anchor), Path(args.candidate), Path(args.out), args.alpha)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
