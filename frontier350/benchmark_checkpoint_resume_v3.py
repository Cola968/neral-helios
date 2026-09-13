from __future__ import annotations

"""Compact-resume serialization probe for Neral 350M.

This does not alter the production trainer. It proves whether a resume artifact can store
only mutable last-four-block tensors plus training state and reconstruct the exact canonical
checkpoint from an immutable SHA-pinned base anchor. No model promotion is allowed.
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch

from model350 import FRONTIER_350
from recovery_identity import file_sha

TRAINABLE_PREFIXES = tuple(f"blocks.{i}." for i in range(FRONTIER_350.n_layers - 4, FRONTIER_350.n_layers))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def discover_anchor(assets: Path) -> tuple[Path, str, int]:
    state = assets / "frontier350_scale_v3_state.json"
    if state.is_file():
        obj = read_json(state)
        path = Path(obj["safe_anchor_path"])
        if path.is_file() and file_sha(path) == obj["safe_anchor_sha256"]:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if int(ck["processed_tokens"]) == int(obj["safe_anchor_tokens"]):
                return path, obj["safe_anchor_sha256"], int(obj["safe_anchor_tokens"])
    rows = []
    for path in (assets / "runs").glob("neral350_*_selected/resume.pt"):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            rows.append((int(ck["processed_tokens"]), path, file_sha(path)))
        except Exception:
            pass
    if not rows:
        raise RuntimeError("no promoted anchor found")
    tokens, path, sha = max(rows, key=lambda x: x[0])
    return path, sha, tokens


def trainable_model_state(model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.items() if k.startswith(TRAINABLE_PREFIXES)}


def build_compact(full: dict, base_path: Path, base_sha: str) -> dict:
    mutable = trainable_model_state(full["model"])
    keep = {
        "schema": 1,
        "base_anchor_path": str(base_path),
        "base_anchor_sha256": base_sha,
        "processed_tokens": int(full["processed_tokens"]),
        "step": int(full.get("step", 0)),
        "mutable_model": mutable,
        "v9_training_state": full.get("v9_training_state"),
        "v9_provenance": full.get("v9_provenance"),
        "best_validation": full.get("best_validation"),
        "validation_identity": full.get("validation_identity"),
        "tokenizer_sha256": full.get("tokenizer_sha256"),
        "config": full.get("config"),
    }
    return keep


def reconstruct(compact: dict, base_path: Path, base_sha: str) -> dict:
    if compact["base_anchor_sha256"] != base_sha or file_sha(base_path) != base_sha:
        raise ValueError("base anchor identity changed")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if base.get("config") != compact.get("config") or base.get("tokenizer_sha256") != compact.get("tokenizer_sha256"):
        raise ValueError("base architecture/tokenizer mismatch")
    out = copy.deepcopy(base)
    model = dict(out["model"])
    for name, value in compact["mutable_model"].items():
        if not name.startswith(TRAINABLE_PREFIXES):
            raise ValueError("compact artifact contains non-trainable tensor")
        model[name] = value
    out["model"] = model
    out["processed_tokens"] = int(compact["processed_tokens"])
    out["step"] = int(compact["step"])
    for key in ("v9_training_state", "v9_provenance", "best_validation", "validation_identity"):
        if compact.get(key) is not None:
            out[key] = compact[key]
    return out


def same_model(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def timed_save(obj: object, path: Path) -> float:
    start = time.perf_counter()
    torch.save(obj, path)
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scratch = Path(a.scratch)
    if scratch.exists():
        raise FileExistsError("scratch already exists")
    scratch.mkdir(parents=True)
    anchor, sha, tokens = discover_anchor(Path(a.assets))
    full = torch.load(anchor, map_location="cpu", weights_only=False)
    if full.get("config") != FRONTIER_350.to_dict():
        raise ValueError("architecture mismatch")

    compact = build_compact(full, anchor, sha)
    full_path = scratch / "full.pt"
    compact_path = scratch / "compact.pt"
    full_save_s = timed_save(full, full_path)
    compact_save_s = timed_save(compact, compact_path)

    start = time.perf_counter()
    loaded_compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    rebuilt = reconstruct(loaded_compact, anchor, sha)
    rebuild_s = time.perf_counter() - start

    exact_model = same_model(full["model"], rebuilt["model"])
    exact_tokens = int(full["processed_tokens"]) == int(rebuilt["processed_tokens"])
    exact_step = int(full.get("step", 0)) == int(rebuilt.get("step", 0))
    full_bytes = os.path.getsize(full_path)
    compact_bytes = os.path.getsize(compact_path)
    reduction = 1.0 - compact_bytes / max(1, full_bytes)
    save_speedup = full_save_s / max(compact_save_s, 1e-9)

    report = {
        "schema": 1,
        "probe": "neral350-compact-resume-v3",
        "anchor": str(anchor),
        "anchor_sha256": sha,
        "anchor_tokens": tokens,
        "trainable_prefixes": list(TRAINABLE_PREFIXES),
        "full_bytes": full_bytes,
        "compact_bytes": compact_bytes,
        "size_reduction_fraction": reduction,
        "full_save_seconds": full_save_s,
        "compact_save_seconds": compact_save_s,
        "save_speedup": save_speedup,
        "reconstruct_seconds": rebuild_s,
        "roundtrip": {
            "model_bit_exact": exact_model,
            "processed_tokens_exact": exact_tokens,
            "step_exact": exact_step,
            "base_sha_reverified": file_sha(anchor) == sha,
        },
        "eligible_for_canary": bool(exact_model and exact_tokens and exact_step and reduction > 0.0),
        "production_integration": False,
        "promotion_allowed": False,
        "requires_guarded_canary": True,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
