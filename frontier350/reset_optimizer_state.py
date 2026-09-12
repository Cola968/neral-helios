from __future__ import annotations

"""Create a checkpoint with identical model weights but fresh AdamW moments.

This is used when the continuation data distribution changes. Reusing Adam moments from
an earlier pretraining phase can keep pushing parameters in stale directions even when
the new corpus and LR have changed. The model, token counters, RNG state, architecture,
and tokenizer contract are preserved; only optimizer moment tensors are discarded.
"""

import argparse
import json
from pathlib import Path

import torch

from model350 import FRONTIER_350
from recovery_identity import file_sha
from train350 import atomic_save


def reset_optimizer(src: Path, out: Path, reason: str) -> dict:
    src = Path(src)
    out = Path(out)
    if src.resolve() == out.resolve():
        raise RuntimeError("source and output checkpoint must differ")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture mismatch")
    if "model" not in ck or "optimizer" not in ck:
        raise RuntimeError("checkpoint is not resumable")
    opt = ck["optimizer"]
    if not isinstance(opt, dict) or not isinstance(opt.get("param_groups"), list):
        raise RuntimeError("unexpected optimizer state")

    old_entries = len(opt.get("state", {}))
    groups = []
    for group in opt["param_groups"]:
        copy = dict(group)
        copy["params"] = list(group["params"])
        groups.append(copy)
    ck["optimizer"] = {"state": {}, "param_groups": groups}
    ck["optimizer_reset"] = {
        "schema": 1,
        "reason": reason,
        "source_checkpoint_sha256": file_sha(src),
        "discarded_state_entries": old_entries,
        "processed_tokens_unchanged": int(ck.get("processed_tokens", 0)),
    }
    atomic_save(ck, out)
    result = {
        "ok": True,
        "out": str(out),
        "source_checkpoint_sha256": ck["optimizer_reset"]["source_checkpoint_sha256"],
        "output_checkpoint_sha256": file_sha(out),
        "discarded_state_entries": old_entries,
        "processed_tokens": int(ck.get("processed_tokens", 0)),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reason", default="distribution_shift_safe_continuation")
    args = ap.parse_args()
    print(json.dumps(reset_optimizer(Path(args.src), Path(args.out), args.reason)), flush=True)


if __name__ == "__main__":
    main()
