"""Two in-memory full-model optimizer steps, never a production or checkpoint run.

Requires independent GitHub workflow preflight on exact SAFE and all corpus shards.
Loads only pinned private own Neral weights; does not serialize or upload anything.
BF16 AdamW optimizer matches native trainer's dtype path; NOT FP32-master training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from model350 import FRONTIER_350, Neral350M

SAFE_SHA = "ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c"
TOKENIZER_SHA = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
EXPECTED_PARAMETERS = 349_825_152
SEQ = 128
STEPS = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("unexpected GPU")
    if sha256(args.anchor) != SAFE_SHA or sha256(args.tokenizer) != TOKENIZER_SHA:
        raise ValueError("protected checkpoint or tokenizer identity mismatch")
    shards = sorted(args.train_data.glob("*.bin"))
    if not shards or any(p.is_symlink() for p in shards):
        raise ValueError("training corpus missing or symlinked")
    shard = np.memmap(shards[0], dtype=np.uint16, mode="r")
    if len(shard) < (STEPS * (SEQ + 1)):
        raise ValueError("verified corpus too short for distinct real data windows")
    source = torch.load(args.anchor, map_location="cpu", weights_only=False)
    if source.get("config") != FRONTIER_350.to_dict() or source.get("tokenizer_sha256") != TOKENIZER_SHA:
        raise ValueError("SAFE architecture/tokenizer mismatch")
    if source.get("processed_tokens") != 1_010_003_584:
        raise ValueError("unexpected SAFE token history")
    torch.manual_seed(35026)
    model = Neral350M().to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(source["model"], strict=True)
    del source
    model.train()
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    if total != EXPECTED_PARAMETERS or not all(p.requires_grad for p in params):
        raise RuntimeError("ALL unique model parameters must be trainable")
    opt = torch.optim.AdamW(params, lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"event": "full_model_smoke_start", "parameters_trainable": total,
                      "steps_max": STEPS, "seq": SEQ, "batch": 1,
                      "new_optimizer_lineage": True, "durable_training_tokens": 0,
                      "source": "sha_verified_own_SAFE", "no_checkpoint_writes": True}), flush=True)
    for step in range(STEPS):
        window = np.asarray(shard[step * (SEQ + 1):(step + 1) * (SEQ + 1)], dtype=np.int64)
        if window.min() < 0 or window.max() >= FRONTIER_350.vocab_size:
            raise ValueError("bad real corpus token ID")
        torch.cuda.synchronize()
        started = time.perf_counter()
        x = torch.tensor(window[:-1].copy(), device="cuda").view(1, SEQ)
        y = torch.tensor(window[1:].copy(), device="cuda").view(1, SEQ)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite full-model loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite full-model gradient")
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(opt.state) != len(params):
            raise RuntimeError("full-model optimizer did not initialize all unique parameters")
        state_dtype = str(opt.state[params[0]]["exp_avg"].dtype)
        print(json.dumps({"event": "full_model_smoke_step", "step": step + 1,
                          "tokens_in_memory_only": SEQ, "loss": float(loss.detach()),
                          "elapsed_seconds": round(elapsed, 5),
                          "diagnostic_tokens_per_second": round(SEQ / elapsed, 2),
                          "optimizer_parameters_with_state": len(opt.state),
                          "optimizer_moment_dtype": state_dtype,
                          "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                          "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                          "durable_training_tokens": 0, "checkpoint_written": False}), flush=True)
    print(json.dumps({"event": "full_model_vram_smoke_passed", "trainable_parameters": total,
                      "steps": STEPS, "tokens_computed_in_memory_only": STEPS * SEQ,
                      "checkpoint_written": False, "durable_training_tokens": 0,
                      "optimizer_continuity_from_SAFE": False, "full_model_production_approved": False}), flush=True)


if __name__ == "__main__":
    main()
