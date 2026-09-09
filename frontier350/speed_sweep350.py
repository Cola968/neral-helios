from __future__ import annotations

"""Isolated Neral 350M RTX speed sweep.

This benchmark is deliberately separate from the release trainer. It never modifies a
training checkpoint and can be queued behind a live self-hosted training run. It measures
full forward/backward/fused-AdamW steps for the locked Frontier architecture on real local
BPE shards, compares cached mmap loading and optional torch.compile, and records OOMs
instead of failing the whole sweep.
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model350 import Neral350M, FRONTIER_350


class MMapPool:
    def __init__(self, root: Path):
        paths = sorted(root.glob("*.bin"))
        if not paths:
            raise FileNotFoundError(f"no *.bin shards in {root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        if any(len(a) < 1025 for a in self.arrays):
            raise ValueError("speed-sweep shard too short")

    def batch(self, batch: int, seq: int, rng: random.Random, device: torch.device):
        rows = np.empty((batch, seq + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(self.arrays)
            start = rng.randrange(0, len(a) - seq - 1)
            rows[i] = a[start:start + seq + 1]
        if rows.max(initial=0) >= FRONTIER_350.vocab_size:
            raise ValueError("token id outside locked vocabulary")
        x = torch.from_numpy(rows[:, :-1].copy()).to(device, non_blocking=True)
        y = torch.from_numpy(rows[:, 1:].copy()).to(device, non_blocking=True)
        return x, y


def one_case(pool: MMapPool, seq: int, mb: int, steps: int, warmup: int, compile_model: bool) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(3502026)
    torch.cuda.manual_seed_all(3502026)
    rng = random.Random(3502026 + mb + int(compile_model) * 1000)
    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(False)
    if model.parameter_report()["total"] != 349_825_152:
        raise RuntimeError(f"locked parameter contract changed: {model.parameter_report()}")
    if compile_model:
        if not hasattr(torch, "compile"):
            return {"microbatch": mb, "seq": seq, "compile": True, "status": "unsupported"}
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
    torch.cuda.reset_peak_memory_stats()
    measured_tokens = 0
    measured_s = 0.0
    losses: list[float] = []
    grads: list[float] = []
    status = "ok"
    error = None
    try:
        for step in range(steps):
            # Data loading is outside the timed GPU section, but uses persistent mmaps so the
            # benchmark also validates the production loader strategy without reopening files.
            x, y = pool.batch(mb, seq, rng, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad)):
                raise RuntimeError("non-finite grad norm")
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= warmup:
                measured_s += dt
                measured_tokens += mb * seq
                losses.append(float(loss.detach()))
                grads.append(float(grad))
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = type(exc).__name__
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        status = "failed"
        error = f"{type(exc).__name__}:{str(exc)[:180]}"
    result = {
        "microbatch": mb,
        "seq": seq,
        "compile": compile_model,
        "status": status,
        "tokens_per_second": measured_tokens / measured_s if measured_s > 0 else None,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "grad_last": grads[-1] if grads else None,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_s,
        "error": error,
    }
    del opt, model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatches", default="4,5,6,8")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--with-compile", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    if args.steps <= args.warmup or args.seq <= 0:
        raise SystemExit("invalid sweep dimensions")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pool = MMapPool(Path(args.train_data))
    mbs = [int(x) for x in args.microbatches.split(",") if x.strip()]
    cases = []
    for compile_model in ([False, True] if args.with_compile else [False]):
        for mb in mbs:
            row = one_case(pool, args.seq, mb, args.steps, args.warmup, compile_model)
            cases.append(row)
            print(json.dumps(row), flush=True)
    valid = [r for r in cases if r["status"] == "ok" and r["tokens_per_second"]]
    best = max(valid, key=lambda r: r["tokens_per_second"]) if valid else None
    report = {
        "architecture": "frontier350-v1",
        "parameters": 349_825_152,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "data_loader": "persistent-numpy-memmap-v1",
        "cases": cases,
        "best": best,
        "note": "Synthetic architecture changes are intentionally excluded; only checkpoint-compatible execution settings are tested here.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
