from __future__ import annotations

"""Stage-2 RTX 4070 speed search for the 15k tok/s Frontier target.

This benchmark never writes model checkpoints. It compares the locked unfused execution
layout with a mathematically equivalent fused QKV + fused SwiGLU layout and optional
TorchInductor compilation. It measures the real training cadence: several microbatches
are accumulated before one grad-clip/AdamW update, matching the 16,384-token/update
Frontier burn-in contract as closely as each microbatch permits.
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

from model350 import FRONTIER_350, Neral350M
from model350_fast import Neral350MFast

TARGET_TPS = 15_000.0
TARGET_TOKENS_PER_UPDATE = 16_384


class MMapPool:
    def __init__(self, root: Path):
        paths = sorted(root.glob("*.bin"))
        if not paths:
            raise FileNotFoundError(f"no *.bin shards in {root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        if any(len(a) < 513 for a in self.arrays):
            raise ValueError("speed shard too short")

    def batch(self, batch: int, seq: int, rng: random.Random, device: torch.device):
        rows = np.empty((batch, seq + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(self.arrays)
            start = rng.randrange(0, len(a) - seq - 1)
            rows[i] = a[start:start + seq + 1]
        if rows.max(initial=0) >= FRONTIER_350.vocab_size:
            raise ValueError("token id outside locked vocabulary")
        whole = torch.from_numpy(rows).to(device, non_blocking=True)
        return whole[:, :-1], whole[:, 1:]


def configure_sdpa(force_flash: bool) -> None:
    if not hasattr(torch.backends, "cuda"):
        return
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(not force_flash)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(not force_flash)


def accumulation_for(mb: int, seq: int) -> int:
    return max(1, int(round(TARGET_TOKENS_PER_UPDATE / (mb * seq))))


def run_case(pool: MMapPool, *, layout: str, mb: int, seq: int, updates: int, warmup: int,
             compile_mode: str, force_flash: bool) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    configure_sdpa(force_flash)
    torch.manual_seed(15000 + mb)
    torch.cuda.manual_seed_all(15000 + mb)
    rng = random.Random(15000 + mb + (1000 if layout == "fused" else 0))
    device = torch.device("cuda")
    grad_accum = accumulation_for(mb, seq)
    tokens_per_update = mb * seq * grad_accum

    raw_model = (Neral350MFast() if layout == "fused" else Neral350M()).to(device=device, dtype=torch.bfloat16)
    raw_model.set_gradient_checkpointing(False)
    report = raw_model.parameter_report()
    if report["total"] != 349_825_152:
        raise RuntimeError(f"parameter contract changed: {report}")
    model = raw_model
    if compile_mode != "none":
        try:
            model = torch.compile(raw_model, mode=compile_mode, fullgraph=False)
        except Exception as exc:
            return {
                "layout": layout, "microbatch": mb, "seq": seq,
                "grad_accum": grad_accum, "tokens_per_update": tokens_per_update,
                "compile_mode": compile_mode, "force_flash": force_flash,
                "status": "compile_failed", "error": f"{type(exc).__name__}:{str(exc)[:220]}",
            }

    opt = torch.optim.AdamW(raw_model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.1, fused=True)
    torch.cuda.reset_peak_memory_stats()
    measured_tokens = 0
    measured_s = 0.0
    losses: list[float] = []
    grads: list[float] = []
    status = "ok"
    error = None
    try:
        for update in range(updates):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(grad_accum):
                x, y = pool.batch(mb, seq, rng, device)
                _, loss = model(x, y)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                (loss / grad_accum).backward()
                running_loss += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            if not math.isfinite(float(grad)):
                raise RuntimeError("non-finite grad norm")
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if update >= warmup:
                measured_tokens += tokens_per_update
                measured_s += dt
                losses.append(running_loss / grad_accum)
                grads.append(float(grad))
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = type(exc).__name__
        torch.cuda.empty_cache()
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}:{str(exc)[:240]}"

    e2e_tps = measured_tokens / measured_s if measured_s else None
    row = {
        "layout": layout,
        "microbatch": mb,
        "seq": seq,
        "grad_accum": grad_accum,
        "tokens_per_update": tokens_per_update,
        "compile_mode": compile_mode,
        "force_flash": force_flash,
        "status": status,
        "e2e_tokens_per_second": e2e_tps,
        "target_15k_reached": bool(e2e_tps is not None and e2e_tps >= TARGET_TPS),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "grad_last": grads[-1] if grads else None,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_s,
        "error": error,
    }
    del opt, model, raw_model
    torch.cuda.empty_cache()
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10, help="optimizer updates per case")
    ap.add_argument("--warmup", type=int, default=4, help="warmup optimizer updates per case")
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    if args.steps <= args.warmup:
        raise SystemExit("steps must exceed warmup")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pool = MMapPool(Path(args.train_data))

    specs = [
        ("unfused", 4, "none", False),
        ("fused", 4, "none", False),
        ("fused", 6, "none", False),
        ("fused", 8, "none", False),
        ("fused", 10, "none", False),
        ("fused", 6, "reduce-overhead", False),
        ("fused", 8, "reduce-overhead", False),
        ("fused", 10, "reduce-overhead", False),
        ("fused", 6, "reduce-overhead", True),
        ("fused", 8, "reduce-overhead", True),
    ]
    cases = []
    for layout, mb, mode, force_flash in specs:
        row = run_case(pool, layout=layout, mb=mb, seq=args.seq, updates=args.steps,
                       warmup=args.warmup, compile_mode=mode, force_flash=force_flash)
        cases.append(row)
        print(json.dumps(row), flush=True)

    valid = [r for r in cases if r.get("status") == "ok" and r.get("e2e_tokens_per_second")]
    best = max(valid, key=lambda r: r["e2e_tokens_per_second"]) if valid else None
    baseline = next((r for r in valid if r["layout"] == "unfused" and r["microbatch"] == 4), None)
    speedup = None
    if best and baseline and baseline["e2e_tokens_per_second"]:
        speedup = best["e2e_tokens_per_second"] / baseline["e2e_tokens_per_second"]
    report = {
        "architecture": "frontier350-v1",
        "execution_search": "fused-qkv-gateup-rmsnorm-v1",
        "parameters": 349_825_152,
        "target_tokens_per_second": TARGET_TPS,
        "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cases": cases,
        "best": best,
        "best_vs_unfused_mb4_speedup": speedup,
        "target_reached": bool(best and best["e2e_tokens_per_second"] >= TARGET_TPS),
        "promotion_allowed": False,
        "note": "Real grad-accum cadence on real BPE shards. Speed result only; fused layout still requires checkpoint conversion and numerical validation before promotion.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
