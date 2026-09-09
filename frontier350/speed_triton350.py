from __future__ import annotations

"""Stage-4 WSL2/Triton speed probe for Neral Arctus 350M.

This benchmark is deliberately isolated from the production trainer. It uses the same
349,825,152-parameter model and real BPE shards, then compares eager vs TorchInductor
(`torch.compile`) execution inside Linux/WSL where Triton is supported.
No model checkpoint is written or modified.
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
        whole = torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)
        return whole[:, :-1], whole[:, 1:]


def configure_sdpa(force_flash: bool) -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(not force_flash)
    torch.backends.cuda.enable_mem_efficient_sdp(not force_flash)


def accumulation_for(mb: int, seq: int) -> int:
    return max(1, int(round(TARGET_TOKENS_PER_UPDATE / (mb * seq))))


def run_case(pool: MMapPool, *, layout: str, mb: int, seq: int, updates: int,
             warmup: int, compile_mode: str, force_flash: bool) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    configure_sdpa(force_flash)
    torch.manual_seed(35040 + mb)
    torch.cuda.manual_seed_all(35040 + mb)
    rng = random.Random(35040 + mb + (1000 if layout == "fused" else 0))
    device = torch.device("cuda")
    grad_accum = accumulation_for(mb, seq)
    tokens_per_update = mb * seq * grad_accum

    raw_model = (Neral350MFast() if layout == "fused" else Neral350M()).to(device=device, dtype=torch.bfloat16)
    raw_model.set_gradient_checkpointing(False)
    if raw_model.parameter_report()["total"] != 349_825_152:
        raise RuntimeError("Arctus parameter contract changed")
    model = raw_model
    compile_error = None
    if compile_mode != "none":
        try:
            model = torch.compile(raw_model, mode=compile_mode, fullgraph=False)
        except Exception as exc:
            compile_error = f"{type(exc).__name__}:{str(exc)[:300]}"

    if compile_error:
        return {
            "layout": layout, "microbatch": mb, "seq": seq,
            "grad_accum": grad_accum, "tokens_per_update": tokens_per_update,
            "compile_mode": compile_mode, "force_flash": force_flash,
            "status": "compile_failed", "e2e_tokens_per_second": None,
            "error": compile_error,
        }

    opt = torch.optim.AdamW(raw_model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.1, fused=True)
    torch.cuda.reset_peak_memory_stats()
    measured_tokens = 0
    measured_s = 0.0
    losses: list[float] = []
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
                # reduce-overhead uses CUDA Graphs. Explicitly mark every logical
                # invocation boundary because gradient accumulation calls the same
                # compiled model multiple times before the optimizer step.
                if compile_mode != "none" and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
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
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = type(exc).__name__
        torch.cuda.empty_cache()
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}:{str(exc)[:300]}"

    tps = measured_tokens / measured_s if measured_s else None
    row = {
        "layout": layout,
        "microbatch": mb,
        "seq": seq,
        "grad_accum": grad_accum,
        "tokens_per_update": tokens_per_update,
        "compile_mode": compile_mode,
        "force_flash": force_flash,
        "status": status,
        "e2e_tokens_per_second": tps,
        "target_15k_reached": bool(tps is not None and tps >= TARGET_TPS),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
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
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    if args.steps <= args.warmup:
        raise SystemExit("steps must exceed warmup")

    import triton
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pool = MMapPool(Path(args.train_data))

    specs = [
        ("unfused", 4, "none", False),
        ("unfused", 4, "default", False),
        ("unfused", 4, "reduce-overhead", False),
        ("unfused", 5, "reduce-overhead", False),
        ("fused", 4, "default", False),
        ("fused", 4, "reduce-overhead", False),
        ("fused", 5, "reduce-overhead", False),
    ]
    cases = []
    for layout, mb, mode, force_flash in specs:
        row = run_case(pool, layout=layout, mb=mb, seq=args.seq, updates=args.steps,
                       warmup=args.warmup, compile_mode=mode, force_flash=force_flash)
        cases.append(row)
        print(json.dumps(row), flush=True)

    valid = [r for r in cases if r.get("status") == "ok" and r.get("e2e_tokens_per_second")]
    best = max(valid, key=lambda r: r["e2e_tokens_per_second"]) if valid else None
    report = {
        "architecture": "frontier350-v1",
        "execution_search": "stage4-wsl2-triton-inductor",
        "parameters": 349_825_152,
        "target_tokens_per_second": TARGET_TPS,
        "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": getattr(triton, "__version__", "unknown"),
        "cases": cases,
        "best": best,
        "target_reached": bool(best and best["e2e_tokens_per_second"] >= TARGET_TPS),
        "promotion_allowed": False,
        "note": "Stage-4 speed probe only. No checkpoint changes; compiled path requires numerical/validation A/B before promotion.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
