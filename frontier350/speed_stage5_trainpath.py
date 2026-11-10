from __future__ import annotations

"""Stage-5 train-path speed sweep for Neral Arctus 350M.

Keeps the locked 349,825,152-parameter model and training math, but removes
benchmark-only host synchronizations from the microstep hot path and tests static
full-graph TorchInductor execution plus exact-token microbatch/grad-accum variants.
No checkpoint is written or modified.
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

from model350 import FRONTIER_350
from model350_fast import Neral350MFast

TARGET_TPS = 22_000.0
TARGET_TOKENS_PER_UPDATE = 16_384
PARAMETERS = 349_825_152


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


def exact_accumulation(mb: int, seq: int) -> int:
    per_micro = mb * seq
    if TARGET_TOKENS_PER_UPDATE % per_micro:
        raise ValueError(f"{mb=} {seq=} does not divide {TARGET_TOKENS_PER_UPDATE}")
    return TARGET_TOKENS_PER_UPDATE // per_micro


def configure_sdpa(force_flash: bool) -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(not force_flash)
    torch.backends.cuda.enable_mem_efficient_sdp(not force_flash)


def run_updates(model, raw_model, opt, pool, *, mb: int, seq: int, grad_accum: int,
                updates: int, rng: random.Random, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    last_grad = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(updates):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = pool.batch(mb, seq, rng, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("model returned no loss")
            (loss / grad_accum).backward()
            # Stay on GPU. Do not call torch.isfinite(), float(), item(), or bool()
            # inside the microstep hot path; those force GPU->CPU synchronization.
            loss_sum.add_(loss.detach().float())
        last_grad = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step()
    return loss_sum, last_grad


def run_case(pool: MMapPool, *, mb: int, seq: int, measured_updates: int,
             warmup_updates: int, fullgraph: bool, dynamic: bool | None,
             force_flash: bool) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    configure_sdpa(force_flash)
    grad_accum = exact_accumulation(mb, seq)
    tokens_per_update = mb * seq * grad_accum
    seed = 35100 + mb + (1000 if fullgraph else 0) + (2000 if force_flash else 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    device = torch.device("cuda")

    raw_model = Neral350MFast().to(device=device, dtype=torch.bfloat16)
    raw_model.set_gradient_checkpointing(False)
    if raw_model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("Arctus parameter contract changed")

    status = "ok"
    error = None
    model = raw_model
    try:
        model = torch.compile(
            raw_model,
            backend="inductor",
            mode="default",
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        opt = torch.optim.AdamW(
            raw_model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.1, fused=True,
        )
        torch.cuda.reset_peak_memory_stats()

        # Compile/warm up outside the measured region.
        warm_loss, warm_grad = run_updates(
            model, raw_model, opt, pool, mb=mb, seq=seq, grad_accum=grad_accum,
            updates=warmup_updates, rng=rng, device=device,
        )
        torch.cuda.synchronize()
        warm_loss_value = float(warm_loss / max(1, warmup_updates * grad_accum))
        warm_grad_value = float(warm_grad)
        if not math.isfinite(warm_loss_value) or not math.isfinite(warm_grad_value):
            raise RuntimeError("non-finite warmup loss/grad")

        # One synchronization around the entire measurement window, rather than
        # one or more forced host synchronizations per microstep/update.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        measured_loss, measured_grad = run_updates(
            model, raw_model, opt, pool, mb=mb, seq=seq, grad_accum=grad_accum,
            updates=measured_updates, rng=rng, device=device,
        )
        torch.cuda.synchronize()
        measured_s = time.perf_counter() - t0
        loss_value = float(measured_loss / (measured_updates * grad_accum))
        grad_value = float(measured_grad)
        if not math.isfinite(loss_value) or not math.isfinite(grad_value):
            raise RuntimeError("non-finite measured loss/grad")

        measured_tokens = tokens_per_update * measured_updates
        tps = measured_tokens / measured_s
        row = {
            "layout": "fused",
            "microbatch": mb,
            "seq": seq,
            "grad_accum": grad_accum,
            "tokens_per_update": tokens_per_update,
            "compile_mode": "default",
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "force_flash": force_flash,
            "status": status,
            "e2e_tokens_per_second": tps,
            "target_22k_reached": tps >= TARGET_TPS,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "warmup_loss": warm_loss_value,
            "measured_loss": loss_value,
            "last_grad_norm": grad_value,
            "measured_tokens": measured_tokens,
            "measured_seconds": measured_s,
            "error": None,
        }
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = type(exc).__name__
        torch.cuda.empty_cache()
        row = {
            "layout": "fused", "microbatch": mb, "seq": seq,
            "grad_accum": grad_accum, "tokens_per_update": tokens_per_update,
            "compile_mode": "default", "fullgraph": fullgraph, "dynamic": dynamic,
            "force_flash": force_flash, "status": status,
            "e2e_tokens_per_second": None, "target_22k_reached": False,
            "error": error,
        }
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}:{str(exc)[:500]}"
        row = {
            "layout": "fused", "microbatch": mb, "seq": seq,
            "grad_accum": grad_accum, "tokens_per_update": tokens_per_update,
            "compile_mode": "default", "fullgraph": fullgraph, "dynamic": dynamic,
            "force_flash": force_flash, "status": status,
            "e2e_tokens_per_second": None, "target_22k_reached": False,
            "error": error,
        }

    try:
        del opt
    except UnboundLocalError:
        pass
    del model, raw_model
    torch.cuda.empty_cache()
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    if args.updates <= 0 or args.warmup <= 0:
        raise SystemExit("updates and warmup must be positive")

    import triton

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pool = MMapPool(Path(args.train_data))

    # Baseline under Stage-5 timing plus static/fullgraph and exact-token MB sweep.
    specs = [
        (4, False, None, False),
        (2, True, False, False),
        (4, True, False, False),
        (8, True, False, False),
        (4, True, False, True),
    ]
    cases = []
    for mb, fullgraph, dynamic, force_flash in specs:
        row = run_case(
            pool, mb=mb, seq=args.seq, measured_updates=args.updates,
            warmup_updates=args.warmup, fullgraph=fullgraph, dynamic=dynamic,
            force_flash=force_flash,
        )
        cases.append(row)
        print(json.dumps(row), flush=True)

    valid = [r for r in cases if r.get("status") == "ok" and r.get("e2e_tokens_per_second")]
    best = max(valid, key=lambda r: r["e2e_tokens_per_second"]) if valid else None
    report = {
        "architecture": "frontier350-v1",
        "execution_search": "stage5-hotpath-static-fullgraph-microbatch",
        "parameters": PARAMETERS,
        "stage4_reference_tps": 19_239.4426024104,
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
        "note": "Speed probe only. Stage-5 timing removes benchmark-induced host syncs; compare variants within Stage-5 first. Production promotion still requires numerical/validation A/B on the real checkpoint.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
