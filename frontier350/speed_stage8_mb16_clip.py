from __future__ import annotations

"""Stage-8 checkpoint-native throughput sweep for Arctus 350M.

This is a report-only benchmark. It loads the immutable Arctus checkpoint and its AdamW
state, but never writes model or optimizer weights. The purpose is to measure two low-risk
training-path optimizations on the real 349,825,152-parameter layout:

1. explicit foreach gradient clipping;
2. MB16/GA2 versus the current MB8/GA4 shape, while keeping exactly 16,384 tokens/update.

Promotion still requires a separate numerical/validation gate. The benchmark must not run
concurrently with a real training job on the same RTX 4070.
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch

from model350 import FRONTIER_350, Neral350M
from train350_compiled import PersistentMMapPool

PARAMETERS = 349_825_152
TOKENS_PER_UPDATE = 16_384


def grad_accum(microbatch: int, seq: int) -> int:
    per_micro = microbatch * seq
    if per_micro <= 0 or TOKENS_PER_UPDATE % per_micro:
        raise ValueError("shape must preserve exactly 16,384 tokens/update")
    return TOKENS_PER_UPDATE // per_micro


def load_case(checkpoint: Path, device: torch.device, lr: float):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture differs from Frontier 350M")

    raw = Neral350M().to(device=device, dtype=torch.bfloat16)
    if raw.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("parameter contract changed")
    raw.load_state_dict(ck["model"])
    raw.set_gradient_checkpointing(False)

    opt = torch.optim.AdamW(
        raw.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.1, fused=True,
    )
    opt.load_state_dict(ck["optimizer"])
    for group in opt.param_groups:
        group["lr"] = lr

    model = torch.compile(
        raw, backend="inductor", mode="default", fullgraph=True, dynamic=False,
    )
    return raw, model, opt


def train_updates(
    raw: Neral350M,
    model,
    opt,
    pool: PersistentMMapPool,
    *,
    microbatch: int,
    seq: int,
    updates: int,
    rng: random.Random,
    device: torch.device,
    foreach_clip: bool | None,
):
    ga = grad_accum(microbatch, seq)
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    grad_norm = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(updates):
        opt.zero_grad(set_to_none=True)
        for _ in range(ga):
            x, y = pool.batch(microbatch, seq, device, rng)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("missing loss")
            (loss / ga).backward()
            loss_sum.add_(loss.detach().float())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw.parameters(), 1.0, foreach=foreach_clip,
        )
        opt.step()

    return loss_sum, grad_norm


def run_case(
    checkpoint: Path,
    pool: PersistentMMapPool,
    *,
    microbatch: int,
    seq: int,
    foreach_clip: bool | None,
    warmup: int,
    updates: int,
    seed: int,
    lr: float,
):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed + 1)

    raw = model = opt = None
    label = "auto" if foreach_clip is None else ("foreach" if foreach_clip else "scalar")
    ga = grad_accum(microbatch, seq)
    try:
        raw, model, opt = load_case(checkpoint, device, lr)
        torch.cuda.reset_peak_memory_stats()

        warm_loss, warm_grad = train_updates(
            raw, model, opt, pool,
            microbatch=microbatch, seq=seq, updates=warmup, rng=rng,
            device=device, foreach_clip=foreach_clip,
        )
        torch.cuda.synchronize()
        if not math.isfinite(float(warm_loss)) or not math.isfinite(float(warm_grad)):
            raise RuntimeError("non-finite warmup")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss, grad = train_updates(
            raw, model, opt, pool,
            microbatch=microbatch, seq=seq, updates=updates, rng=rng,
            device=device, foreach_clip=foreach_clip,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        mean_loss = float(loss / (updates * ga))
        grad_value = float(grad)
        if not math.isfinite(mean_loss) or not math.isfinite(grad_value):
            raise RuntimeError("non-finite timed result")

        return {
            "status": "ok",
            "microbatch": microbatch,
            "grad_accum": ga,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "clip_mode": label,
            "compile_mode": "default",
            "fullgraph": True,
            "updates": updates,
            "seconds": elapsed,
            "tokens_per_second": TOKENS_PER_UPDATE * updates / elapsed,
            "mean_loss": mean_loss,
            "last_grad_norm": grad_value,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "status": "oom",
            "microbatch": microbatch,
            "grad_accum": ga,
            "clip_mode": label,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "microbatch": microbatch,
            "grad_accum": ga,
            "clip_mode": label,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    finally:
        del model, opt, raw
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--updates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=38001)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 required")
    if args.seq != 512:
        raise ValueError("Stage-8 contract currently fixes seq=512")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    pool = PersistentMMapPool(Path(args.train_data), FRONTIER_350.vocab_size)

    specs = [
        (8, None),
        (8, True),
        (16, True),
    ]
    rows = []
    for microbatch, foreach_clip in specs:
        result = run_case(
            checkpoint, pool,
            microbatch=microbatch,
            seq=args.seq,
            foreach_clip=foreach_clip,
            warmup=args.warmup,
            updates=args.updates,
            seed=args.seed,
            lr=args.lr,
        )
        rows.append(result)
        print(json.dumps(result), flush=True)

    ok = [row for row in rows if row["status"] == "ok"]
    baseline = next(
        (row for row in ok if row["microbatch"] == 8 and row["clip_mode"] == "auto"),
        None,
    )
    if baseline:
        for row in ok:
            row["vs_mb8_auto_pct"] = (
                row["tokens_per_second"] / baseline["tokens_per_second"] - 1.0
            ) * 100.0

    report = {
        "schema": 1,
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "checkpoint_mutated": False,
        "promotion_allowed": False,
        "purpose": "measure foreach clipping and MB16/GA2 on the real checkpoint-native training path",
        "cases": rows,
        "best": max(ok, key=lambda row: row["tokens_per_second"]) if ok else None,
        "guardrail": "Report only. Do not promote without numerical and fixed-validation equivalence checks.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
