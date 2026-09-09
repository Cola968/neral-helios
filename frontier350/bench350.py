from __future__ import annotations

"""Bounded RTX hardware/architecture pilot for Neral 350M Frontier.

This script never touches release benchmarks or long-run corpora. It measures the exact
frontier architecture on synthetic byte-token batches so the 10B campaign is not started
before VRAM, throughput, numerics and resume mechanics are proven on the target GPU.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from model350 import Neral350M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")

    torch.manual_seed(350)
    torch.cuda.manual_seed_all(350)
    device = torch.device("cuda")
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(True)
    report = model.parameter_report()
    if not 330_000_000 <= report["total"] <= 370_000_000:
        raise SystemExit(f"parameter contract failed: {report}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    torch.cuda.reset_peak_memory_stats()
    total_tokens = 0
    losses = []
    measured_s = 0.0

    for step in range(args.steps):
        x = torch.randint(0, model.cfg.vocab_size, (args.microbatch, args.seq), device=device)
        y = torch.roll(x, shifts=-1, dims=1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(grad)):
            raise SystemExit(f"non-finite grad norm at step {step}: {grad}")
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            measured_s += elapsed
            total_tokens += args.microbatch * args.seq
            losses.append(float(loss.detach()))
        print(json.dumps({"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(grad), "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9}), flush=True)

    tps = total_tokens / measured_s if measured_s else 0.0
    result = {
        "architecture": "Neral 350M Frontier v1",
        "parameters": report,
        "seq": args.seq,
        "microbatch": args.microbatch,
        "measured_steps": max(0, args.steps-args.warmup),
        "tokens_per_second": tps,
        "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "finite": all(math.isfinite(v) for v in losses),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
