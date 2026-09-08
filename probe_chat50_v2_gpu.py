from __future__ import annotations

"""Measure the fastest safe Neral-50M Chat V2 batch size for each sequence bucket.

This is a real BF16 training-path probe: forward CE, backward, gradient clipping and fused AdamW.
It does not train a useful model and never consumes private checkpoints or corpora. The resulting
JSON profile is hardware evidence only and can be fed to Chat V2 so short and long sequences do
not share a naive equal-token batch rule.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import FIFTY
from fastpath50 import upgrade_runtime_inplace
from model50 import Neral50M

DEFAULT_WIDTHS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


def candidates(width: int, max_padded_tokens: int, max_batch: int) -> list[int]:
    cap = max(1, min(max_batch, max_padded_tokens // width))
    rows = []
    value = 1
    while value <= cap:
        rows.append(value); value *= 2
    if rows[-1] != cap:
        rows.append(cap)
    return sorted(set(rows))


def training_step(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return loss, grad


def measure(model, opt, width: int, rows: int, warmup: int, steps: int, device: torch.device) -> dict:
    x = torch.randint(4, FIFTY.vocab_size, (rows, width), device=device, dtype=torch.long)
    y = torch.randint(4, FIFTY.vocab_size, (rows, width), device=device, dtype=torch.long)
    for _ in range(warmup):
        training_step(model, opt, x, y)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = grad = None
    for _ in range(steps):
        loss, grad = training_step(model, opt, x, y)
    end.record(); end.synchronize()
    seconds = start.elapsed_time(end) / 1000.0
    tokens = rows * width * steps
    result = {
        "width": width, "batch_rows": rows, "padded_tokens_per_step": rows * width,
        "steps": steps, "seconds": seconds, "tokens_per_second": tokens / max(seconds, 1e-9),
        "tokens_per_hour": tokens / max(seconds, 1e-9) * 3600.0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "loss": float(loss.detach()), "grad_norm": float(grad), "oom": False,
    }
    del x, y, loss, grad
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chat50_v2_gpu_profile")
    ap.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS))
    ap.add_argument("--max-padded-tokens", type=int, default=32768)
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--max-reserved-fraction", type=float, default=.90)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU profile requires BF16 CUDA")
    if min(args.widths) <= 0 or max(args.widths) > 2048:
        raise ValueError("probe widths must be within Chat V2 E1 1..2048")
    if min(args.max_padded_tokens, args.max_batch, args.steps) <= 0 or args.warmup < 0:
        raise ValueError("invalid probe budget")
    if not .5 <= args.max_reserved_fraction < 1.0:
        raise ValueError("max-reserved-fraction must be in [.5,1)")

    device = torch.device("cuda")
    torch.manual_seed(968_570); torch.cuda.manual_seed_all(968_570)
    torch.set_float32_matmul_precision("high")
    model = Neral50M(FIFTY).to(device)
    upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95), weight_decay=.025, fused=True)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    rows_out = []
    profile = {}

    # Allocate optimizer state once at the smallest shape so later measurements don't include it.
    sx = torch.randint(4, FIFTY.vocab_size, (1, min(args.widths)), device=device)
    sy = torch.randint(4, FIFTY.vocab_size, (1, min(args.widths)), device=device)
    training_step(model, opt, sx, sy); torch.cuda.synchronize(); del sx, sy

    for width in args.widths:
        successful = []
        for batch_rows in candidates(width, args.max_padded_tokens, args.max_batch):
            gc.collect(); torch.cuda.empty_cache()
            try:
                result = measure(model, opt, width, batch_rows, args.warmup, args.steps, device)
                result["reserved_fraction"] = result["peak_reserved_bytes"] / total_vram
                result["headroom_eligible"] = result["reserved_fraction"] <= args.max_reserved_fraction
                rows_out.append(result)
                if result["headroom_eligible"]:
                    successful.append(result)
                print(json.dumps(result), flush=True)
            except torch.OutOfMemoryError:
                result = {"width": width, "batch_rows": batch_rows, "padded_tokens_per_step": width * batch_rows, "oom": True}
                rows_out.append(result); print(json.dumps(result), flush=True)
                opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                break
        if not successful:
            raise RuntimeError(f"no headroom-safe batch for width={width}")
        # Throughput wins; if two are within 1%, prefer the smaller VRAM footprint.
        fastest = max(r["tokens_per_second"] for r in successful)
        near = [r for r in successful if r["tokens_per_second"] >= fastest * .99]
        chosen = min(near, key=lambda r: (r["peak_reserved_bytes"], -r["tokens_per_second"]))
        profile[str(width)] = int(chosen["batch_rows"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {
        "version": 1,
        "purpose": "Neral-50M Chat V2 per-bucket BF16 training autotune",
        "gpu": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()), "context_ceiling_probed": max(args.widths),
        "max_reserved_fraction": args.max_reserved_fraction, "total_vram_bytes": total_vram,
        "profile_rows_by_width": profile, "measurements": rows_out,
        "selection": "maximum measured tok/s among <= VRAM fraction; within 1% choose smaller reserved memory",
        "quality_claim": False,
    }
    (out / "profile.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    (out / "batch_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf8")
    print(json.dumps({"selected_profile": profile, "gpu": report["gpu"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
