from __future__ import annotations

"""Isolated RTX 4070 benchmark for V6 frozen-head chunked CE.

This deliberately uses random model weights and synthetic token IDs. It benchmarks the
actual Neral350M topology with only the last four blocks trainable, but it never opens or
modifies a training checkpoint, corpus, optimizer state, or continuity journal.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model350 import FRONTIER_350, Neral350M
from protection_v9 import configure_trainable
from v6_chunked_frozen_head_ce import frozen_head_chunked_ce_sum


def backbone(model: Neral350M, idx: torch.Tensor) -> torch.Tensor:
    x = model.tok_emb(idx)
    for block in model.blocks:
        x = block(x)
    return model.norm(x)


def one_step(model: Neral350M, idx: torch.Tensor, targets: torch.Tensor, mode: str, chunk: int) -> float:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(model, idx)
        if mode == "standard":
            logits = F.linear(hidden, model.lm_head.weight)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), ignore_index=-100)
        elif mode == "chunked":
            valid = targets.ne(-100).sum().clamp_min(1)
            loss = frozen_head_chunked_ce_sum(hidden, model.lm_head.weight, targets,
                                              chunk_size=chunk, ignore_index=-100) / valid
        else:
            raise ValueError(mode)
    loss.backward()
    return float(loss.detach())


def bench_mode(model: Neral350M, batch: int, seq: int, mode: str, chunk: int,
               warmup: int, measured: int) -> dict:
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(61000 + batch + (0 if mode == "standard" else 1000))
    idx = torch.randint(0, FRONTIER_350.vocab_size, (batch, seq), device="cuda")
    targets = torch.randint(0, FRONTIER_350.vocab_size, (batch, seq), device="cuda")
    targets[:, ::37] = -100
    times: list[float] = []
    losses: list[float] = []
    try:
        for i in range(warmup + measured):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            loss = one_step(model, idx, targets, mode, chunk)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt); losses.append(loss)
        valid_tokens = int(targets.ne(-100).sum().item())
        rates = [valid_tokens / x for x in times]
        return {
            "ok": True, "batch": batch, "seq": seq, "mode": mode,
            "median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
            "median_seconds": statistics.median(times), "loss_last": losses[-1],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1073741824,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1073741824,
        }
    except torch.cuda.OutOfMemoryError as exc:
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()
        return {"ok": False, "batch": batch, "seq": seq, "mode": mode, "oom": True,
                "error": str(exc)[:500]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batches", default="24,16,8,4")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--measured", type=int, default=3)
    a = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"expected RTX 4070, got {gpu}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(60999); torch.cuda.manual_seed_all(60999)
    model = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    named = configure_trainable(model, 4)
    if any(p.requires_grad for p in model.lm_head.parameters()):
        raise RuntimeError("LM head must remain frozen for V6 chunked CE")
    trainable = sum(p.numel() for _, p in named)

    rows = []
    selected = None
    for batch in [int(x) for x in a.batches.split(",") if x.strip()]:
        standard = bench_mode(model, batch, a.seq, "standard", a.chunk, a.warmup, a.measured)
        chunked = bench_mode(model, batch, a.seq, "chunked", a.chunk, a.warmup, a.measured)
        row = {"batch": batch, "standard": standard, "chunked": chunked}
        if standard.get("ok") and chunked.get("ok"):
            row["chunked_vs_standard_tps_ratio"] = chunked["median_tps"] / standard["median_tps"]
            row["reserved_gib_saved"] = standard["peak_reserved_gib"] - chunked["peak_reserved_gib"]
            if selected is None:
                selected = row
        rows.append(row)
        if selected is not None:
            break

    report = {
        "schema": 1, "gpu": gpu, "torch": torch.__version__, "seq": a.seq,
        "trainable_parameters": trainable, "total_parameters": sum(p.numel() for p in model.parameters()),
        "chunk_size": a.chunk, "rows": rows, "selected": selected,
        "note": "synthetic full-model benchmark; no checkpoint/data/optimizer/continuity state loaded or modified",
    }
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if selected is None:
        raise SystemExit("no common batch fit for standard and chunked benchmark")


if __name__ == "__main__":
    main()
