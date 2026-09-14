from __future__ import annotations

"""Synthetic RTX 4070 microbenchmark for V6 Fast2.

Compares the current V6 execution shape (compiled model + external weighted CE) against
Fast2 (compiled model+weighted CE objective) on the exact fused 48x512 physical batch.
The benchmark deliberately disables CUDA Graphs because PyTorch 2.8 graph-tree replay is
not compatible with model350.py's process-global RoPE tensor cache. TorchInductor and
max-autotune remain enabled. No checkpoints, corpora, optimizer state, continuity journal,
or training assets are read or modified.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model350 import FRONTIER_350, Neral350M, _rope_table
from protection_v9 import configure_trainable

BATCH = 48
SEQ = 512
REPLAY_ROWS = 12
REPLAY_BOOST = 1.5
COMPILE_MODE = "max-autotune-no-cudagraphs"


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(62001)
    x = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y[:, ::41] = -100
    return x, y


def weighted_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ry = y[:REPLAY_ROWS]
    my = y[REPLAY_ROWS:]
    rloss = F.cross_entropy(logits[:REPLAY_ROWS].reshape(-1, logits.size(-1)), ry.reshape(-1), ignore_index=-100, reduction="sum")
    mloss = F.cross_entropy(logits[REPLAY_ROWS:].reshape(-1, logits.size(-1)), my.reshape(-1), ignore_index=-100, reduction="sum")
    rv = ry.ne(-100).sum(dtype=torch.float32)
    mv = my.ne(-100).sum(dtype=torch.float32)
    return (rloss * REPLAY_BOOST + mloss) / (mv + REPLAY_BOOST * rv)


def prewarm_rope_cache() -> None:
    # Keep cache creation outside torch.compile as a second layer of protection. The
    # selected compile mode already disables CUDA Graphs, so cached RoPE storage cannot
    # become graph-owned or be overwritten by graph replay.
    _rope_table(
        FRONTIER_350.max_seq_len,
        FRONTIER_350.head_dim,
        FRONTIER_350.rope_theta,
        torch.device("cuda"),
        torch.bfloat16,
    )


def trainable_params(model: Neral350M) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def bench_current(model: Neral350M, x: torch.Tensor, y: torch.Tensor, warmup: int, measured: int) -> dict:
    run = torch.compile(model, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    params = trainable_params(model)
    rates, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + measured):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = run(x, None)
            loss = weighted_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i >= warmup:
            rates.append(int(y.ne(-100).sum()) / dt)
            losses.append(float(loss.detach()))
        del logits, loss
    out = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates), "loss": losses[-1],
           "peak_allocated_gib": torch.cuda.max_memory_allocated()/1073741824,
           "peak_reserved_gib": torch.cuda.max_memory_reserved()/1073741824}
    del run; gc.collect(); torch.cuda.empty_cache()
    return out


def bench_fast2(model: Neral350M, x: torch.Tensor, y: torch.Tensor, warmup: int, measured: int) -> dict:
    def objective(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        logits, _ = model(xb, None)
        return weighted_loss(logits, yb)

    run = torch.compile(objective, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    params = trainable_params(model)
    rates, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + measured):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = run(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i >= warmup:
            rates.append(int(y.ne(-100).sum()) / dt)
            losses.append(float(loss.detach()))
        del loss
    out = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates), "loss": losses[-1],
           "peak_allocated_gib": torch.cuda.max_memory_allocated()/1073741824,
           "peak_reserved_gib": torch.cuda.max_memory_reserved()/1073741824}
    del run; gc.collect(); torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--measured", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=1.06)
    a = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"expected RTX 4070, got {gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(62000); torch.cuda.manual_seed_all(62000)
    model = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    configure_trainable(model, 4)
    prewarm_rope_cache()
    x, y = make_inputs()
    current = bench_current(model, x, y, a.warmup, a.measured)
    fast2 = bench_fast2(model, x, y, a.warmup, a.measured)
    ratio = fast2["median_tps"] / current["median_tps"]
    loss_diff = abs(fast2["loss"] - current["loss"])
    use_fast2 = ratio >= a.min_ratio and loss_diff <= 2e-4
    report = {"schema":3,"gpu":gpu,"torch":torch.__version__,"batch":BATCH,"seq":SEQ,
              "compile_mode":COMPILE_MODE,"cudagraphs":False,
              "current":current,"fast2":fast2,"fast2_ratio":ratio,"loss_abs_diff":loss_diff,
              "min_ratio":a.min_ratio,"use_fast2":use_fast2,"rope_cache_prewarmed":True,
              "note":"synthetic execution benchmark; TorchInductor max-autotune without CUDA Graphs; no checkpoint/data/continuity state touched"}
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
