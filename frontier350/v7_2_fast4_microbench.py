from __future__ import annotations

"""Benchmark-only Fast4 experiment for the RTX 4070.

Compares the current proven turbo50 execution shape with a CUDA-Graph candidate that
prewarms the process-global RoPE cache before torch.compile.  It never loads or writes
training checkpoints, corpora, optimizer state, or continuity journals.
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
BASELINE_MODE = "max-autotune-no-cudagraphs"
CANDIDATE_MODE = "max-autotune"


def prewarm_rope() -> None:
    _rope_table(
        FRONTIER_350.max_seq_len,
        FRONTIER_350.head_dim,
        FRONTIER_350.rope_theta,
        torch.device("cuda"),
        torch.bfloat16,
    )


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(72001)
    x = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y[:, ::41] = -100
    return x, y


def weighted_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ry, my = y[:REPLAY_ROWS], y[REPLAY_ROWS:]
    rloss = F.cross_entropy(
        logits[:REPLAY_ROWS].reshape(-1, logits.size(-1)), ry.reshape(-1),
        ignore_index=-100, reduction="sum")
    mloss = F.cross_entropy(
        logits[REPLAY_ROWS:].reshape(-1, logits.size(-1)), my.reshape(-1),
        ignore_index=-100, reduction="sum")
    rv = ry.ne(-100).sum(dtype=torch.float32)
    mv = my.ne(-100).sum(dtype=torch.float32)
    return (rloss * REPLAY_BOOST + mloss) / (mv + REPLAY_BOOST * rv)


def trainable(model: Neral350M) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def summarize(rates: list[float], losses: list[float]) -> dict:
    return {
        "median_tps": statistics.median(rates),
        "mean_tps": statistics.fmean(rates),
        "loss": losses[-1],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1073741824,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1073741824,
    }


def bench_turbo50(model: Neral350M, x: torch.Tensor, y: torch.Tensor,
                  warmup: int, measured: int, tokens: int) -> dict:
    run = torch.compile(model, fullgraph=True, dynamic=False, mode=BASELINE_MODE)
    params = trainable(model)
    rates: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(warmup + measured):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = run(x, None)
                loss = weighted_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            if i >= warmup:
                rates.append(tokens / elapsed)
                losses.append(float(loss.detach()))
            del logits, loss
        out = summarize(rates, losses)
        out["status"] = "ok"
        return out
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:800]}"}
    finally:
        del run
        gc.collect(); torch.cuda.empty_cache()


def bench_fast4(model: Neral350M, x: torch.Tensor, y: torch.Tensor,
                warmup: int, measured: int, tokens: int) -> dict:
    # Critical difference from failed turbo60: create the cached RoPE tensors before
    # graph capture, then compile the complete weighted objective rather than returning
    # the enormous logits tensor across the compiled-call boundary.
    prewarm_rope()

    def objective(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        logits, _ = model(xb, None)
        return weighted_loss(logits, yb)

    run = torch.compile(objective, fullgraph=True, dynamic=False, mode=CANDIDATE_MODE)
    params = trainable(model)
    rates: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(warmup + measured):
            model.zero_grad(set_to_none=True)
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = run(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            if i >= warmup:
                rates.append(tokens / elapsed)
                losses.append(float(loss.detach()))
            del loss
        out = summarize(rates, losses)
        out["status"] = "ok"
        return out
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}:{str(exc)[:800]}"}
    finally:
        del run
        gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measured", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.08)
    ap.add_argument("--max-loss-diff", type=float, default=2e-4)
    ap.add_argument("--max-peak-reserved-gib", type=float, default=9.0)
    a = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"expected RTX 4070, got {gpu}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(72000); torch.cuda.manual_seed_all(72000)
    model = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    configure_trainable(model, 4)
    prewarm_rope()
    x, y = make_inputs()
    tokens = int(y.ne(-100).sum().item())

    baseline = bench_turbo50(model, x, y, a.warmup, a.measured, tokens)
    fast4 = bench_fast4(model, x, y, a.warmup, a.measured, tokens)

    ratio = None
    loss_diff = None
    failures: list[str] = []
    if baseline.get("status") != "ok":
        failures.append("baseline_failed")
    if fast4.get("status") != "ok":
        failures.append("fast4_failed")
    if not failures:
        ratio = fast4["median_tps"] / baseline["median_tps"]
        loss_diff = abs(fast4["loss"] - baseline["loss"])
        if ratio < a.min_ratio:
            failures.append("speed_ratio")
        if loss_diff > a.max_loss_diff:
            failures.append("loss_fidelity")
        if fast4["peak_reserved_gib"] > a.max_peak_reserved_gib:
            failures.append("peak_reserved")

    report = {
        "schema": 1,
        "gpu": gpu,
        "torch": torch.__version__,
        "batch": BATCH,
        "seq": SEQ,
        "valid_tokens_per_step": tokens,
        "baseline": {"profile": "turbo50", "compile_mode": BASELINE_MODE, **baseline},
        "fast4": {"profile": "fast4", "compile_mode": CANDIDATE_MODE,
                  "rope_cache_prewarmed": True, "cudagraph_step_marker": True, **fast4},
        "fast4_ratio": ratio,
        "loss_abs_diff": loss_diff,
        "limits": {"min_ratio": a.min_ratio, "max_loss_diff": a.max_loss_diff,
                   "max_peak_reserved_gib": a.max_peak_reserved_gib},
        "promote_fast4": not failures,
        "failures": failures,
        "note": "benchmark only; no checkpoint/data/continuity state touched",
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
