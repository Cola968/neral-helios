from __future__ import annotations

"""Benchmark-only compile-mode sweep for the Fast4 execution shape on RTX 4070.

All cases use the exact same Neral 350M architecture, deterministic weights/inputs,
weighted replay/main objective, RoPE prewarm, CUDA-Graph step marker and trainable-last-4
contract.  Only torch.compile mode changes.  No checkpoint, corpus, optimizer state or
continuity journal is read or written.
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
SEED_MODEL = 72400
SEED_INPUT = 72401
MODES = ("max-autotune", "reduce-overhead", "default")


def prewarm_rope() -> None:
    _rope_table(
        FRONTIER_350.max_seq_len,
        FRONTIER_350.head_dim,
        FRONTIER_350.rope_theta,
        torch.device("cuda"),
        torch.bfloat16,
    )


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED_INPUT)
    x = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y[:, ::41] = -100
    return x, y


def weighted_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ry, my = y[:REPLAY_ROWS], y[REPLAY_ROWS:]
    rloss = F.cross_entropy(
        logits[:REPLAY_ROWS].reshape(-1, logits.size(-1)),
        ry.reshape(-1), ignore_index=-100, reduction="sum")
    mloss = F.cross_entropy(
        logits[REPLAY_ROWS:].reshape(-1, logits.size(-1)),
        my.reshape(-1), ignore_index=-100, reduction="sum")
    rv = ry.ne(-100).sum(dtype=torch.float32)
    mv = my.ne(-100).sum(dtype=torch.float32)
    return (rloss * REPLAY_BOOST + mloss) / (mv + REPLAY_BOOST * rv)


def run_case(mode: str, x: torch.Tensor, y: torch.Tensor,
             warmup: int, measured: int, valid_tokens: int) -> dict:
    torch.manual_seed(SEED_MODEL)
    torch.cuda.manual_seed_all(SEED_MODEL)
    model = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    named = configure_trainable(model, 4)
    params = [p for _, p in named]
    if sum(p.numel() for p in model.parameters()) != 349_825_152:
        raise RuntimeError("parameter contract changed")
    prewarm_rope()

    def objective(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        logits, _ = model(xb, None)
        return weighted_loss(logits, yb)

    compiled = torch.compile(objective, fullgraph=True, dynamic=False, mode=mode)
    rates: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    status = "ok"
    error = None
    try:
        for i in range(warmup + measured):
            model.zero_grad(set_to_none=True)
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compiled(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                rates.append(valid_tokens / elapsed)
                losses.append(float(loss.detach()))
            del loss
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}:{str(exc)[:800]}"

    out = {
        "mode": mode,
        "status": status,
        "median_tps": statistics.median(rates) if rates else None,
        "mean_tps": statistics.fmean(rates) if rates else None,
        "loss": losses[-1] if losses else None,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1073741824,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1073741824,
        "error": error,
    }
    del compiled, model, params, named
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measured", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.03)
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
    x, y = make_inputs()
    valid_tokens = int(y.ne(-100).sum().item())

    cases = [run_case(mode, x, y, a.warmup, a.measured, valid_tokens) for mode in MODES]
    baseline = next(c for c in cases if c["mode"] == "max-autotune")
    valid = [c for c in cases if c["status"] == "ok" and c["median_tps"] is not None]
    best = max(valid, key=lambda c: c["median_tps"]) if valid else None
    failures: list[str] = []
    ratio = None
    loss_diff = None
    if baseline["status"] != "ok":
        failures.append("fast4_baseline_failed")
    if best is None:
        failures.append("no_valid_candidate")
    elif baseline["status"] == "ok":
        ratio = best["median_tps"] / baseline["median_tps"]
        loss_diff = abs(best["loss"] - baseline["loss"])
        if best["mode"] == "max-autotune":
            failures.append("no_compile_mode_improvement")
        elif ratio < a.min_ratio:
            failures.append("speed_ratio")
        if loss_diff > a.max_loss_diff:
            failures.append("loss_fidelity")
        if best["peak_reserved_gib"] > a.max_peak_reserved_gib:
            failures.append("peak_reserved")

    report = {
        "schema": 1,
        "gpu": gpu,
        "torch": torch.__version__,
        "batch": BATCH,
        "seq": SEQ,
        "valid_tokens_per_step": valid_tokens,
        "cases": cases,
        "baseline_mode": "max-autotune",
        "best": best,
        "best_vs_fast4_ratio": ratio,
        "loss_abs_diff_vs_fast4": loss_diff,
        "limits": {
            "min_ratio": a.min_ratio,
            "max_loss_diff": a.max_loss_diff,
            "max_peak_reserved_gib": a.max_peak_reserved_gib,
        },
        "promote_compile_mode": not failures,
        "failures": failures,
        "note": "benchmark only; identical math/weights/inputs, compile mode only; no continuity state touched",
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
