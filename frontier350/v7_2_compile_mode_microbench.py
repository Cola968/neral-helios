from __future__ import annotations

"""Process-isolated compile-mode sweep for the Fast4 execution shape on RTX 4070.

Each compile mode runs in a fresh Python process so TorchDynamo/CUDAGraph state and the
process-global RoPE cache cannot leak between candidates.  RoPE is prewarmed using the
actual model device (cuda:0), matching the cache key used by the forward pass.  This is a
benchmark-only tool: no checkpoint, corpus, optimizer state or continuity journal is read
or written.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
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
MODES = (
    "max-autotune-no-cudagraphs",
    "max-autotune",
    "reduce-overhead",
    "default",
)
BASELINE_MODE = "max-autotune-no-cudagraphs"


def prewarm_rope_for_model(model: Neral350M) -> str:
    device = next(model.parameters()).device
    _rope_table(
        FRONTIER_350.max_seq_len,
        FRONTIER_350.head_dim,
        FRONTIER_350.rope_theta,
        device,
        torch.bfloat16,
    )
    return str(device)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED_INPUT)
    torch.cuda.manual_seed_all(SEED_INPUT)
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


def run_single(mode: str, warmup: int, measured: int) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"expected RTX 4070, got {gpu}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED_MODEL)
    torch.cuda.manual_seed_all(SEED_MODEL)
    model = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    named = configure_trainable(model, 4)
    params = [p for _, p in named]
    if sum(p.numel() for p in model.parameters()) != 349_825_152:
        raise RuntimeError("parameter contract changed")

    rope_device = prewarm_rope_for_model(model)
    x, y = make_inputs()
    valid_tokens = int(y.ne(-100).sum().item())

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
            if mode != "max-autotune-no-cudagraphs" and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
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
        error = f"{type(exc).__name__}:{str(exc)[:1200]}"

    return {
        "mode": mode,
        "status": status,
        "median_tps": statistics.median(rates) if rates else None,
        "mean_tps": statistics.fmean(rates) if rates else None,
        "loss": losses[-1] if losses else None,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1073741824,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1073741824,
        "rope_prewarm_device": rope_device,
        "valid_tokens_per_step": valid_tokens,
        "error": error,
    }


def child_main(args: argparse.Namespace) -> None:
    result = run_single(args.single_mode, args.warmup, args.measured)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def parent_main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"expected RTX 4070, got {gpu}")

    script = Path(__file__).resolve()
    cases: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="neral_v72_modes_") as td:
        temp = Path(td)
        for mode in MODES:
            result_path = temp / (mode.replace("-", "_") + ".json")
            env = os.environ.copy()
            base_cache = env.get("TORCHINDUCTOR_CACHE_DIR", str(temp / "inductor"))
            env["TORCHINDUCTOR_CACHE_DIR"] = base_cache + "_" + mode.replace("-", "_")
            command = [
                sys.executable, "-u", str(script),
                "--out", str(result_path),
                "--single-mode", mode,
                "--warmup", str(args.warmup),
                "--measured", str(args.measured),
            ]
            completed = subprocess.run(command, env=env)
            if result_path.exists():
                cases.append(json.loads(result_path.read_text(encoding="utf-8")))
            else:
                cases.append({
                    "mode": mode,
                    "status": "failed",
                    "median_tps": None,
                    "mean_tps": None,
                    "loss": None,
                    "peak_allocated_gib": None,
                    "peak_reserved_gib": None,
                    "rope_prewarm_device": None,
                    "valid_tokens_per_step": None,
                    "error": f"child_exit_without_report:{completed.returncode}",
                })

    baseline = next(c for c in cases if c["mode"] == BASELINE_MODE)
    valid = [c for c in cases if c["status"] == "ok" and c["median_tps"] is not None]
    best = max(valid, key=lambda c: c["median_tps"]) if valid else None
    failures: list[str] = []
    ratio = None
    loss_diff = None
    if baseline["status"] != "ok":
        failures.append("baseline_failed")
    if best is None:
        failures.append("no_valid_candidate")
    elif baseline["status"] == "ok":
        ratio = best["median_tps"] / baseline["median_tps"]
        loss_diff = abs(best["loss"] - baseline["loss"])
        if best["mode"] == BASELINE_MODE:
            failures.append("no_compile_mode_improvement")
        elif ratio < args.min_ratio:
            failures.append("speed_ratio")
        if loss_diff > args.max_loss_diff:
            failures.append("loss_fidelity")
        if best["peak_reserved_gib"] > args.max_peak_reserved_gib:
            failures.append("peak_reserved")

    report = {
        "schema": 2,
        "gpu": gpu,
        "torch": torch.__version__,
        "batch": BATCH,
        "seq": SEQ,
        "process_isolation": True,
        "exact_device_rope_prewarm": True,
        "cases": cases,
        "baseline_mode": BASELINE_MODE,
        "best": best,
        "best_vs_baseline_ratio": ratio,
        "loss_abs_diff_vs_baseline": loss_diff,
        "limits": {
            "min_ratio": args.min_ratio,
            "max_loss_diff": args.max_loss_diff,
            "max_peak_reserved_gib": args.max_peak_reserved_gib,
        },
        "promote_compile_mode": not failures,
        "failures": failures,
        "note": "benchmark only; fresh process per mode; exact cuda:0 RoPE cache key; no continuity state touched",
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-mode", choices=MODES, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measured", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.03)
    ap.add_argument("--max-loss-diff", type=float, default=2e-4)
    ap.add_argument("--max-peak-reserved-gib", type=float, default=9.0)
    args = ap.parse_args()
    if args.single_mode:
        child_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
