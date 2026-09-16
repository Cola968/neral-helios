"""Opt-in COMPUTE-ONLY GPU probe of Neral's own 350M architecture.

Synthetic fixed batch, no real data/checkpoints/accepted stages or quality proof.
The 50k target in this file applies ONLY to synthetic compute-only tokens/s.
Never execute while any existing Neral GPU workload is active.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SYNTHETIC_TARGET_TPS = 50_000
MAX_BATCH_TOKENS = 8192


def validate_shape(batch: int, seq: int, train_blocks: int) -> None:
    """Fail before loading CUDA for oversized unproven shapes."""
    if batch not in (2, 4, 6, 8, 12, 16) or seq not in (256, 512, 1024):
        raise ValueError("Unsupported synthetic shape")
    if train_blocks not in (4, 8, 24):
        raise ValueError("Unsupported trainable block count")
    if batch * seq > MAX_BATCH_TOKENS:
        raise ValueError("Synthetic candidate exceeds guarded 8192-token microbatch ceiling")
    if batch >= 12 and train_blocks != 4:
        raise ValueError("Large experimental batches restricted to four trainable blocks")


def gpu_temperature() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    values = result.stdout.strip().splitlines()
    if len(values) != 1 or not values[0].strip().isdigit():
        raise RuntimeError("Expected exactly one readable GPU temperature")
    return int(values[0].strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="explicit consent to use idle local GPU")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-blocks", type=int, choices=(4, 8, 24), default=4)
    p.add_argument("--execution", choices=("eager", "compile-default", "compile-overhead"), default="eager")
    p.add_argument("--batch", type=int, choices=(2, 4, 6, 8, 12, 16), default=4)
    p.add_argument("--seq", type=int, choices=(256, 512, 1024), default=512)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=12)
    args = p.parse_args()
    if not args.execute:
        p.error("GPU execution requires --execute; script is opt-in")
    if args.warmup < 2 or not 3 <= args.steps <= 100:
        p.error("warmup >=2 and measured steps 3..100 required")
    if args.out.exists():
        p.error("refusing to overwrite existing evidence")
    try:
        validate_shape(args.batch, args.seq, args.train_blocks)
    except ValueError as exc:
        p.error(str(exc))

    import torch
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model350 import Neral350M, FRONTIER_350
    from protection_v9 import configure_trainable

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTX CUDA BF16 GPU required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"Unexpected device: {gpu}")
    if gpu_temperature() > 78:
        raise RuntimeError("Thermal preflight failed (>78 C)")
    free, _ = torch.cuda.mem_get_info()
    if free < 9000 * 1024**2:
        raise RuntimeError("GPU not idle: less than 9000 MiB free; refuse contention")

    setup_start = time.perf_counter()
    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Neral350M().to(device="cuda", dtype=torch.bfloat16)
    named = configure_trainable(model, args.train_blocks)
    params = [param for _, param in named]
    if sum(p.numel() for p in model.parameters()) != 349_825_152:
        raise RuntimeError("Changed Neral model architecture")
    optimizer = torch.optim.AdamW(params, lr=2e-5, fused=True)
    execution = model
    if args.execution != "eager":
        mode = "default" if args.execution == "compile-default" else "reduce-overhead"
        execution = torch.compile(model, mode=mode, dynamic=False, fullgraph=True)
    tokens = torch.randint(0, FRONTIER_350.vocab_size, (args.batch, args.seq + 1), device="cuda")
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    # One fixed batch is reused to isolate compute. Synthetic loss is NOT learning evidence.
    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = execution(x, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite loss before update")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("non-finite gradient")
        optimizer.step()
        return float(loss.detach())

    peak_temp = gpu_temperature()
    for _ in range(args.warmup):
        temp = gpu_temperature()
        peak_temp = max(peak_temp, temp)
        if temp > 82:
            raise RuntimeError("Thermal guard hit in warmup")
        step()
        torch.cuda.synchronize()
        if torch.cuda.max_memory_reserved() > 9 * 1024**3:
            raise RuntimeError("9 GiB reserved VRAM guard hit in warmup; candidate rejected")
    warmup_and_setup_seconds = time.perf_counter() - setup_start
    torch.cuda.reset_peak_memory_stats()
    losses = []
    durations = []
    for _ in range(args.steps):
        temp = gpu_temperature()
        peak_temp = max(peak_temp, temp)
        if temp > 82:
            raise RuntimeError("Thermal guard hit; probe aborted")
        started = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        if torch.cuda.max_memory_reserved() > 9 * 1024**3:
            raise RuntimeError("9 GiB reserved VRAM guard hit; candidate rejected")
    peak_temp = max(peak_temp, gpu_temperature())
    if peak_temp > 82:
        raise RuntimeError("Thermal guard hit after measurement")
    total_duration = sum(durations)
    tokens_timed = args.steps * args.batch * args.seq
    rate = tokens_timed / total_duration
    report = {
        "evidence_version": 2,
        "classification": "compute_only_synthetic_unaccepted_tokens",
        "gpu": gpu,
        "model_sha256": hashlib.sha256((root / "model350.py").read_bytes()).hexdigest(),
        "parameters": 349_825_152,
        "trainable_parameters": sum(param.numel() for param in params),
        "train_blocks": args.train_blocks,
        "execution": args.execution,
        "batch": args.batch, "seq": args.seq, "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "timed_synthetic_tokens": tokens_timed,
        "compute_only_tokens_per_second": rate,
        "measured_wall_seconds": total_duration,
        "setup_and_warmup_wall_seconds": warmup_and_setup_seconds,
        "setup_inclusive_synthetic_tokens_per_second": tokens_timed / (warmup_and_setup_seconds + total_duration),
        "slowest_measured_step_tokens_per_second": args.batch * args.seq / max(durations),
        "synthetic_50k_target": SYNTHETIC_TARGET_TPS,
        "synthetic_50k_timed_target_met": rate >= SYNTHETIC_TARGET_TPS,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "peak_temperature_c": peak_temp,
        "first_measured_loss_synthetic": losses[0],
        "last_measured_loss_synthetic": losses[-1],
        "eligible_for_production": False,
        "limitations": "No real corpus, H2D, replay, validation, checkpoint/recovery, frozen evaluation, accepted stages, NIS2 or capability test. Compilation/setup/warmup excluded from steady-state rate and separately reported. Do not compare directly to end-to-end rates.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    try:
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(tmp, args.out)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
