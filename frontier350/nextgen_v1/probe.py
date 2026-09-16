"""Opt-in COMPUTE-ONLY GPU probe of Neral's own 350M architecture.

Synthetic repeated tokens, no real data/checkpoints/accepted stages, no quality proof.
This script does NOT implement or launch the future 60k end-to-end trainer.
Do not execute while an existing Neral GPU workflow is active.
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
    p.add_argument("--batch", type=int, choices=(2, 4, 6, 8), default=4)
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
    free, total = torch.cuda.mem_get_info()
    if free < 9000 * 1024**2:
        raise RuntimeError("GPU not idle: less than 9000 MiB free; refuse contention")

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
    # A single fixed batch is deliberately reused to isolate compute-path performance.
    # This cannot be interpreted as learning, data-pipeline rate, or quality.
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
        if gpu_temperature() > 82:
            raise RuntimeError("Thermal guard hit in warmup")
        step()
        torch.cuda.synchronize()
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
    report = {
        "evidence_version": 1,
        "classification": "compute_only_synthetic_unaccepted_tokens",
        "gpu": gpu,
        "model_sha256": hashlib.sha256((root / "model350.py").read_bytes()).hexdigest(),
        "parameters": 349_825_152,
        "trainable_parameters": sum(param.numel() for param in params),
        "train_blocks": args.train_blocks,
        "execution": args.execution,
        "batch": args.batch, "seq": args.seq, "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "compute_only_tokens_per_second": args.steps * args.batch * args.seq / total_duration,
        "measured_wall_seconds": total_duration,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "peak_temperature_c": peak_temp,
        "first_measured_loss_synthetic": losses[0],
        "last_measured_loss_synthetic": losses[-1],
        "eligible_for_production": False,
        "limitations": "No corpus, H2D, replay, validation, checkpoints, recovery, frozen evaluation, accepted stages, NIS2 or capability test. Do not compare directly to end-to-end rates.",
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
