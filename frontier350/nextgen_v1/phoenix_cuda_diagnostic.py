"""Opt-in read-only CUDA health diagnostic of the pinned own 350M SAFE weights.

Eight PUBLIC engineering prompts, 64 tokens each at most. No training, downloads,
private prompts, model output logs, checkpoint writes or NIS2/quality claims.
This is CUDA *inference* only, never training-throughput evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from helios350_preview import SAFE_RELATIVE, SAFE_SHA, load_verified_model
from phoenix_real_smoke import require_idle_gpu, smoke


MIN_FREE_MIB = 9000
IDLE_SAMPLES = 3
IDLE_INTERVAL_SECONDS = 2
MAX_TEMPERATURE_C = 78


def require_stably_idle_gpu() -> None:
    """Require ALL three spaced snapshots to pass the original strict guard.

    A transient busy sample fails closed; no average or majority vote can hide
    a spike. Only nvidia-smi runs here; no weights, Torch or CUDA allocation.
    """
    for sample in range(1, IDLE_SAMPLES + 1):
        try:
            require_idle_gpu()  # Util <=15%, VRAM used <=2500 MiB, temp <=78 C.
        except Exception as exc:
            raise RuntimeError(
                f"GPU not stably idle at sample {sample}/{IDLE_SAMPLES}; "
                "no CUDA inference started"
            ) from exc
        if sample < IDLE_SAMPLES:
            time.sleep(IDLE_INTERVAL_SECONDS)


def require_safe_runtime_temperature() -> None:
    """Monitor heat BETWEEN public prompts; inference itself raises utilization.

    Do not use idle-utilization limits mid-inference: they would measure this
    diagnostic's own work. Missing/malformed telemetry fails closed instead.
    """
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    )
    rows = raw.strip().splitlines()
    if len(rows) != 1 or not rows[0].strip().isdecimal():
        raise RuntimeError("GPU runtime temperature unavailable; stop inference")
    temperature = int(rows[0].strip())
    if temperature < 0 or temperature > MAX_TEMPERATURE_C:
        raise RuntimeError("GPU runtime temperature outside safety limit; stop inference")


def require_cuda_ready(torch) -> None:
    """Fail before loading weights if the user's GPU is unavailable or occupied."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 unavailable; refusing GPU diagnostic")
    if torch.cuda.device_count() != 1 or "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected exactly one local RTX 4070")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < MIN_FREE_MIB * 1024 * 1024:
        raise RuntimeError("insufficient free VRAM; refusing parallel diagnostic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--assets-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute required")
    require_stably_idle_gpu()  # Three strict snapshots before importing Torch or loading weights.
    import torch
    require_cuda_ready(torch)
    root = args.assets_root.resolve(strict=True)
    model, tok, bos, eos, processed, torch = load_verified_model(
        root, root / SAFE_RELATIVE, "safe", "cuda")
    if processed != 1_010_003_584:
        raise ValueError("unexpected pinned SAFE pretraining counter")
    print(json.dumps({"status": "own_safe_cuda_diagnostic_started",
                      "checkpoint_sha256": SAFE_SHA, "device": "cuda_bf16",
                      "training_started": False, "quality_evaluated": False}), flush=True)
    result = smoke(model, tok, bos, eos, torch, budget=64,
                   before_case=require_safe_runtime_temperature,
                   after_case=require_safe_runtime_temperature)
    result.update({"checkpoint_sha256": SAFE_SHA, "device": "cuda_bf16",
                   "classification": "public_fixture_cuda_inference_health_only_not_training_speed",
                   "quality_evaluated": False, "training_started": False,
                   "checkpoint_modified": False})
    if result["duration_seconds"] > 0:
        result["generated_tokens_per_second_including_prompt_prefill"] = round(
            result["generated_tokens_total"] / result["duration_seconds"], 2)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
