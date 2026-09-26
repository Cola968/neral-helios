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
EMERGENCY_TEMPERATURE_C = 84
COOLDOWN_RESUME_C = 74
COOLDOWN_POLL_SECONDS = 5
COOLDOWN_MAX_SECONDS = 120


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


def runtime_temperature_c() -> int:
    """Return one fail-closed nvidia-smi temperature sample."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    )
    rows = raw.strip().splitlines()
    if len(rows) != 1 or not rows[0].strip().isdecimal():
        raise RuntimeError("GPU runtime temperature unavailable")
    temperature = int(rows[0].strip())
    if temperature < 0:
        raise RuntimeError("GPU runtime temperature malformed")
    return temperature


def require_safe_runtime_temperature() -> None:
    """Strict check for diagnostics/benchmarks: do not continue above 78 C."""
    temperature = runtime_temperature_c()
    if temperature > MAX_TEMPERATURE_C:
        raise RuntimeError("GPU runtime temperature outside safety limit; stop inference")


def cooldown_until_safe() -> dict:
    """Pause compute after a thermal spike instead of discarding training lineage.

    No CUDA work is performed while cooling. Resume only below a lower hysteresis
    threshold. Telemetry failures or a GPU that does not cool within two minutes
    still fail closed.
    """
    first = runtime_temperature_c()
    if first <= MAX_TEMPERATURE_C:
        return {"cooled": False, "initial_c": first, "final_c": first, "waited_seconds": 0}
    if first > EMERGENCY_TEMPERATURE_C:
        raise RuntimeError(
            f"GPU training temperature {first} C exceeds emergency limit "
            f"{EMERGENCY_TEMPERATURE_C} C"
        )

    waited = 0
    temperature = first
    while waited < COOLDOWN_MAX_SECONDS:
        time.sleep(COOLDOWN_POLL_SECONDS)
        waited += COOLDOWN_POLL_SECONDS
        temperature = runtime_temperature_c()
        if temperature > EMERGENCY_TEMPERATURE_C:
            raise RuntimeError(
                f"GPU temperature rose to {temperature} C during cooldown; hard stop"
            )
        if temperature <= COOLDOWN_RESUME_C:
            return {
                "cooled": True, "initial_c": first, "final_c": temperature,
                "waited_seconds": waited,
            }
    raise RuntimeError(
        f"GPU remained above cooldown resume threshold after {waited}s; stop compute"
    )


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
