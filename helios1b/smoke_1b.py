from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .config import HeliosConfig
from .model import HeliosForCausalLM


def gib(value: int) -> float:
    return value / 2**30


def persistent_memory_estimates(parameters: int) -> dict[str, float]:
    """Simple tensor-only estimates; allocator/activations are deliberately excluded."""
    bf16 = 2 * parameters
    fp32 = 4 * parameters
    return {
        "bf16_parameters_gib": gib(bf16),
        "bf16_gradients_gib": gib(bf16),
        "two_fp32_adam_moments_gib": gib(2 * fp32),
        "bf16_params_grads_plus_fp32_adam_moments_gib": gib(
            bf16 + bf16 + 2 * fp32
        ),
        "plus_fp32_master_weights_gib": gib(
            bf16 + bf16 + 2 * fp32 + fp32
        ),
    }


def construct_bf16_cuda_model(cfg: HeliosConfig) -> HeliosForCausalLM:
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = HeliosForCausalLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("full-size smoke model was not created directly on CUDA")
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("full-size smoke model is not entirely BF16")
    return model


def run_smoke(
    *,
    seq: int,
    batch: int,
    warmup: int,
    steps: int,
    max_reserved_gib: float,
    out: str | None,
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Helios 1B smoke requires CUDA with BF16 support")
    if min(seq, batch, steps) <= 0 or warmup < 0:
        raise ValueError("invalid smoke dimensions")

    cfg = HeliosConfig()
    if seq > cfg.max_seq_len:
        raise ValueError("sequence exceeds Helios architecture contract")

    torch.manual_seed(1_006_740_992 % 2**31)
    torch.cuda.manual_seed_all(1_006_740_992 % 2**31)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = construct_bf16_cuda_model(cfg)
    model.gradient_checkpointing_enable()
    model.train()

    report = model.parameter_report()
    if report["actual"] != 1_006_740_992:
        raise RuntimeError(
            f"full model parameter mismatch: {report['actual']:,}"
        )
    if report["actual"] != report["analytical"]:
        raise RuntimeError("analytical/materialized parameter mismatch")

    model_only_allocated = torch.cuda.memory_allocated()
    model_only_reserved = torch.cuda.memory_reserved()

    x = torch.randint(
        0,
        cfg.vocab_size,
        (batch, seq),
        device="cuda",
        dtype=torch.long,
    )
    y = torch.randint(
        0,
        cfg.vocab_size,
        (batch, seq),
        device="cuda",
        dtype=torch.long,
    )

    losses: list[float] = []

    def iteration() -> float:
        model.zero_grad(set_to_none=True)
        result = model(x, y)
        if (
            result.loss is None
            or result.cross_entropy is None
            or not torch.isfinite(result.loss)
            or not torch.isfinite(result.cross_entropy)
        ):
            raise FloatingPointError("non-finite Helios 1B smoke loss")
        result.loss.backward()

        first_grad = model.blocks[0].attn.q_proj.weight.grad
        last_grad = model.blocks[-1].mlp.down.weight.grad
        if first_grad is None or last_grad is None:
            raise RuntimeError("missing gradients in full-size smoke")
        if not torch.isfinite(first_grad).all() or not torch.isfinite(last_grad).all():
            raise FloatingPointError("non-finite gradients in full-size smoke")
        return float(result.loss.detach())

    for _ in range(warmup):
        losses.append(iteration())

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        losses.append(iteration())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tokens = batch * seq * steps
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    device_props = torch.cuda.get_device_properties(0)

    result = {
        "schema": 1,
        "suite": "helios1b-bf16-fwd-bwd-v1",
        "purpose": (
            "Full-size architecture/VRAM/throughput feasibility only. "
            "No optimizer step and no learning-quality claim."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "bf16": True,
        "parameters": report["actual"],
        "seq": seq,
        "batch": batch,
        "warmup": warmup,
        "measured_steps": steps,
        "activation_checkpointing": True,
        "optimizer_included": False,
        "loss_last": losses[-1],
        "finite_losses": all(math.isfinite(v) for v in losses),
        "elapsed_seconds": elapsed,
        "tokens_measured": tokens,
        "tokens_per_second_fwd_bwd": tokens / max(elapsed, 1e-9),
        "model_only_allocated_gib": gib(model_only_allocated),
        "model_only_reserved_gib": gib(model_only_reserved),
        "peak_allocated_gib": gib(peak_allocated),
        "peak_reserved_gib": gib(peak_reserved),
        "total_vram_gib": gib(device_props.total_memory),
        "vram_headroom_gib": gib(device_props.total_memory - peak_reserved),
        "max_reserved_gate_gib": max_reserved_gib,
        "persistent_tensor_memory_estimates": persistent_memory_estimates(
            report["actual"]
        ),
        "pass": (
            peak_reserved <= max_reserved_gib * 2**30
            and all(math.isfinite(v) for v in losses)
        ),
        "notes": [
            "Throughput excludes optimizer.step(), dataloading and checkpoint I/O.",
            "Standard FP32 Adam-state feasibility is not inferred from this smoke.",
            "The next optimizer-inclusive gate must be measured separately.",
        ],
    }

    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["pass"]:
        raise SystemExit(
            "Helios 1B VRAM/finite gate failed: "
            f"peak_reserved={result['peak_reserved_gib']:.2f} GiB"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--max-reserved-gib", type=float, default=11.25)
    ap.add_argument("--out")
    args = ap.parse_args()
    run_smoke(
        seq=args.seq,
        batch=args.batch,
        warmup=args.warmup,
        steps=args.steps,
        max_reserved_gib=args.max_reserved_gib,
        out=args.out,
    )


if __name__ == "__main__":
    main()
