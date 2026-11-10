from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import HeliosConfig
from .data import DeterministicTokenBatcher, fixed_eval_batches
from .learning_experiment import build_learnability_stream, write_stream
from .model import HeliosForCausalLM
from .smoke_1b import construct_bf16_cuda_model, gib


def optimizer_state_summary(optimizer: torch.optim.Optimizer) -> dict:
    by_key: dict[str, dict[str, int]] = {}
    total_bytes = 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            dtype = str(value.dtype).replace("torch.", "")
            row = by_key.setdefault(key, {})
            row[dtype] = row.get(dtype, 0) + int(value.numel())
            total_bytes += value.numel() * value.element_size()
    return {
        "tensor_elements_by_state_key_and_dtype": by_key,
        "total_state_bytes": total_bytes,
        "total_state_gib": gib(total_bytes),
    }


def require_bf16_adam_moments(summary: dict) -> None:
    by_key = summary["tensor_elements_by_state_key_and_dtype"]
    for key in ("exp_avg", "exp_avg_sq"):
        dtypes = by_key.get(key, {})
        if not dtypes:
            raise RuntimeError(f"missing Adam state: {key}")
        if set(dtypes) != {"bfloat16"}:
            raise RuntimeError(
                f"{key} is not exclusively BF16: {sorted(dtypes)}"
            )


@torch.inference_mode()
def eval_proxy(
    model: HeliosForCausalLM,
    val_path: Path,
    *,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for x, y in fixed_eval_batches(
        val_path,
        seq_len=seq_len,
        batch_size=batch_size,
        batches=batches,
        seed=seed,
        device=torch.device("cuda"),
        dtype="uint16",
    ):
        out = model(x, y)
        if out.cross_entropy is None or not torch.isfinite(out.cross_entropy):
            raise FloatingPointError("non-finite proxy validation CE")
        total += float(out.cross_entropy) * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count


def lr_for(step: int, total_steps: int, peak: float, floor: float) -> float:
    warmup = max(2, total_steps // 10)
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return floor + (peak - floor) * cosine


def run_proxy_precision_ab(*, steps: int, out: str | Path) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    if steps < 30:
        raise ValueError("proxy optimizer A/B requires at least 30 steps")

    cfg = replace(
        HeliosConfig.tiny_proxy(),
        vocab_size=128,
        max_seq_len=64,
    )
    seq_len = 48
    batch_size = 8
    peak_lr = 1e-3
    floor_lr = 1e-4
    seed = 96813

    with tempfile.TemporaryDirectory(prefix="helios1b-opt-ab-") as td:
        root = Path(td)
        train_path = root / "train.bin"
        val_path = root / "val.bin"
        write_stream(train_path, build_learnability_stream(96810, 800))
        write_stream(val_path, build_learnability_stream(96811, 240))

        torch.manual_seed(seed)
        base_cpu = HeliosForCausalLM(cfg).cpu()
        initial_state = {
            k: v.detach().clone() for k, v in base_cpu.state_dict().items()
        }
        del base_cpu

        def one_case(name: str, dtype: torch.dtype) -> dict:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            old_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                with torch.device("cuda"):
                    model = HeliosForCausalLM(cfg)
            finally:
                torch.set_default_dtype(old_dtype)
            model.load_state_dict(initial_state, strict=True)
            model.train()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=peak_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.05,
                fused=True,
            )
            batcher = DeterministicTokenBatcher(
                train_path,
                seq_len=seq_len,
                batch_size=batch_size,
                seed=seed + 101,
                dtype="uint16",
                vocab_size=cfg.vocab_size,
            )

            initial_ce = eval_proxy(
                model,
                val_path,
                seq_len=seq_len,
                batch_size=batch_size,
                batches=8,
                seed=seed + 202,
            )
            best_ce = initial_ce
            losses: list[float] = []
            started = time.perf_counter()
            for step in range(1, steps + 1):
                lr = lr_for(step, steps, peak_lr, floor_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                x, y = batcher.next_batch(torch.device("cuda"))
                optimizer.zero_grad(set_to_none=True)
                result = model(x, y)
                if result.loss is None or not torch.isfinite(result.loss):
                    raise FloatingPointError(f"{name}: non-finite train loss")
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                losses.append(float(result.loss.detach()))

                if step % 10 == 0 or step == steps:
                    ce = eval_proxy(
                        model,
                        val_path,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        batches=8,
                        seed=seed + 202,
                    )
                    best_ce = min(best_ce, ce)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            state = optimizer_state_summary(optimizer)
            final_ce = eval_proxy(
                model,
                val_path,
                seq_len=seq_len,
                batch_size=batch_size,
                batches=8,
                seed=seed + 202,
            )
            relative = (initial_ce - best_ce) / initial_ce
            return {
                "name": name,
                "parameter_dtype": str(dtype).replace("torch.", ""),
                "initial_val_ce": initial_ce,
                "final_val_ce": final_ce,
                "best_val_ce": best_ce,
                "relative_val_improvement": relative,
                "finite_train_losses": all(math.isfinite(v) for v in losses),
                "optimizer_state": state,
                "elapsed_seconds": elapsed,
                "training_tokens": steps * batch_size * seq_len,
                "tokens_per_second_with_optimizer": (
                    steps * batch_size * seq_len / max(elapsed, 1e-9)
                ),
                "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
            }

        fp32 = one_case("fp32-params-fp32-adam-state", torch.float32)
        bf16 = one_case("bf16-params-bf16-adam-state", torch.bfloat16)
        require_bf16_adam_moments(bf16["optimizer_state"])

    quality_ratio = bf16["best_val_ce"] / fp32["best_val_ce"]
    passed = (
        fp32["relative_val_improvement"] >= 0.45
        and bf16["relative_val_improvement"] >= 0.45
        and quality_ratio <= 1.10
        and fp32["finite_train_losses"]
        and bf16["finite_train_losses"]
    )
    report = {
        "schema": 1,
        "suite": "helios1b-optimizer-precision-ab-v1",
        "purpose": (
            "Synthetic learning-stability comparison for the memory-feasible "
            "BF16 Adam-state path; not a language-quality benchmark."
        ),
        "steps": steps,
        "fp32_control": fp32,
        "bf16_candidate": bf16,
        "bf16_to_fp32_best_val_ce_ratio": quality_ratio,
        "gate": {
            "minimum_relative_improvement_each": 0.45,
            "maximum_bf16_to_fp32_best_ce_ratio": 1.10,
        },
        "pass": passed,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("Helios BF16 optimizer proxy A/B failed")
    return report


def run_full_optimizer_smoke(
    *,
    seq: int,
    batch: int,
    warmup: int,
    steps: int,
    max_reserved_gib: float,
    out: str | Path,
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    if min(seq, batch, steps) <= 0 or warmup < 1:
        raise ValueError("invalid full optimizer smoke dimensions")

    cfg = HeliosConfig()
    if seq > cfg.max_seq_len:
        raise ValueError("sequence exceeds model context contract")

    torch.manual_seed(96814)
    torch.cuda.manual_seed_all(96814)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = construct_bf16_cuda_model(cfg)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    x = torch.randint(
        0, cfg.vocab_size, (batch, seq), device="cuda", dtype=torch.long
    )
    y = torch.randint(
        0, cfg.vocab_size, (batch, seq), device="cuda", dtype=torch.long
    )

    def update() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        result = model(x, y)
        if result.loss is None or not torch.isfinite(result.loss):
            raise FloatingPointError("non-finite full optimizer loss")
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        return float(result.loss.detach()), float(grad_norm)

    warmup_losses = []
    for _ in range(warmup):
        loss, _ = update()
        warmup_losses.append(loss)
    torch.cuda.synchronize()

    state = optimizer_state_summary(optimizer)
    require_bf16_adam_moments(state)
    persistent_allocated = torch.cuda.memory_allocated()
    persistent_reserved = torch.cuda.memory_reserved()

    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    grad_norms: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        loss, grad_norm = update()
        losses.append(loss)
        grad_norms.append(grad_norm)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    tokens = seq * batch * steps
    passed = (
        all(math.isfinite(v) for v in warmup_losses + losses + grad_norms)
        and peak_reserved <= max_reserved_gib * 2**30
    )
    report = {
        "schema": 1,
        "suite": "helios1b-full-optimizer-smoke-v1",
        "purpose": (
            "Memory/throughput/numerical feasibility of full 1B fused AdamW "
            "with BF16 parameter-matched moments. Not a language-quality claim."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "parameters": model.parameter_report()["actual"],
        "seq": seq,
        "batch": batch,
        "warmup_updates": warmup,
        "measured_updates": steps,
        "activation_checkpointing": True,
        "optimizer": "torch.optim.AdamW(fused=True)",
        "optimizer_state": state,
        "persistent_allocated_gib_after_state_init": gib(persistent_allocated),
        "persistent_reserved_gib_after_state_init": gib(persistent_reserved),
        "peak_allocated_gib": gib(peak_allocated),
        "peak_reserved_gib": gib(peak_reserved),
        "total_vram_gib": gib(total_vram),
        "vram_headroom_gib": gib(total_vram - peak_reserved),
        "max_reserved_gate_gib": max_reserved_gib,
        "losses": losses,
        "grad_norms": grad_norms,
        "finite": all(
            math.isfinite(v) for v in warmup_losses + losses + grad_norms
        ),
        "elapsed_seconds": elapsed,
        "tokens_measured": tokens,
        "tokens_per_second_end_to_end_update": tokens / max(elapsed, 1e-9),
        "pass": passed,
        "notes": [
            "Optimizer state dtype is measured, not assumed.",
            "This is still synthetic random-token smoke data.",
            "Promotion to a real-data pilot also requires the separate proxy precision A/B gate.",
        ],
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("Helios full 1B optimizer smoke failed")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("proxy-ab", "full-smoke"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-reserved-gib", type=float, default=11.25)
    args = ap.parse_args()

    if args.mode == "proxy-ab":
        run_proxy_precision_ab(steps=args.steps or 60, out=args.out)
    else:
        run_full_optimizer_smoke(
            seq=args.seq,
            batch=args.batch,
            warmup=args.warmup,
            steps=args.steps or 2,
            max_reserved_gib=args.max_reserved_gib,
            out=args.out,
        )


if __name__ == "__main__":
    main()
