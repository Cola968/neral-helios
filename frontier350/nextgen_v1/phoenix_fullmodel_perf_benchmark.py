"""Disposable 24/24-block Neral 350M training-throughput benchmark.

This benchmark measures real full-model BF16 forward/backward/gradient clipping/fused
AdamW update throughput on the owner's RTX 4070. It is NOT productive training and
NOT a quality candidate. It reads the immutable SHA-pinned historical checkpoint,
uses deterministic synthetic token IDs only, writes aggregate benchmark JSON only,
and never writes model or optimizer weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
EXPECTED_PARAMETERS = 349_825_152
LR = 1e-5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--trusted-own-checkpoint", action="store_true")
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--microbatch", type=int, required=True)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
    ap.add_argument("--compile-mode", choices=("none", "default"), default="none")
    ap.add_argument("--warmup-updates", type=int, default=2)
    ap.add_argument("--measure-updates", type=int, default=6)
    args = ap.parse_args()

    if not args.execute or not args.trusted_own_checkpoint:
        ap.error("explicit execution and trust in the owner's SHA-pinned checkpoint are required")
    if min(args.seq, args.microbatch, args.grad_accum, args.warmup_updates, args.measure_updates) <= 0:
        raise ValueError("positive benchmark dimensions required")

    from phoenix_continuity_cpu_probe import require_pinned_checkpoint
    from phoenix_weight_only_pilot import _checked_historical_weights
    import phoenix_cuda_diagnostic as gpu_guard

    source = require_pinned_checkpoint(args.source_checkpoint)
    if sha256_file(source) != SOURCE_SHA:
        raise RuntimeError("historical checkpoint SHA mismatch")

    gpu_guard.require_stably_idle_gpu()

    import torch
    from model350 import FRONTIER_350, Neral350M

    gpu_guard.require_cuda_ready(torch)
    if args.seq > FRONTIER_350.max_seq_len:
        raise ValueError("sequence exceeds architecture context")

    torch.manual_seed(35024)
    torch.cuda.manual_seed_all(35024)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    historical = _checked_historical_weights(
        source, torch, Neral350M, FRONTIER_350.to_dict()
    )
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(args.gradient_checkpointing == "on")
    if model.parameter_report()["total"] != EXPECTED_PARAMETERS:
        raise RuntimeError("locked parameter count changed")
    model.load_state_dict(historical["model"], strict=True)
    del historical

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.1, fused=True
    )
    if optimizer.state_dict()["state"]:
        raise RuntimeError("benchmark optimizer must start fresh")

    train_model = model
    if args.compile_mode != "none":
        train_model = torch.compile(
            model, backend="inductor", mode=args.compile_mode,
            fullgraph=False, dynamic=False
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(35024)

    def one_update() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        weighted = torch.zeros((), dtype=torch.float32, device=device)
        for _ in range(args.grad_accum):
            x = torch.randint(
                low=0, high=FRONTIER_350.vocab_size,
                size=(args.microbatch, args.seq),
                generator=generator, device=device, dtype=torch.long
            )
            y = torch.randint(
                low=0, high=FRONTIER_350.vocab_size,
                size=(args.microbatch, args.seq),
                generator=generator, device=device, dtype=torch.long
            )
            _, loss = train_model(x, y)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError("non-finite benchmark loss")
            (loss / args.grad_accum).backward()
            weighted.add_(loss.detach().float() / args.grad_accum)
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if not math.isfinite(float(grad.detach())):
            raise FloatingPointError("non-finite benchmark gradient norm")
        optimizer.step()
        return float(weighted.detach()), float(grad.detach())

    tokens_per_update = args.seq * args.microbatch * args.grad_accum
    model.train()

    # Warm-up is deliberately excluded from reported throughput so torch.compile
    # compilation and first-use allocator/optimizer costs do not inflate runtime.
    warmup_seconds = 0.0
    for _ in range(args.warmup_updates):
        t0 = time.perf_counter()
        one_update()
        torch.cuda.synchronize()
        warmup_seconds += time.perf_counter() - t0
        gpu_guard.require_safe_runtime_temperature()

    torch.cuda.reset_peak_memory_stats()
    measured_losses = []
    measured_grad_norms = []
    per_update_seconds = []
    start = time.perf_counter()
    for _ in range(args.measure_updates):
        t0 = time.perf_counter()
        loss, grad = one_update()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_update_seconds.append(dt)
        measured_losses.append(loss)
        measured_grad_norms.append(grad)
        gpu_guard.require_safe_runtime_temperature()
    measured_seconds = time.perf_counter() - start

    measured_tokens = tokens_per_update * args.measure_updates
    source_unchanged = sha256_file(source) == SOURCE_SHA
    if not source_unchanged:
        raise RuntimeError("historical source checkpoint changed")

    summary = {
        "schema": "phoenix_fullmodel_perf_benchmark_v1",
        "classification": "disposable_engineering_benchmark_not_quality_training",
        "parameters": EXPECTED_PARAMETERS,
        "full_model_trainable": True,
        "n_layers": FRONTIER_350.n_layers,
        "seq": args.seq,
        "microbatch": args.microbatch,
        "grad_accum": args.grad_accum,
        "tokens_per_update": tokens_per_update,
        "gradient_checkpointing": args.gradient_checkpointing == "on",
        "compile_mode": args.compile_mode,
        "warmup_updates": args.warmup_updates,
        "warmup_seconds": warmup_seconds,
        "measure_updates": args.measure_updates,
        "measured_tokens": measured_tokens,
        "measured_seconds": measured_seconds,
        "tokens_per_second": measured_tokens / measured_seconds,
        "seconds_per_update": measured_seconds / args.measure_updates,
        "min_update_seconds": min(per_update_seconds),
        "max_update_seconds": max(per_update_seconds),
        "mean_loss": sum(measured_losses) / len(measured_losses),
        "mean_grad_norm": sum(measured_grad_norms) / len(measured_grad_norms),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "gpu": torch.cuda.get_device_name(0),
        "source_sha256": SOURCE_SHA,
        "source_unchanged": source_unchanged,
        "synthetic_token_ids_only": True,
        "checkpoint_written": False,
        "quality_improvement_verified": False,
        "productive_training_tokens": 0,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
