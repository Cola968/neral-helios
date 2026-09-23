"""Steady-state eager vs torch.compile probe for the winning Neral 350M shape.

Development engineering only. Uses exact historical weights and the SHA-pinned
foundation refresh corpus. Every mode starts from identical weights/seed and a fresh
AdamW optimizer. No checkpoint is written. Compile warmup is excluded from steady-state
throughput but reported separately so we can decide whether it pays off for long runs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helios350_preview import TOKENIZER_SHA
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_foundation_refresh_dev import verify_refresh_manifest

SEQ = 256
MICROBATCH = 4
GRAD_ACCUM = 1
TOKENS_PER_UPDATE = 1024
TIMED_UPDATES = 24
LR = 1e-5
MODES = ("eager", "compile-default")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if not a.execute or not a.trusted_own_checkpoint:
        p.error("explicit own-checkpoint opt-in required")
    if not a.out.is_absolute() or a.out.exists():
        p.error("new absolute report required")

    source = require_pinned_checkpoint(a.source_checkpoint)
    refresh = verify_refresh_manifest(a.corpus_root, a.manifest_sha256, TOKENIZER_SHA)

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    from model350 import FRONTIER_350, Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from train350_compiled import PersistentMMapPool

    guard.require_cuda_ready(torch)
    base = _checked_historical_weights(source, torch, Neral350M, FRONTIER_350.to_dict())
    base_weights = base["model"]
    del base
    pool = PersistentMMapPool(refresh["root"] / "train", FRONTIER_350.vocab_size)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    results = []

    for index, mode in enumerate(MODES):
        guard.require_safe_runtime_temperature()
        torch.manual_seed(20260923)
        torch.cuda.manual_seed_all(20260923)
        random.seed(20260923)
        np.random.seed(20260923)
        rng = random.Random(20260924)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = Neral350M().to(device=device, dtype=torch.bfloat16)
        model.set_gradient_checkpointing(False)
        model.load_state_dict(base_weights, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9,0.95), eps=1e-8,
            weight_decay=0.1, fused=True
        )
        train_model = model
        if mode == "compile-default":
            train_model = torch.compile(
                model, backend="inductor", mode="default", fullgraph=False, dynamic=False
            )
        model.train()

        def update():
            optimizer.zero_grad(set_to_none=True)
            x, y = pool.batch(MICROBATCH, SEQ, device, rng)
            _, loss = train_model(x, y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite compile-probe loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            return float(loss.detach()), float(grad)

        warm_start = time.perf_counter()
        warm_loss, warm_grad = update()
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - warm_start
        guard.require_safe_runtime_temperature()

        losses = []
        start = time.perf_counter()
        for _ in range(TIMED_UPDATES):
            loss, grad = update()
            if not math.isfinite(grad):
                raise FloatingPointError("nonfinite compile-probe grad")
            losses.append(loss)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        guard.require_safe_runtime_temperature()

        row = {
            "mode": mode,
            "seq": SEQ,
            "microbatch": MICROBATCH,
            "grad_accum": GRAD_ACCUM,
            "gradient_checkpointing": False,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "warmup_seconds": warm_seconds,
            "warmup_loss": warm_loss,
            "warmup_grad_norm": warm_grad,
            "timed_updates": TIMED_UPDATES,
            "timed_tokens": TIMED_UPDATES * TOKENS_PER_UPDATE,
            "elapsed_seconds": elapsed,
            "tokens_per_second": TIMED_UPDATES * TOKENS_PER_UPDATE / elapsed,
            "mean_loss": sum(losses) / len(losses),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved()/1024**2, 1),
        }
        print(json.dumps({"event":"fullmodel_compile_probe", **row}, sort_keys=True), flush=True)
        results.append(row)
        del optimizer, train_model, model
        gc.collect()
        torch.cuda.empty_cache()

    eager = next(r for r in results if r["mode"] == "eager")
    compiled = next(r for r in results if r["mode"] == "compile-default")
    speedup = compiled["tokens_per_second"] / eager["tokens_per_second"]
    # Break-even in optimizer updates from one-time extra warmup cost.
    extra_warmup = max(0.0, compiled["warmup_seconds"] - eager["warmup_seconds"])
    saved_per_update = (
        TOKENS_PER_UPDATE / eager["tokens_per_second"]
        - TOKENS_PER_UPDATE / compiled["tokens_per_second"]
    )
    break_even_updates = (
        extra_warmup / saved_per_update if saved_per_update > 0 else None
    )
    report = {
        "event":"phoenix_fullmodel_compile_probe_finished",
        "classification":"ENGINEERING_ONLY_NO_CHECKPOINT",
        "layers":24,
        "parameters":349_825_152,
        "manifest_sha256":a.manifest_sha256,
        "results":results,
        "compile_speedup_vs_eager":speedup,
        "compile_extra_warmup_seconds":extra_warmup,
        "compile_break_even_updates":break_even_updates,
        "recommended_for_new_long_lineage": bool(speedup >= 1.05 and break_even_updates is not None),
        "training_candidate_written":False,
        "production_eligible":False,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
