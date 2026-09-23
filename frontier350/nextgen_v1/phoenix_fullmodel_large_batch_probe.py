"""Probe larger full-model microbatches on RTX 4070 for Neral 350M.

Engineering-only throughput benchmark. Each candidate starts from the immutable
historical weights with a fresh AdamW optimizer. OOM is an expected result, not a
benchmark failure. No checkpoint or training candidate is retained.
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
BATCHES = (4, 8, 12, 16)
TIMED_UPDATES = 8
LR = 1e-5
MAX_RESERVED_MIB = 11_000


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
        p.error("new absolute output required")

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

    for index, microbatch in enumerate(BATCHES):
        guard.require_safe_runtime_temperature()
        torch.manual_seed(20260923)
        torch.cuda.manual_seed_all(20260923)
        random.seed(20260923)
        np.random.seed(20260923)
        rng = random.Random(20260924)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = None
        optimizer = None
        row = {
            "index": index,
            "seq": SEQ,
            "microbatch": microbatch,
            "grad_accum": 1,
            "gradient_checkpointing": False,
            "tokens_per_update": SEQ * microbatch,
        }
        try:
            model = Neral350M().to(device=device, dtype=torch.bfloat16)
            model.set_gradient_checkpointing(False)
            model.load_state_dict(base_weights, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, betas=(0.9,0.95), eps=1e-8,
                weight_decay=0.1, fused=True
            )
            model.train()

            def update():
                optimizer.zero_grad(set_to_none=True)
                x, y = pool.batch(microbatch, SEQ, device, rng)
                _, loss = model(x, y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite batch-scaling loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                return float(loss.detach()), float(grad)

            # Warmup materializes AdamW state and kernels.
            update()
            torch.cuda.synchronize()
            guard.require_safe_runtime_temperature()

            losses = []
            start = time.perf_counter()
            for _ in range(TIMED_UPDATES):
                loss, grad = update()
                if not math.isfinite(grad):
                    raise FloatingPointError("nonfinite batch-scaling gradient")
                losses.append(loss)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            guard.require_safe_runtime_temperature()
            reserved = torch.cuda.max_memory_reserved() / 1024**2
            allocated = torch.cuda.max_memory_allocated() / 1024**2
            if reserved > MAX_RESERVED_MIB:
                raise RuntimeError("reserved VRAM exceeds conservative ceiling")
            timed_tokens = TIMED_UPDATES * SEQ * microbatch
            row.update({
                "status":"pass",
                "timed_updates":TIMED_UPDATES,
                "timed_tokens":timed_tokens,
                "elapsed_seconds":elapsed,
                "tokens_per_second":timed_tokens/elapsed,
                "mean_loss":sum(losses)/len(losses),
                "peak_allocated_mib":round(allocated,1),
                "peak_reserved_mib":round(reserved,1),
            })
        except torch.cuda.OutOfMemoryError:
            row["status"] = "oom"
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                row["status"] = "oom"
            else:
                row.update({"status":"runtime_fail","reason":type(exc).__name__})
        finally:
            print(json.dumps({"event":"fullmodel_large_batch_candidate", **row},
                             sort_keys=True), flush=True)
            results.append(row)
            if optimizer is not None:
                del optimizer
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    passing = [r for r in results if r.get("status") == "pass"]
    if not passing:
        raise RuntimeError("no large-batch shape passed")
    best = max(passing, key=lambda r:r["tokens_per_second"])
    report = {
        "event":"phoenix_fullmodel_large_batch_probe_finished",
        "classification":"ENGINEERING_ONLY_NO_CHECKPOINT",
        "parameters":349_825_152,
        "layers":24,
        "seq":SEQ,
        "results":results,
        "best":best,
        "candidate_written":False,
        "production_eligible":False,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
