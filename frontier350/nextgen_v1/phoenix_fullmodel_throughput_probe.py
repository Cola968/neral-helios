"""Real-data full-model throughput probe for the own Neral 350M checkpoint.

Measures disposable BF16 forward/backward/fused-AdamW configurations on the owner's
RTX 4070 using already-staged uint16 foundation shards. It writes only aggregate
JSON; no checkpoint is saved or promoted and source weights/data stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
PARAMETERS = 349_825_152
MAX_RESERVED_MIB = 10_500
CONFIGS = (
    {"seq": 128, "accum": 8, "gradient_checkpointing": False},
    {"seq": 256, "accum": 4, "gradient_checkpointing": False},
    {"seq": 512, "accum": 2, "gradient_checkpointing": False},
    {"seq": 512, "accum": 2, "gradient_checkpointing": True},
    {"seq": 1024, "accum": 1, "gradient_checkpointing": True},
)


def first_shard(root: Path) -> Path:
    files = sorted(root.glob("*.bin"))
    if not files:
        raise FileNotFoundError("no uint16 foundation shard found")
    path = files[0].resolve(strict=True)
    if path.is_symlink() or path.stat().st_size < 4 * 1024 * 1024 or path.stat().st_size % 2:
        raise ValueError("foundation shard failed structural checks")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--trusted-own-checkpoint", action="store_true")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit execution and trust in own checkpoint required")
    if not args.out.is_absolute() or args.out.exists() or args.out.is_symlink():
        p.error("output must be a new absolute non-symlink file")

    from phoenix_continuity_cpu_probe import require_pinned_checkpoint
    from phoenix_fullsize_gpu_smoke import sha256_file

    source = require_pinned_checkpoint(args.checkpoint)
    if sha256_file(source) != SOURCE_SHA:
        raise ValueError("historical anchor SHA changed")
    shard = first_shard(args.train_data.resolve(strict=True))
    stream = np.memmap(shard, dtype=np.uint16, mode="r")
    if len(stream) < 100_000:
        raise ValueError("foundation shard is unexpectedly small")

    import phoenix_cuda_diagnostic as guard
    guard.require_stably_idle_gpu()

    import torch
    from model350 import FRONTIER_350, Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights

    guard.require_cuda_ready(torch)
    historical = _checked_historical_weights(source, torch, Neral350M, FRONTIER_350.to_dict())
    base_weights = historical["model"]
    del historical

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    results = []

    def get_batch(seq: int, offset: int):
        limit = len(stream) - seq - 1
        start = offset % limit
        chunk = np.asarray(stream[start:start + seq + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy()).view(1, seq).to(device)
        y = torch.from_numpy(chunk[1:].copy()).view(1, seq).to(device)
        return x, y

    for index, cfg in enumerate(CONFIGS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(20260923 + index)
        torch.cuda.manual_seed_all(20260923 + index)
        row = {"index": index, **cfg, "tokens_per_update": cfg["seq"] * cfg["accum"]}
        model = optimizer = None
        try:
            model = Neral350M().to(device=device, dtype=torch.bfloat16)
            model.load_state_dict(base_weights, strict=True)
            if model.parameter_report()["total"] != PARAMETERS:
                raise RuntimeError("parameter count changed")
            model.set_gradient_checkpointing(cfg["gradient_checkpointing"])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-8,
                weight_decay=0.1, fused=True
            )
            model.train()

            # One unmeasured update materializes optimizer state and kernels.
            optimizer.zero_grad(set_to_none=True)
            for micro in range(cfg["accum"]):
                x, y = get_batch(cfg["seq"], (index * 100_000) + micro * (cfg["seq"] + 17))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite warmup loss")
                (loss / cfg["accum"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            guard.require_safe_runtime_temperature()

            measured_updates = 4
            measured_tokens = measured_updates * cfg["seq"] * cfg["accum"]
            losses = []
            started = time.perf_counter()
            for step in range(measured_updates):
                optimizer.zero_grad(set_to_none=True)
                weighted = 0.0
                for micro in range(cfg["accum"]):
                    offset = (index * 100_000) + (step + 1) * 10_000 + micro * (cfg["seq"] + 31)
                    x, y = get_batch(cfg["seq"], offset)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    if loss is None or not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite measured loss")
                    (loss / cfg["accum"]).backward()
                    weighted += float(loss.detach()) / cfg["accum"]
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(grad)):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                losses.append(weighted)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            guard.require_safe_runtime_temperature()
            reserved = torch.cuda.max_memory_reserved() / 1024**2
            allocated = torch.cuda.max_memory_allocated() / 1024**2
            if reserved > MAX_RESERVED_MIB:
                raise RuntimeError("configuration exceeds conservative VRAM ceiling")
            row.update({
                "status": "pass",
                "measured_updates": measured_updates,
                "measured_tokens": measured_tokens,
                "measured_seconds": elapsed,
                "tokens_per_second": measured_tokens / elapsed,
                "mean_train_loss": sum(losses) / len(losses),
                "peak_allocated_mib": round(allocated, 1),
                "peak_reserved_mib": round(reserved, 1),
            })
        except torch.cuda.OutOfMemoryError:
            row.update({"status": "oom"})
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg:
                row.update({"status": "oom"})
            else:
                row.update({"status": "runtime_fail", "reason": type(exc).__name__})
        finally:
            if optimizer is not None:
                del optimizer
            if model is not None:
                del model
            torch.cuda.empty_cache()
        print(json.dumps({"event": "fullmodel_shape_probe", **row}, sort_keys=True), flush=True)
        results.append(row)

    passing = [r for r in results if r.get("status") == "pass"]
    best = max(passing, key=lambda r: r["tokens_per_second"]) if passing else None
    report = {
        "event": "phoenix_fullmodel_throughput_probe_finished",
        "classification": "development_throughput_only_not_quality_training",
        "parameters": PARAMETERS,
        "layers": 24,
        "source_sha256": SOURCE_SHA,
        "real_foundation_shard_used": True,
        "checkpoint_written": False,
        "quality_gain_verified": False,
        "results": results,
        "best": best,
        "source_unchanged": sha256_file(source) == SOURCE_SHA,
    }
    if not report["source_unchanged"]:
        raise RuntimeError("historical source checkpoint changed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("event","parameters","layers","best","source_unchanged")}, sort_keys=True), flush=True)
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
