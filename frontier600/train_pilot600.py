from __future__ import annotations

"""Matched short learning-curve pilots for Neral 600M architecture search.

Every candidate is trained from scratch on the same uint16 BPE shard stream, seed,
sequence length, effective tokens/update and optimizer recipe.  This is architecture
selection code, not the final production trainer.
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from architecture_search600 import CANDIDATES, PHASE_C
from model600 import make_model


class MMapPool:
    def __init__(self, root: Path):
        self.paths = sorted(root.glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no .bin shards in {root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]

    def batch(self, batch: int, seq: int, device: torch.device, rng: random.Random):
        rows = np.empty((batch, seq + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(self.arrays)
            if len(a) <= seq + 1:
                raise RuntimeError("shard shorter than sequence")
            start = rng.randrange(0, len(a) - seq - 1)
            rows[i] = a[start:start + seq + 1]
        if rows.max(initial=0) >= 16_384:
            raise RuntimeError("token outside locked Neral BPE vocabulary")
        t = torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


def evaluate(model, pool: MMapPool, *, seq: int, batch: int, batches: int,
             device: torch.device, seed: int) -> float:
        # Fixed independent validation RNG makes all candidates see identical samples.
        rng = random.Random(seed)
        was_training = model.training
        model.eval()
        vals = []
        with torch.no_grad():
            for _ in range(batches):
                x, y = pool.batch(batch, seq, device, rng)
                _, loss = model(x, y)
                vals.append(float(loss))
        model.train(was_training)
        return sum(vals) / len(vals)


def run_candidate(candidate: str, args, train_pool: MMapPool, valid_pool: MMapPool, out_root: Path) -> dict:
    device = torch.device("cuda")
    cfg = CANDIDATES[candidate]
    if args.tokens_per_update % (args.seq * args.microbatch):
        raise ValueError("tokens/update must divide seq*microbatch exactly")
    grad_accum = args.tokens_per_update // (args.seq * args.microbatch)

    # Identical seed and identical sampler stream for every architecture.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data_rng = random.Random(args.seed + 1)

    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    model = make_model(candidate, checkpoint_every=args.checkpoint_every).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.1, fused=True)
    train_model = model
    if args.compile_mode != "none":
        train_model = torch.compile(model, backend="inductor", mode=args.compile_mode,
                                    fullgraph=False, dynamic=False)

    run_dir = out_root / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf8")

    def log(row: dict):
        row = {"candidate": candidate, **row}
        print(json.dumps(row), flush=True)
        with metrics_path.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")

    initial_val = evaluate(model, valid_pool, seq=args.seq, batch=1, batches=args.eval_batches,
                           device=device, seed=args.seed + 999_999)
    log({"event": "validation", "processed_tokens": 0, "validation_loss": initial_val,
         "perplexity": math.exp(min(initial_val, 20.0))})

    model.train()
    processed = 0
    step = 0
    start_time = time.perf_counter()
    last_time = start_time
    last_tokens = 0
    last_loss = float("nan")
    last_grad = float("nan")
    best_val = initial_val

    while processed < args.target_tokens:
        warm = min(1.0, max(args.tokens_per_update, processed + args.tokens_per_update) / args.warmup_tokens)
        lr = args.lr * warm
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = torch.zeros((), device=device, dtype=torch.float32)
        micros = 0
        for _ in range(grad_accum):
            x, y = train_pool.batch(args.microbatch, args.seq, device, data_rng)
            _, loss = train_model(x, y)
            if loss is None:
                raise RuntimeError("training loss missing")
            (loss / grad_accum).backward()
            acc += loss.detach().float()
            micros += 1
            processed += x.numel()
            if processed >= args.target_tokens:
                break
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0 or processed >= args.target_tokens:
            torch.cuda.synchronize()
            now = time.perf_counter()
            last_loss = float(acc / max(1, micros))
            last_grad = float(grad)
            if not math.isfinite(last_loss) or not math.isfinite(last_grad):
                raise RuntimeError(f"non-finite training state loss={last_loss} grad={last_grad}")
            log({
                "event": "train", "step": step, "processed_tokens": processed,
                "loss": last_loss, "grad_norm": last_grad, "lr": lr,
                "tokens_per_second": (processed - last_tokens) / max(1e-9, now - last_time),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            })
            last_time, last_tokens = now, processed

        if step % args.eval_every == 0 or processed >= args.target_tokens:
            val = evaluate(model, valid_pool, seq=args.seq, batch=1, batches=args.eval_batches,
                           device=device, seed=args.seed + 999_999)
            best_val = min(best_val, val)
            row = {"event": "validation", "step": step, "processed_tokens": processed,
                   "validation_loss": val, "best_validation_loss": best_val,
                   "perplexity": math.exp(min(val, 20.0))}
            if args.bytes_per_token > 0:
                row["bpb_estimate"] = val / math.log(2.0) / args.bytes_per_token
                row["bpb_note"] = "estimated from externally supplied corpus bytes/token"
            log(row)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    summary = {
        "candidate": candidate,
        "config": cfg.to_dict(),
        "processed_tokens": processed,
        "steps": step,
        "seq": args.seq,
        "microbatch": args.microbatch,
        "grad_accum": grad_accum,
        "tokens_per_update": args.tokens_per_update,
        "checkpoint_every": args.checkpoint_every,
        "compile_mode": args.compile_mode,
        "best_validation_loss": best_val,
        "elapsed_s": elapsed,
        "average_tokens_per_second": processed / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    log({"event": "summary", **summary})
    del train_model, opt, model
    torch.cuda.empty_cache(); gc.collect()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--candidates", nargs="*", default=list(PHASE_C))
    ap.add_argument("--out", default="frontier600_report/phase_d_5m")
    ap.add_argument("--target-tokens", type=int, default=5_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--tokens-per-update", type=int, default=16_384)
    ap.add_argument("--checkpoint-every", type=int, default=2)
    ap.add_argument("--compile-mode", choices=("none", "default", "max-autotune"), default="default")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-tokens", type=int, default=500_000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=600_001)
    ap.add_argument("--bytes-per-token", type=float, default=0.0)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")
    if args.target_tokens <= 0 or args.warmup_tokens <= 0:
        raise ValueError("positive token budgets required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_pool = MMapPool(Path(args.train_data))
    valid_pool = MMapPool(Path(args.valid_data))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for candidate in args.candidates:
        if candidate not in CANDIDATES:
            raise KeyError(candidate)
        summaries.append(run_candidate(candidate, args, train_pool, valid_pool, out))
    (out / "all_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
