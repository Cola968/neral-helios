from __future__ import annotations

"""Resumable Neral 350M Frontier trainer.

Design goals:
- local/self-hosted RTX training
- exact token accounting against the full 10B schedule
- ability to pause after a bounded burn-in and resume the SAME optimizer/model run
- BF16 + fused AdamW + cosine decay + warmup
- optional activation checkpointing
- atomic checkpoints with optimizer plus global and train-sampler RNG state
- deterministic internal validation sampling for comparable learning curves
- JSONL + stdout live metrics for GitHub/Trackio bridges
- periodic validation and fail-fast numerical checks

Dataset input is a directory containing uint16 token shards (*.bin). Token IDs must fit
the locked 16K Neral BPE vocabulary. Frozen benchmark holdouts must never be present.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from model350 import Neral350M, FRONTIER_350


CHECKPOINT_SCHEMA = 2


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def list_shards(root: Path):
    shards = sorted(root.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"no *.bin shards in {root}")
    return shards


def sample_batch(shards, batch, seq, device, rng, vocab_size):
    rows = []
    targets = []
    for _ in range(batch):
        p = rng.choice(shards)
        a = np.memmap(p, dtype=np.uint16, mode="r")
        if len(a) < seq + 1:
            raise ValueError(f"shard too short: {p}")
        # There are len(a) - seq valid windows of length seq+1. The previous
        # upper bound excluded the last valid window and failed when len(a)==seq+1.
        start = rng.randrange(0, len(a) - seq)
        chunk = np.asarray(a[start:start + seq + 1], dtype=np.int64)
        if chunk.max(initial=0) >= vocab_size:
            raise ValueError(f"token id outside vocabulary in {p}")
        rows.append(torch.from_numpy(chunk[:-1].copy()))
        targets.append(torch.from_numpy(chunk[1:].copy()))
    return (
        torch.stack(rows).to(device, non_blocking=True),
        torch.stack(targets).to(device, non_blocking=True),
    )


def cosine_lr(step, warmup, total, peak, min_ratio=0.08):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    u = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * u)))


def evaluate(model, shards, seq, batch, batches, device, seed):
    rng = random.Random(seed)
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(shards, batch, seq, device, rng, model.cfg.vocab_size)
            _, loss = model(x, y)
            vals.append(float(loss))
    model.train()
    return sum(vals) / len(vals)


def checkpoint_state(
    model,
    opt,
    step,
    processed,
    best,
    target_tokens,
    tokenizer_sha256,
    train_sampler_rng: random.Random | None = None,
    validation_identity: str = "",
):
    state = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "processed_tokens": processed,
        "best_validation": best,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "config": FRONTIER_350.to_dict(),
        "target_tokens": target_tokens,
        "tokenizer_sha256": tokenizer_sha256,
        "validation_identity": validation_identity,
    }
    if train_sampler_rng is not None:
        state["train_sampler_rng"] = train_sampler_rng.getstate()
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=10_000_000_000,
                    help="full LR schedule target; Frontier release contract is 10B")
    ap.add_argument("--stop-after-tokens", type=int, default=0,
                    help="pause this invocation at this cumulative token count; 0 means target-tokens")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=35001)
    ap.add_argument("--resume", default="")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--tokenizer-sha256", default="")
    args = ap.parse_args()

    if args.target_tokens <= 0:
        raise ValueError("positive target token budget required")
    run_until = args.stop_after_tokens or args.target_tokens
    if run_until <= 0 or run_until > args.target_tokens:
        raise ValueError("stop-after-tokens must be in (0,target-tokens]")
    if args.seq <= 0 or args.microbatch <= 0 or args.grad_accum <= 0:
        raise ValueError("invalid batch dimensions")
    if args.seq > FRONTIER_350.max_seq_len:
        raise ValueError("sequence exceeds architecture context")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.jsonl"
    train = list_shards(Path(args.train_data))
    valid = list_shards(Path(args.valid_data))
    rng = random.Random(args.seed + 1)
    sampler_resume_source = "fresh_seed"

    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    report = model.parameter_report()
    if not (345_000_000 <= report["total"] <= 360_000_000):
        raise RuntimeError(f"Frontier parameter budget violated: {report}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay, fused=True,
    )

    tokens_per_update = args.seq * args.microbatch * args.grad_accum
    total_updates = math.ceil(args.target_tokens / tokens_per_update)
    warmup = max(1, int(total_updates * args.warmup_frac))
    step = 0
    processed = 0
    best = float("inf")

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck.get("config") != FRONTIER_350.to_dict():
            raise RuntimeError("resume architecture differs from locked Frontier 350M config")
        if int(ck.get("target_tokens", -1)) != args.target_tokens:
            raise RuntimeError("resume target token schedule differs")
        if args.tokenizer_sha256 and ck.get("tokenizer_sha256") not in ("", args.tokenizer_sha256):
            raise RuntimeError("resume tokenizer digest differs")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        processed = int(ck["processed_tokens"])
        best = float(ck.get("best_validation", best))
        torch.set_rng_state(ck["torch_rng"])
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        if "cuda_rng" in ck:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        if "train_sampler_rng" in ck:
            rng.setstate(ck["train_sampler_rng"])
            sampler_resume_source = "checkpoint"
        else:
            # Legacy checkpoints did not persist the dedicated sampler RNG. Keep
            # compatibility, but make the reset explicit in telemetry rather than
            # silently pretending the exact sample stream was resumed.
            sampler_resume_source = "legacy_seed_reset"
        if processed >= run_until:
            raise RuntimeError(f"resume already at {processed:,} tokens, requested stop is {run_until:,}")

    invocation_start_tokens = processed
    validation_seed = args.seed + 999
    startup = {
        "event": "startup",
        "parameters": report,
        "config": FRONTIER_350.to_dict(),
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "target_tokens": args.target_tokens,
        "run_until_tokens": run_until,
        "processed_tokens": processed,
        "tokens_per_update": tokens_per_update,
        "total_updates": total_updates,
        "warmup_updates": warmup,
        "gradient_checkpointing": args.gradient_checkpointing,
        "tokenizer_sha256": args.tokenizer_sha256,
        "train_sampler_rng_source": sampler_resume_source,
        "validation_seed": validation_seed,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(startup), flush=True)
    append_jsonl(metrics, startup)

    model.train()
    start_time = time.time()
    last_time = start_time
    last_tokens = processed
    last_eval_step = -1

    while processed < run_until:
        lr = cosine_lr(step, warmup, total_updates, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        running = 0.0
        microsteps = 0
        for _ in range(args.grad_accum):
            x, y = sample_batch(train, args.microbatch, args.seq, device, rng, model.cfg.vocab_size)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            (loss / args.grad_accum).backward()
            running += float(loss.detach())
            microsteps += 1
            processed += x.numel()
            if processed >= run_until:
                break
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        opt.step()
        step += 1

        if step % args.log_every == 0 or processed >= run_until:
            now = time.time()
            tps = (processed - last_tokens) / max(1e-9, now - last_time)
            invocation_denom = max(1, run_until - invocation_start_tokens)
            row = {
                "event": "train",
                "step": step,
                "processed_tokens": processed,
                "target_tokens": args.target_tokens,
                "run_until_tokens": run_until,
                "progress_total": processed / args.target_tokens,
                "progress_invocation": (processed - invocation_start_tokens) / invocation_denom,
                "loss": running / max(1, microsteps),
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_second": tps,
                "elapsed_s": now - start_time,
                "gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 2**30,
                "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(row), flush=True)
            append_jsonl(metrics, row)
            last_time = now
            last_tokens = processed

        if step % args.eval_every == 0 or processed >= run_until:
            val = evaluate(model, valid, args.seq, max(1, args.microbatch), args.eval_batches, device, validation_seed)
            ev = {
                "event": "validation",
                "step": step,
                "processed_tokens": processed,
                "validation_loss": val,
                "best_validation_loss": min(best, val),
                "validation_seed": validation_seed,
            }
            print(json.dumps(ev), flush=True)
            append_jsonl(metrics, ev)
            last_eval_step = step
            if val < best:
                best = val
                atomic_save({
                    "model": model.state_dict(),
                    "step": step,
                    "processed_tokens": processed,
                    "validation_loss": val,
                    "config": FRONTIER_350.to_dict(),
                    "tokenizer_sha256": args.tokenizer_sha256,
                }, out / "best_validation.pt")

        if step % args.save_every == 0 or processed >= run_until:
            atomic_save(
                checkpoint_state(
                    model, opt, step, processed, best, args.target_tokens,
                    args.tokenizer_sha256, train_sampler_rng=rng,
                ),
                out / "resume.pt",
            )

    if last_eval_step != step:
        val = evaluate(model, valid, args.seq, max(1, args.microbatch), args.eval_batches, device, validation_seed)
        best = min(best, val)
        ev = {
            "event": "validation", "step": step, "processed_tokens": processed,
            "validation_loss": val, "best_validation_loss": best,
            "validation_seed": validation_seed,
        }
        print(json.dumps(ev), flush=True)
        append_jsonl(metrics, ev)

    atomic_save(
        checkpoint_state(
            model, opt, step, processed, best, args.target_tokens,
            args.tokenizer_sha256, train_sampler_rng=rng,
        ),
        out / "resume.pt",
    )
    complete = processed >= args.target_tokens
    summary = {
        "complete": complete,
        "paused": not complete,
        "parameters": report,
        "processed_tokens": processed,
        "target_tokens": args.target_tokens,
        "run_until_tokens": run_until,
        "steps": step,
        "best_validation_loss": best,
        "elapsed_s": time.time() - start_time,
        "config": FRONTIER_350.to_dict(),
        "tokenizer_sha256": args.tokenizer_sha256,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "train_sampler_rng_source": sampler_resume_source,
        "validation_seed": validation_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
