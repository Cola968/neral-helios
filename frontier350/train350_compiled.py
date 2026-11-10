from __future__ import annotations

"""Checkpoint-native compiled trainer candidate for Neral Arctus 350M.

This keeps the locked unfused Neral350M parameter/state-dict layout so an existing
Frontier checkpoint and AdamW optimizer state can resume without conversion. The only
execution changes are persistent mmap input shards and optional TorchInductor compile.
Promotion to the long run still requires the compile/eager equivalence gate.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model350 import FRONTIER_350, Neral350M
from train350 import CHECKPOINT_SCHEMA, append_jsonl, atomic_save, checkpoint_state, cosine_lr
from recovery_identity import file_sha, object_sha

PARAMETERS = 349_825_152


def update_token_counts(remaining: int, micro_tokens: int, accumulation: int) -> list[int]:
    if min(remaining, micro_tokens, accumulation) <= 0:
        raise ValueError("positive update dimensions required")
    budget = min(remaining, micro_tokens * accumulation)
    return [min(micro_tokens, budget - i) for i in range(0, budget, micro_tokens)]


def mask_budget_targets(targets: torch.Tensor, count: int) -> torch.Tensor:
    if not 0 < count <= targets.numel():
        raise ValueError("invalid target budget")
    if count == targets.numel():
        return targets
    result = targets.contiguous().clone()
    result.view(-1)[count:] = -100
    return result


class PersistentMMapPool:
    """Open uint16 shards once instead of recreating np.memmap objects per sample."""

    def __init__(self, root: Path, vocab_size: int):
        self.paths = sorted(root.glob("*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no *.bin shards in {root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.vocab_size = int(vocab_size)

    def batch(self, batch: int, seq: int, device: torch.device, rng: random.Random):
        rows = np.empty((batch, seq + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(self.arrays)
            if len(a) < seq + 1:
                raise ValueError("shard too short for requested sequence")
            # len(a)-seq valid windows exist for a sequence+target sample.
            start = rng.randrange(0, len(a) - seq)
            rows[i] = a[start:start + seq + 1]
        if rows.max(initial=0) >= self.vocab_size:
            raise ValueError("token id outside locked vocabulary")
        whole = torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)
        return whole[:, :-1], whole[:, 1:]


def evaluate(model: Neral350M, pool: PersistentMMapPool, *, seq: int, batch: int,
             batches: int, device: torch.device, seed: int) -> float:
    rng = random.Random(seed)
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = pool.batch(batch, seq, device, rng)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("validation returned no loss")
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError("non-finite validation loss")
            vals.append(value)
    model.train()
    return sum(vals) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--valid-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--stop-after-tokens", type=int, default=0)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    ap.add_argument("--resume-only", action="store_true", help="Keep resumable weights; omit redundant model-only best snapshots")
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
    ap.add_argument("--compile-mode", choices=("none", "default", "max-autotune"), default="default")
    ap.add_argument("--compile-fullgraph", action="store_true")
    ap.add_argument("--numerical-check-every", type=int, default=1,
                    help="host-side finite check interval in optimizer updates; 1 is safest")
    args = ap.parse_args()

    if args.target_tokens <= 0:
        raise ValueError("positive target token budget required")
    run_until = args.stop_after_tokens or args.target_tokens
    if run_until <= 0 or run_until > args.target_tokens:
        raise ValueError("stop-after-tokens must be in (0,target-tokens]")
    if min(args.seq, args.microbatch, args.grad_accum, args.numerical_check_every,
           args.log_every, args.eval_every, args.save_every, args.eval_batches) <= 0:
        raise ValueError("invalid batch/check dimensions")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    if args.seq > FRONTIER_350.max_seq_len:
        raise ValueError("sequence exceeds architecture context")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and (not args.resume or Path(args.resume).resolve() != (out / "resume.pt").resolve()):
        raise RuntimeError("existing output must be resumed from its own resume.pt")
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.jsonl"
    train_pool = PersistentMMapPool(Path(args.train_data), FRONTIER_350.vocab_size)
    valid_pool = PersistentMMapPool(Path(args.valid_data), FRONTIER_350.vocab_size)
    validation_identity = object_sha({'shards':[(p.name,file_sha(p)) for p in valid_pool.paths],
                                     'seq':args.seq,'batch':args.microbatch,'batches':args.eval_batches,'seed':args.seed+999})
    rng = random.Random(args.seed + 1)
    sampler_resume_source = "fresh_seed"

    # Keep the original model/state-dict layout so resume.pt is directly compatible.
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    report = model.parameter_report()
    if report["total"] != PARAMETERS:
        raise RuntimeError(f"Frontier parameter contract changed: {report}")
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
        if args.tokenizer_sha256 and ck.get("tokenizer_sha256") != args.tokenizer_sha256:
            raise RuntimeError("resume tokenizer digest differs")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"])
        processed = int(ck["processed_tokens"])
        best = float(ck.get("best_validation", best))
        if ck.get('validation_identity') != validation_identity:
            best = evaluate(model, valid_pool, seq=args.seq, batch=args.microbatch,
                            batches=args.eval_batches, device=device, seed=args.seed+999)
        torch.set_rng_state(ck["torch_rng"])
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        if "cuda_rng" in ck:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        if "train_sampler_rng" in ck:
            rng.setstate(ck["train_sampler_rng"])
            sampler_resume_source = "checkpoint"
        else:
            sampler_resume_source = "legacy_seed_reset"
        if processed >= run_until:
            raise RuntimeError(f"resume already at {processed:,} tokens, requested stop is {run_until:,}")

    train_model = model
    if args.compile_mode != "none":
        train_model = torch.compile(
            model,
            backend="inductor",
            mode=args.compile_mode,
            fullgraph=args.compile_fullgraph,
            dynamic=False,
        )

    invocation_start_tokens = processed
    validation_seed = args.seed + 999
    startup = {
        "event": "startup",
        "parameters": report,
        "config": FRONTIER_350.to_dict(),
        "execution_layout": "checkpoint-native-unfused",
        "compile_mode": args.compile_mode,
        "compile_fullgraph": args.compile_fullgraph,
        "persistent_mmap": True,
        "source_checkpoint_sha256": file_sha(args.resume) if args.resume else None,
        "train_shards_sha256": {p.name:file_sha(p) for p in train_pool.paths},
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "target_tokens": args.target_tokens,
        "run_until_tokens": run_until,
        "processed_tokens": processed,
        "tokens_per_update": tokens_per_update,
        "total_updates": total_updates,
        "warmup_updates": warmup,
        "gradient_checkpointing": args.gradient_checkpointing,
        "numerical_check_every": args.numerical_check_every,
        "tokenizer_sha256": args.tokenizer_sha256,
        "train_sampler_rng_source": sampler_resume_source,
        "validation_seed": validation_seed,
        "validation_identity": validation_identity,
        "lr_schedule": args.lr_schedule,
        "lr_peak": args.lr,
        "numerical_checks": "every optimizer update, before mutation",
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(startup), flush=True)
    append_jsonl(metrics, startup)

    model.train()
    start_time = time.time()
    last_time = start_time
    last_tokens = processed
    last_eval_step = -1
    last_loss_value = float("nan")
    last_grad_value = float("nan")

    while processed < run_until:
        lr = args.lr if args.lr_schedule == "constant" else cosine_lr(step, warmup, total_updates, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device, dtype=torch.float32)
        microsteps = 0
        token_counts = update_token_counts(run_until - processed, args.seq * args.microbatch, args.grad_accum)
        update_tokens = sum(token_counts)
        for count in token_counts:
            x, y = train_pool.batch(args.microbatch, args.seq, device, rng)
            y = mask_budget_targets(y, count)
            _, loss = train_model(x, y)
            if loss is None:
                raise RuntimeError(f"no loss at step {step}")
            (loss * (count / update_tokens)).backward()
            loss_acc.add_(loss.detach().float() * (count / update_tokens))
            microsteps += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        last_loss_value = float(loss_acc)
        last_grad_value = float(grad_norm)
        if not math.isfinite(last_loss_value):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.step()
        processed += update_tokens
        step += 1

        if step % args.log_every == 0 or processed >= run_until:
            # Synchronize only at a logging boundary so TPS includes queued GPU work.
            torch.cuda.synchronize()
            now = time.time()
            if not math.isfinite(last_loss_value):
                last_loss_value = float(loss_acc / max(1, microsteps))
            if not math.isfinite(last_grad_value):
                last_grad_value = float(grad_norm)
            invocation_denom = max(1, run_until - invocation_start_tokens)
            row = {
                "event": "train",
                "step": step,
                "processed_tokens": processed,
                "target_tokens": args.target_tokens,
                "run_until_tokens": run_until,
                "progress_total": processed / args.target_tokens,
                "progress_invocation": (processed - invocation_start_tokens) / invocation_denom,
                "loss": last_loss_value,
                "lr": lr,
                "grad_norm": last_grad_value,
                "tokens_per_second": (processed - last_tokens) / max(1e-9, now - last_time),
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
            val = evaluate(
                model, valid_pool, seq=args.seq, batch=max(1, args.microbatch),
                batches=args.eval_batches, device=device, seed=validation_seed,
            )
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
                if not args.resume_only: atomic_save({
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
                    args.tokenizer_sha256, train_sampler_rng=rng, validation_identity=validation_identity,
                ),
                out / "resume.pt",
            )

    if last_eval_step != step:
        val = evaluate(
            model, valid_pool, seq=args.seq, batch=max(1, args.microbatch),
            batches=args.eval_batches, device=device, seed=validation_seed,
        )
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
            args.tokenizer_sha256, train_sampler_rng=rng, validation_identity=validation_identity,
        ),
        out / "resume.pt",
    )
    summary = {
        "complete": processed >= args.target_tokens,
        "paused": processed < args.target_tokens,
        "parameters": report,
        "processed_tokens": processed,
        "target_tokens": args.target_tokens,
        "run_until_tokens": run_until,
        "steps": step,
        "best_validation_loss": best,
        "elapsed_s": time.time() - start_time,
        "config": FRONTIER_350.to_dict(),
        "tokenizer_sha256": args.tokenizer_sha256,
        "compile_mode": args.compile_mode,
        "compile_fullgraph": args.compile_fullgraph,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "train_sampler_rng_source": sampler_resume_source,
        "validation_seed": validation_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
