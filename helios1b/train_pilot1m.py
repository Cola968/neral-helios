from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_torch_save,
    capture_rng_state,
)
from .config import HeliosConfig
from .data import (
    DeterministicTokenBatcher,
    fixed_eval_batches,
    fingerprint_token_file,
)
from .optimizer_gate import optimizer_state_summary, require_bf16_adam_moments
from .smoke_1b import construct_bf16_cuda_model, gib


@torch.inference_mode()
def evaluate(
    model,
    val_path: Path,
    *,
    seq_len: int,
    batches: int,
    seed: int,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    for x, y in fixed_eval_batches(
        val_path,
        seq_len=seq_len,
        batch_size=1,
        batches=batches,
        seed=seed,
        device=torch.device("cuda"),
        dtype="uint16",
    ):
        out = model(x, y)
        if out.cross_entropy is None or not torch.isfinite(out.cross_entropy):
            raise FloatingPointError("non-finite pilot validation CE")
        total += float(out.cross_entropy) * y.numel()
        tokens += y.numel()
    model.train(was_training)
    return total / tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--peak-lr", type=float, default=2e-4)
    ap.add_argument("--warmup-updates", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=96815)
    ap.add_argument("--min-relative-val-improvement", type=float, default=0.05)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("pilot requires CUDA BF16")
    if min(
        args.target_tokens,
        args.seq,
        args.grad_accum,
        args.eval_every,
        args.eval_batches,
    ) <= 0:
        raise ValueError("invalid pilot budget")
    if args.seq > HeliosConfig().max_seq_len:
        raise ValueError("pilot seq exceeds Helios context")

    root = Path(args.data)
    train_path = root / "train.bin"
    val_path = root / "val.bin"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_fp = fingerprint_token_file(train_path, "uint16")
    val_fp = fingerprint_token_file(val_path, "uint16")
    if train_fp.sha256 != manifest["train_sha256"]:
        raise ValueError("pilot train data hash mismatch")
    if val_fp.sha256 != manifest["val_sha256"]:
        raise ValueError("pilot val data hash mismatch")

    tokens_per_update = args.seq * args.grad_accum
    updates = math.ceil(args.target_tokens / tokens_per_update)
    actual_tokens = updates * tokens_per_update
    required_blocks = updates * args.grad_accum
    available_blocks = (train_fp.tokens - 1) // args.seq
    if required_blocks > available_blocks:
        raise ValueError(
            f"pilot would replay train blocks: need {required_blocks}, "
            f"have {available_blocks}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    cfg = HeliosConfig()
    model = construct_bf16_cuda_model(cfg)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    batcher = DeterministicTokenBatcher(
        train_path,
        seq_len=args.seq,
        batch_size=1,
        seed=args.seed + 101,
        dtype="uint16",
        vocab_size=cfg.vocab_size,
    )

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    initial_ce = evaluate(
        model,
        val_path,
        seq_len=args.seq,
        batches=args.eval_batches,
        seed=args.seed + 202,
    )
    best_ce = initial_ce
    metrics_path = out / "metrics.jsonl"
    started = time.perf_counter()
    measured_train_seconds = 0.0
    peak_reserved = 0
    loss_ema = None

    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        if update <= args.warmup_updates:
            lr = args.peak_lr * update / args.warmup_updates
        else:
            lr = args.peak_lr
        for group in optimizer.param_groups:
            group["lr"] = lr

        update_loss = 0.0
        tick = time.perf_counter()
        for _ in range(args.grad_accum):
            x, y = batcher.next_batch(torch.device("cuda"))
            result = model(x, y)
            if result.loss is None or not torch.isfinite(result.loss):
                raise FloatingPointError("non-finite pilot train loss")
            (result.loss / args.grad_accum).backward()
            update_loss += float(result.loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - tick
        measured_train_seconds += dt
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        mean_loss = update_loss / args.grad_accum
        loss_ema = mean_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * mean_loss

        row = {
            "update": update,
            "tokens_seen": update * tokens_per_update,
            "train_loss": mean_loss,
            "train_loss_ema": loss_ema,
            "grad_norm": float(grad_norm),
            "lr": lr,
            "update_seconds": dt,
            "tokens_per_second": tokens_per_update / max(dt, 1e-9),
        }

        if update % args.eval_every == 0 or update == updates:
            val_ce = evaluate(
                model,
                val_path,
                seq_len=args.seq,
                batches=args.eval_batches,
                seed=args.seed + 202,
            )
            best_ce = min(best_ce, val_ce)
            row["val_ce"] = val_ce
            row["best_val_ce"] = best_ce
            print(json.dumps(row, sort_keys=True), flush=True)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    state_summary = optimizer_state_summary(optimizer)
    require_bf16_adam_moments(state_summary)
    final_ce = evaluate(
        model,
        val_path,
        seq_len=args.seq,
        batches=args.eval_batches,
        seed=args.seed + 202,
    )
    relative_improvement = (initial_ce - best_ce) / initial_ce

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "neral-helios-1b-realdata-pilot-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "rng": capture_rng_state(),
        "batcher": batcher.state_dict(),
        "completed_updates": updates,
        "tokens_seen": actual_tokens,
        "pilot_schedule": {
            "seq": args.seq,
            "grad_accum": args.grad_accum,
            "peak_lr": args.peak_lr,
            "warmup_updates": args.warmup_updates,
            "target_tokens": args.target_tokens,
            "actual_tokens": actual_tokens,
        },
        "data_manifest": manifest,
        "initial_val_ce": initial_ce,
        "best_val_ce": best_ce,
        "final_val_ce": final_ce,
    }
    checkpoint_sha = atomic_torch_save(checkpoint, out / "latest.pt")

    passed = (
        math.isfinite(final_ce)
        and best_ce < initial_ce
        and relative_improvement >= args.min_relative_val_improvement
        and batcher.epoch == 0
    )
    summary = {
        "schema": 1,
        "pilot": "helios1b-realdata-1m-v1",
        "parameters": model.parameter_report()["actual"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "target_tokens": args.target_tokens,
        "actual_tokens": actual_tokens,
        "updates": updates,
        "seq": args.seq,
        "grad_accum": args.grad_accum,
        "train_block_replay": batcher.epoch != 0,
        "initial_val_ce": initial_ce,
        "final_val_ce": final_ce,
        "best_val_ce": best_ce,
        "relative_val_improvement": relative_improvement,
        "minimum_relative_val_improvement": args.min_relative_val_improvement,
        "optimizer_state": state_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_train_seconds": measured_train_seconds,
        "optimizer_tokens_per_second": actual_tokens
        / max(measured_train_seconds, 1e-9),
        "peak_reserved_gib": gib(peak_reserved),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_local_only": True,
        "pass": passed,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("Helios 1M real-data pilot gate failed")


if __name__ == "__main__":
    main()
