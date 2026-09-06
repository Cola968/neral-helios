from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from config import NeralConfig
from model import Neral20M


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_memmap(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run prepare_data.py first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size: int, seq_len: int, device: torch.device):
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Dataset has only {len(data)} tokens; need > {seq_len + 1}.")
    starts = np.random.randint(0, max_start, size=batch_size)
    x_np = np.stack([np.asarray(data[s:s+seq_len], dtype=np.int64) for s in starts])
    y_np = np.stack([np.asarray(data[s+1:s+seq_len+1], dtype=np.int64) for s in starts])
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_for_step(step: int, max_steps: int, peak_lr: float, warmup_steps: int, min_lr_ratio: float):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return peak_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)


@torch.no_grad()
def estimate_loss(model, data, batch_size, seq_len, device, eval_batches, autocast_ctx):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data, batch_size, seq_len, device)
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean = float(np.mean(losses))
    return mean, mean / math.log(2.0)  # bits per byte/token for byte-level LM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/neral20m")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8, help="micro-batch")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="Load model weights only; start a fresh optimizer")
    ap.add_argument("--stop-after", type=int, default=None, help="Run at most this many optimizer updates in this invocation")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="Stop after N validation checks without improvement; 0 disables")
    args = ap.parse_args()

    if args.seq_len > 4096:
        raise ValueError("This Neral 20M config uses max_seq_len=4096")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not detected. Training 20M on CPU will be very slow.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_data = load_memmap(Path(args.data) / "train.bin")
    val_data = load_memmap(Path(args.data) / "val.bin")

    cfg = NeralConfig()
    model = Neral20M(cfg).to(device)
    report = model.parameter_report()
    print(f"Neral 20M parameters: {report['total']:,} ({report['millions']:.3f}M)")

    if device.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_dtype = torch.float32
        def autocast_ctx():
            return nullcontext()

    fused_ok = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay, fused=fused_ok,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    start_step = 0
    best_val = float("inf")
    bad_eval_count = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from step {start_step}")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded model weights from {args.init_from}; optimizer starts fresh")

    raw_model = model
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile unavailable; continuing eager: {exc}")

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"device={device} dtype={amp_dtype} tokens/update={tokens_per_step:,}")

    log_path = out / "metrics.jsonl"
    t0 = time.perf_counter()
    end_step = args.steps if args.stop_after is None else min(args.steps, start_step + max(0, args.stop_after))
    for step in range(start_step, end_step):
        lr = lr_for_step(step, args.steps, args.lr, args.warmup, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
            with autocast_ctx():
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            total_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        train_loss = total_loss / args.grad_accum

        if step % 10 == 0:
            tok_s = ((step - start_step + 1) * tokens_per_step) / max(elapsed, 1e-9)
            print(f"step {step:6d} | loss {train_loss:.4f} | lr {lr:.2e} | grad {float(grad_norm):.3f} | {tok_s:,.0f} tok/s")

        record = {"step": step, "train_loss": train_loss, "lr": lr, "grad_norm": float(grad_norm)}

        if step % args.eval_every == 0 or step == end_step - 1:
            val_loss, val_bpb = estimate_loss(raw_model, val_data, args.batch_size, args.seq_len, device, args.eval_batches, autocast_ctx)
            print(f"  VAL loss={val_loss:.4f} | BPB={val_bpb:.4f}")
            record.update({"val_loss": val_loss, "val_bpb": val_bpb})
            if val_loss < best_val:
                best_val = val_loss
                bad_eval_count = 0
                torch.save({
                    "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "best_val": best_val, "config": cfg.to_dict(), "args": vars(args),
                }, out / "best.pt")
            else:
                bad_eval_count += 1

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if step > 0 and (step % args.save_every == 0 or step == end_step - 1):
            torch.save({
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val": best_val, "config": cfg.to_dict(), "args": vars(args),
            }, out / "latest.pt")

        if args.early_stop_patience > 0 and bad_eval_count >= args.early_stop_patience:
            print(f"Early stopping at step {step}: no validation improvement for {bad_eval_count} checks.")
            break

    print(f"Finished. Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
