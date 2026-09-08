"""From-scratch Neral-100M Helix pretraining with RTX-4070 fallback and live telemetry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from architectures100_variants import EXPECTED, VARIANTS
from model100 import Neral100M
from neral100_checkpoint import (
    checkpoint_payload,
    make_training_contract,
    validate_checkpoint,
)
from training_observability import TrainingObserver


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stream(stream: np.memmap, vocab_size: int, label: str, chunk_tokens: int = 16_000_000) -> dict:
    if len(stream) < 1024:
        raise ValueError(f"{label} stream too short")
    maximum = 0
    for start in range(0, len(stream), chunk_tokens):
        block = np.asarray(stream[start:start + chunk_tokens])
        if block.size:
            maximum = max(maximum, int(block.max()))
    if maximum >= vocab_size:
        raise ValueError(f"{label} token id {maximum} exceeds vocab {vocab_size}")
    return {"tokens": int(len(stream)), "max_token_id": maximum, "dtype": str(stream.dtype)}


def batch_from_stream(stream, batch: int, seq: int, device, rng):
    if len(stream) <= seq + 1:
        raise ValueError("stream too short")
    starts = rng.integers(0, len(stream) - seq - 1, size=batch)
    x = np.stack([np.asarray(stream[i:i + seq], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(stream[i + 1:i + seq + 1], dtype=np.int64) for i in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def train_once(model, optimizer, stream, seq, microbatch, grad_accum, device, rng, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    tokens = 0
    for _ in range(grad_accum):
        x, y = batch_from_stream(stream, microbatch, seq, device, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        (loss / grad_accum).backward()
        total += float(loss.detach())
        tokens += x.numel()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return total / grad_accum, tokens, float(grad_norm.detach())


@torch.no_grad()
def evaluate(model, stream, seq, batch, batches, device, rng):
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            x, y = batch_from_stream(stream, batch, seq, device, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            losses.append(float(loss))
    finally:
        model.train()
    return float(sum(losses) / len(losses))


def fallback_shapes(seq: int, micro: int):
    seen = set()
    while (seq, micro) not in seen:
        seen.add((seq, micro))
        yield seq, micro
        if micro > 1:
            micro = max(1, micro // 2)
        elif seq > 512:
            seq = max(512, seq // 2)
        else:
            break


def lr_for_step(step: int, total_steps: int, peak: float, floor: float, warmup_fraction: float) -> float:
    warmup = max(1, int(total_steps * warmup_fraction))
    if step <= warmup:
        return peak * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return floor + .5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def capture_rng(rng) -> dict:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": rng.bit_generator.state,
    }


def restore_rng(payload: dict, rng) -> None:
    state = payload.get("rng_state") or {}
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        rng.bit_generator.state = state["numpy"]


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="a1")
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin")
    ap.add_argument("--train-manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--steps", type=int, default=1000, help="Total target optimizer steps, including resumed steps")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--warmup-fraction", type=float, default=.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=100968)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    if min(args.steps, args.seq, args.microbatch, args.grad_accum, args.eval_every, args.eval_batches, args.save_every, args.log_every) <= 0:
        raise ValueError("positive budgets required")
    if not (0 <= args.min_lr <= args.lr) or not 0 <= args.warmup_fraction < 1 or args.grad_clip <= 0:
        raise ValueError("invalid optimizer schedule")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)

    train_path = Path(args.train_bin).resolve()
    stream = np.memmap(train_path, dtype="<u2", mode="r")
    train_info = validate_stream(stream, VARIANTS[args.variant].vocab_size, "train")
    val_stream = None
    val_info = None
    if args.val_bin:
        val_stream = np.memmap(Path(args.val_bin).resolve(), dtype="<u2", mode="r")
        val_info = validate_stream(val_stream, VARIANTS[args.variant].vocab_size, "validation")

    model = Neral100M(VARIANTS[args.variant]).to(device)
    model.set_gradient_checkpointing(True)
    if sum(p.numel() for p in model.parameters()) != EXPECTED[args.variant]:
        raise ValueError("parameter drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.1, fused=True)

    contract = make_training_contract(
        args.variant,
        seq=args.seq,
        microbatch=args.microbatch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_fraction=args.warmup_fraction,
        grad_clip=args.grad_clip,
    )
    contract["total_steps"] = int(args.steps)
    contract["eval_every"] = int(args.eval_every)
    contract["eval_batches"] = int(args.eval_batches)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    observer = TrainingObserver(
        out,
        run_name=f"neral100-{args.variant}-{out.name}",
        group="100m-pretraining",
        config={
            "variant": args.variant,
            "parameters": EXPECTED[args.variant],
            "steps": args.steps,
            "seq": args.seq,
            "microbatch": args.microbatch,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_fraction": args.warmup_fraction,
            "validation_enabled": val_stream is not None,
        },
    )

    tokens = 0
    start_step = 0
    effective_seq, effective_micro = args.seq, args.microbatch
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        meta = validate_checkpoint(resume, require_optimizer=True, require_training_contract=True)
        if meta["variant"] != args.variant or resume["training_contract"] != contract:
            raise ValueError("resume training contract mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_step = meta["step"]
        tokens = meta["tokens"]
        restore_rng(resume, rng)
        effective = (resume.get("provenance") or {}).get("effective_shape")
        if isinstance(effective, (list, tuple)) and len(effective) == 2:
            effective_seq, effective_micro = int(effective[0]), int(effective[1])
        observer.alert("Resume accepted", f"Resuming at step {start_step} and {tokens} tokens", "INFO", start_step)
    if start_step >= args.steps:
        raise ValueError("resume checkpoint already reached target steps")

    manifest_sha = sha256_file(Path(args.train_manifest)) if args.train_manifest else None
    provenance_base = {
        "from_scratch": not bool(args.resume),
        "resume": bool(args.resume),
        "oom_fallback": True,
        "train_stream_tokens": train_info["tokens"],
        "train_stream_max_token_id": train_info["max_token_id"],
        "train_manifest_sha256": manifest_sha,
        "validation": val_info,
    }

    loss_ema = None
    throughput_ema = None
    status = "failed"
    try:
        for step in range(start_step + 1, args.steps + 1):
            lr = lr_for_step(step, args.steps, args.lr, args.min_lr, args.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = lr

            success = False
            requested_shape = (effective_seq, effective_micro)
            for seq, micro in fallback_shapes(effective_seq, effective_micro):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    loss, n, grad_norm = train_once(
                        model, optimizer, stream, seq, micro, args.grad_accum, device, rng, args.grad_clip
                    )
                    elapsed = max(1e-9, time.perf_counter() - t0)
                    tokens += n
                    if (seq, micro) != requested_shape:
                        observer.alert(
                            "CUDA OOM fallback",
                            f"Shape changed from {requested_shape} to {(seq, micro)} and will persist",
                            "WARN",
                            step,
                        )
                    effective_seq, effective_micro = seq, micro
                    success = True
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            if not success:
                observer.alert("Fatal CUDA OOM", "OOM even at seq=512 microbatch=1", "ERROR", step)
                raise RuntimeError("100M OOM even at seq=512 microbatch=1")

            throughput = n / elapsed
            loss_ema = loss if loss_ema is None else .95 * loss_ema + .05 * loss
            throughput_ema = throughput if throughput_ema is None else .95 * throughput_ema + .05 * throughput
            metrics = {
                "train_loss": loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "tokens_total": tokens,
                "tokens_per_second": throughput,
                "seq": effective_seq,
                "microbatch": effective_micro,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }

            if loss_ema is not None and step > max(20, start_step + 20) and loss > 2.5 * loss_ema:
                observer.alert("Loss spike", f"loss={loss:.4f}, ema={loss_ema:.4f}", "WARN", step)
            if throughput_ema is not None and step > max(20, start_step + 20) and throughput < .5 * throughput_ema:
                observer.alert("Throughput collapse", f"tok/s={throughput:.1f}, ema={throughput_ema:.1f}", "WARN", step)

            if val_stream is not None and (step % args.eval_every == 0 or step == args.steps):
                eval_seq = min(effective_seq, 2048)
                val_loss = evaluate(model, val_stream, eval_seq, 1, args.eval_batches, device, eval_rng)
                metrics["val_loss"] = val_loss

            if step % args.log_every == 0 or step == 1 or step == args.steps or "val_loss" in metrics:
                observer.log(step, metrics)
                print(json.dumps({"step": step, **metrics}), flush=True)

            if step % args.save_every == 0 or step == args.steps:
                provenance = {**provenance_base, "effective_shape": [effective_seq, effective_micro]}
                payload = checkpoint_payload(
                    model,
                    args.variant,
                    tokens=tokens,
                    step=step,
                    optimizer_state=optimizer.state_dict(),
                    training_contract=contract,
                    rng_state=capture_rng(rng),
                    provenance=provenance,
                )
                atomic_torch_save(payload, out / "latest.pt")

        summary = {
            "variant": args.variant,
            "parameters": EXPECTED[args.variant],
            "steps": args.steps,
            "tokens": tokens,
            "effective_shape": [effective_seq, effective_micro],
            "from_scratch": not bool(args.resume),
            "resume": bool(args.resume),
            "foreign_weights": False,
            "training_contract": contract,
            "train_stream": train_info,
            "validation_stream": val_info,
            "trackio_remote_enabled": observer.remote_enabled,
            "telemetry_file": "telemetry.jsonl",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
        status = "completed"
    except Exception as exc:
        observer.alert("Training failed", f"{type(exc).__name__}: {str(exc)[:400]}", "ERROR")
        raise
    finally:
        observer.finish(status)


if __name__ == "__main__":
    main()
