from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
)
from .config import HeliosConfig
from .data import (
    DeterministicTokenBatcher,
    fingerprint_token_file,
    fixed_eval_batches,
)
from .model import HeliosForCausalLM


@dataclass(frozen=True)
class ProxyTrainingConfig:
    train_bin: str
    val_bin: str
    out_dir: str
    total_steps: int = 100
    stop_after: int | None = None
    resume: str | None = None
    seq_len: int = 64
    batch_size: int = 8
    grad_accum: int = 1
    eval_every: int = 10
    eval_batches: int = 4
    save_every: int = 10
    seed: int = 9681
    peak_lr: float = 8e-4
    min_lr: float = 8e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    min_delta: float = 0.0
    dtype: str = "uint16"
    device: str = "auto"
    fused_adamw: bool = False

    def validate(self) -> None:
        positive = {
            "total_steps": self.total_steps,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "eval_every": self.eval_every,
            "eval_batches": self.eval_batches,
            "save_every": self.save_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.stop_after is not None and self.stop_after <= 0:
            raise ValueError("stop_after must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup < total_steps")
        if not 0 < self.min_lr <= self.peak_lr:
            raise ValueError("learning rates must satisfy 0 < min_lr <= peak_lr")
        if self.weight_decay < 0 or self.grad_clip <= 0 or self.min_delta < 0:
            raise ValueError("invalid optimizer/selection settings")


class WarmupCosineScheduler:
    VERSION = 1

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        peak_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.completed_steps = 0

    def lr_for_next_update(self) -> float:
        i = self.completed_steps
        if i < self.warmup_steps:
            return self.peak_lr * (i + 1) / max(1, self.warmup_steps)
        if self.total_steps - self.warmup_steps <= 1:
            return self.min_lr
        progress = (i - self.warmup_steps) / (
            self.total_steps - self.warmup_steps - 1
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def apply_for_next_update(self) -> float:
        lr = self.lr_for_next_update()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self) -> None:
        if self.completed_steps >= self.total_steps:
            raise RuntimeError("scheduler stepped past total_steps")
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "version": self.VERSION,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"scheduler resume mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}"
                )
        completed = int(state["completed_steps"])
        if not 0 <= completed <= self.total_steps:
            raise ValueError("invalid scheduler completed_steps")
        self.completed_steps = completed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def autocast_factory(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16), "bf16"
    return lambda: nullcontext(), "fp32"


def _fingerprint_contract(path: str, dtype: str) -> dict:
    row = fingerprint_token_file(path, dtype).to_dict()
    row.pop("path", None)
    return row


def build_contract(
    train_cfg: ProxyTrainingConfig,
    model_cfg: HeliosConfig,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "model": model_cfg.to_dict(),
        "train_data": _fingerprint_contract(train_cfg.train_bin, train_cfg.dtype),
        "val_data": _fingerprint_contract(train_cfg.val_bin, train_cfg.dtype),
        "total_steps": train_cfg.total_steps,
        "seq_len": train_cfg.seq_len,
        "batch_size": train_cfg.batch_size,
        "grad_accum": train_cfg.grad_accum,
        "eval_every": train_cfg.eval_every,
        "eval_batches": train_cfg.eval_batches,
        "save_every": train_cfg.save_every,
        "seed": train_cfg.seed,
        "peak_lr": train_cfg.peak_lr,
        "min_lr": train_cfg.min_lr,
        "warmup_steps": train_cfg.warmup_steps,
        "weight_decay": train_cfg.weight_decay,
        "grad_clip": train_cfg.grad_clip,
        "min_delta": train_cfg.min_delta,
        "dtype": train_cfg.dtype,
        "optimizer": "AdamW",
        "fused_adamw": bool(train_cfg.fused_adamw and device.type == "cuda"),
        "device_type": device.type,
        "precision": precision,
        "torch_version": str(torch.__version__),
    }


def _same_contract(expected: dict, actual: dict) -> None:
    if expected == actual:
        return
    keys = sorted(set(expected) | set(actual))
    changed = [
        key
        for key in keys
        if expected.get(key) != actual.get(key)
    ]
    raise ValueError(
        "resume contract mismatch: " + ", ".join(changed[:20])
    )


def _truncate_metrics(path: Path, completed_step: int) -> None:
    if not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("step", -1)) <= completed_step:
            kept.append(json.dumps(row, sort_keys=True))
    path.write_text(
        ("\n".join(kept) + "\n") if kept else "",
        encoding="utf-8",
    )


@torch.inference_mode()
def evaluate_ce(
    model: HeliosForCausalLM,
    *,
    val_bin: str,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    dtype: str,
    autocast_ctx,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for x, y in fixed_eval_batches(
        val_bin,
        seq_len=seq_len,
        batch_size=batch_size,
        batches=batches,
        seed=seed,
        device=device,
        dtype=dtype,
    ):
        with autocast_ctx():
            out = model(x, y)
        if out.cross_entropy is None or not torch.isfinite(out.cross_entropy):
            raise FloatingPointError("non-finite validation cross entropy")
        tokens = y.numel()
        total += float(out.cross_entropy) * tokens
        count += tokens
    model.train(was_training)
    return total / max(1, count)


def _all_finite(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def run_training(
    train_cfg: ProxyTrainingConfig,
    model_cfg: HeliosConfig | None = None,
) -> dict[str, Any]:
    train_cfg.validate()
    model_cfg = model_cfg or HeliosConfig.tiny_proxy()
    if train_cfg.seq_len > model_cfg.max_seq_len:
        raise ValueError("seq_len exceeds model max_seq_len")

    device = resolve_device(train_cfg.device)
    autocast_ctx, precision = autocast_factory(device)
    torch.set_float32_matmul_precision("high")
    seed_everything(train_cfg.seed)

    out = Path(train_cfg.out_dir)
    if out.exists() and any(out.iterdir()) and not train_cfg.resume:
        raise FileExistsError(
            f"fresh run requires an empty output directory: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    contract = build_contract(
        train_cfg,
        model_cfg,
        device=device,
        precision=precision,
    )

    model = HeliosForCausalLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
        fused=bool(train_cfg.fused_adamw and device.type == "cuda"),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=train_cfg.total_steps,
        warmup_steps=train_cfg.warmup_steps,
        peak_lr=train_cfg.peak_lr,
        min_lr=train_cfg.min_lr,
    )
    batcher = DeterministicTokenBatcher(
        train_cfg.train_bin,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed + 101,
        dtype=train_cfg.dtype,
        vocab_size=model_cfg.vocab_size,
    )

    completed_step = 0
    tokens_seen = 0
    best_val_ce = float("inf")
    initial_val_ce: float | None = None
    safe_to_checkpoint = True

    if train_cfg.resume:
        payload = load_checkpoint(train_cfg.resume, map_location="cpu")
        _same_contract(contract, payload["contract"])
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        batcher.load_state_dict(payload["batcher"])
        completed_step = int(payload["completed_step"])
        tokens_seen = int(payload["tokens_seen"])
        best_val_ce = float(payload["best_val_ce"])
        initial_val_ce = float(payload["initial_val_ce"])
        restore_rng_state(payload["rng"])
        if scheduler.completed_steps != completed_step:
            raise ValueError("scheduler/checkpoint step mismatch")
        _truncate_metrics(metrics_path, completed_step)
    else:
        initial_val_ce = evaluate_ce(
            model,
            val_bin=train_cfg.val_bin,
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            batches=train_cfg.eval_batches,
            seed=train_cfg.seed + 202,
            device=device,
            dtype=train_cfg.dtype,
            autocast_ctx=autocast_ctx,
        )
        best_val_ce = initial_val_ce

    if not _all_finite(model):
        raise FloatingPointError("model contains non-finite parameters")

    def checkpoint_payload() -> dict[str, Any]:
        if not safe_to_checkpoint:
            raise RuntimeError(
                "refusing to checkpoint in the middle of an optimizer update"
            )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "neral-helios-proxy",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "batcher": batcher.state_dict(),
            "rng": capture_rng_state(),
            "contract": contract,
            "model_config": model_cfg.to_dict(),
            "completed_step": completed_step,
            "tokens_seen": tokens_seen,
            "best_val_ce": best_val_ce,
            "initial_val_ce": initial_val_ce,
        }

    run_meta = {
        "kind": "Neral Helios proxy training",
        "model_parameters": model.parameter_report()["actual"],
        "model_config": model_cfg.to_dict(),
        "training": asdict(train_cfg),
        "contract": contract,
        "checkpoint_semantics": (
            "Exact resume at completed optimizer-update boundaries. "
            "A crash mid-update replays that update from the previous atomic checkpoint."
        ),
    }
    run_meta["training"]["resume"] = None
    if not train_cfg.resume:
        (out / "run.json").write_text(
            json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        atomic_torch_save(checkpoint_payload(), out / "latest.pt")
        atomic_torch_save(checkpoint_payload(), out / "best.pt")

    stop_step = train_cfg.total_steps
    if train_cfg.stop_after is not None:
        stop_step = min(
            train_cfg.total_steps,
            completed_step + train_cfg.stop_after,
        )

    started = time.perf_counter()
    stop_reason = "target_reached"
    model.train()
    try:
        while completed_step < stop_step:
            safe_to_checkpoint = False
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.apply_for_next_update()
            loss_sum = 0.0
            ce_sum = 0.0
            update_tokens = 0
            update_started = time.perf_counter()

            for _ in range(train_cfg.grad_accum):
                x, y = batcher.next_batch(device)
                with autocast_ctx():
                    result = model(x, y)
                    if result.loss is None or result.cross_entropy is None:
                        raise RuntimeError("training forward did not return loss")
                    scaled = result.loss / train_cfg.grad_accum

                if not torch.isfinite(result.loss):
                    raise FloatingPointError("non-finite training loss")
                scaled.backward()
                loss_sum += float(result.loss.detach())
                ce_sum += float(result.cross_entropy.detach())
                update_tokens += y.numel()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg.grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            completed_step += 1
            tokens_seen += update_tokens
            safe_to_checkpoint = True

            if device.type == "cuda":
                torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            row: dict[str, Any] = {
                "step": completed_step,
                "tokens_seen": tokens_seen,
                "train_loss": loss_sum / train_cfg.grad_accum,
                "train_ce": ce_sum / train_cfg.grad_accum,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "seconds": update_seconds,
                "tokens_per_second": update_tokens / max(update_seconds, 1e-9),
            }

            eval_due = (
                completed_step % train_cfg.eval_every == 0
                or completed_step == stop_step
                or completed_step == train_cfg.total_steps
            )
            if eval_due:
                val_ce = evaluate_ce(
                    model,
                    val_bin=train_cfg.val_bin,
                    seq_len=train_cfg.seq_len,
                    batch_size=train_cfg.batch_size,
                    batches=train_cfg.eval_batches,
                    seed=train_cfg.seed + 202,
                    device=device,
                    dtype=train_cfg.dtype,
                    autocast_ctx=autocast_ctx,
                )
                improved = val_ce < best_val_ce - train_cfg.min_delta
                if improved:
                    best_val_ce = val_ce
                row.update(
                    {
                        "val_ce": val_ce,
                        "val_perplexity": math.exp(min(20.0, val_ce)),
                        "selected": improved,
                    }
                )
                if improved:
                    atomic_torch_save(checkpoint_payload(), out / "best.pt")

            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

            if (
                completed_step % train_cfg.save_every == 0
                or eval_due
                or completed_step == stop_step
            ):
                atomic_torch_save(checkpoint_payload(), out / "latest.pt")

            if not _all_finite(model):
                raise FloatingPointError(
                    "non-finite model parameter after optimizer update"
                )

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        if safe_to_checkpoint:
            atomic_torch_save(checkpoint_payload(), out / "latest.pt")
    except Exception:
        stop_reason = "failed"
        if safe_to_checkpoint:
            atomic_torch_save(checkpoint_payload(), out / "latest.pt")
        raise
    finally:
        elapsed = time.perf_counter() - started

    if completed_step < train_cfg.total_steps and stop_reason == "target_reached":
        stop_reason = "segment_complete"

    summary = {
        "stop_reason": stop_reason,
        "completed_step": completed_step,
        "total_steps": train_cfg.total_steps,
        "tokens_seen": tokens_seen,
        "initial_val_ce": initial_val_ce,
        "best_val_ce": best_val_ce,
        "relative_val_improvement": (
            (initial_val_ce - best_val_ce) / initial_val_ce
            if initial_val_ce and math.isfinite(initial_val_ce)
            else None
        ),
        "elapsed_seconds_this_invocation": elapsed,
        "latest_checkpoint": str((out / "latest.pt").resolve()),
        "best_checkpoint": str((out / "best.pt").resolve()),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def cli_model_config(args) -> HeliosConfig:
    cfg = HeliosConfig.tiny_proxy()
    if args.vocab_size is not None:
        cfg = replace(cfg, vocab_size=args.vocab_size)
    if args.nope_every:
        cfg = cfg.with_nope_ablation(args.nope_every)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Guarded resume-exact trainer for Helios architecture proxies."
    )
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--total-steps", type=int, default=100)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=9681)
    ap.add_argument("--peak-lr", type=float, default=8e-4)
    ap.add_argument("--min-lr", type=float, default=8e-5)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fused-adamw", action="store_true")
    ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--nope-every", type=int, default=0)
    args = ap.parse_args()

    cfg = ProxyTrainingConfig(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        out_dir=args.out,
        total_steps=args.total_steps,
        stop_after=args.stop_after,
        resume=args.resume,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        seed=args.seed,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        min_delta=args.min_delta,
        dtype=args.dtype,
        device=args.device,
        fused_adamw=args.fused_adamw,
    )
    summary = run_training(cfg, cli_model_config(args))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
