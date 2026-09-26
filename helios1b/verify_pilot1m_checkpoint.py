from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from .checkpointing import restore_rng_state, sha256_file
from .config import HeliosConfig
from .data import DeterministicTokenBatcher, fixed_eval_batches
from .optimizer_gate import optimizer_state_summary, require_bf16_adam_moments
from .smoke_1b import construct_bf16_cuda_model, gib


@torch.inference_mode()
def evaluate(model, val_path: Path, seq: int, batches: int, seed: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for x, y in fixed_eval_batches(
        val_path,
        seq_len=seq,
        batch_size=1,
        batches=batches,
        seed=seed,
        device=torch.device("cuda"),
        dtype="uint16",
    ):
        out = model(x, y)
        if out.cross_entropy is None or not torch.isfinite(out.cross_entropy):
            raise FloatingPointError("non-finite resumed validation CE")
        total += float(out.cross_entropy) * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=96815)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("checkpoint verification requires CUDA BF16")

    checkpoint_path = Path(args.checkpoint)
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if not checkpoint_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("pilot checkpoint or SHA sidecar missing")

    sidecar_sha = sidecar.read_text(encoding="ascii").strip()
    if sidecar_sha != args.expected_sha256:
        raise ValueError("checkpoint sidecar does not match recorded pilot summary")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != sidecar_sha:
        raise ValueError("checkpoint bytes fail SHA-256 integrity check")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if payload.get("kind") != "neral-helios-1b-realdata-pilot-v1":
        raise ValueError("unexpected checkpoint kind")

    cfg = HeliosConfig(**payload["config"])
    if cfg.parameter_count() != 1_006_740_992:
        raise ValueError("unexpected Helios architecture in checkpoint")

    model = construct_bf16_cuda_model(cfg)
    model.gradient_checkpointing_enable()
    model.load_state_dict(payload["model"], strict=True)
    del payload["model"]
    gc.collect()

    schedule = payload["pilot_schedule"]
    seq = int(schedule["seq"])
    grad_accum = int(schedule["grad_accum"])
    lr = float(schedule["peak_lr"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    optimizer.load_state_dict(payload["optimizer"])
    del payload["optimizer"]
    state_summary = optimizer_state_summary(optimizer)
    require_bf16_adam_moments(state_summary)

    data_root = Path(args.data)
    train_path = data_root / "train.bin"
    val_path = data_root / "val.bin"
    batcher = DeterministicTokenBatcher(
        train_path,
        seq_len=seq,
        batch_size=1,
        seed=args.seed + 101,
        dtype="uint16",
        vocab_size=cfg.vocab_size,
    )
    batcher.load_state_dict(payload["batcher"])
    restore_rng_state(payload["rng"])

    stored_final_ce = float(payload["final_val_ce"])
    stored_best_ce = float(payload["best_val_ce"])
    stored_updates = int(payload["completed_updates"])
    stored_tokens = int(payload["tokens_seen"])
    pre_cursor = dict(batcher.state_dict())

    reproduced_ce = evaluate(
        model,
        val_path,
        seq=seq,
        batches=args.eval_batches,
        seed=args.seed + 202,
    )
    ce_delta = abs(reproduced_ce - stored_final_ce)
    if ce_delta > 1e-6:
        raise RuntimeError(
            f"resumed validation mismatch: {reproduced_ce} vs {stored_final_ce}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _ in range(grad_accum):
        x, y = batcher.next_batch(torch.device("cuda"))
        result = model(x, y)
        if result.loss is None or not torch.isfinite(result.loss):
            raise FloatingPointError("non-finite resumed next-update loss")
        (result.loss / grad_accum).backward()
        loss_sum += float(result.loss.detach())

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    torch.cuda.synchronize()
    post_cursor = dict(batcher.state_dict())

    if post_cursor["epoch"] != 0:
        raise RuntimeError("resume verification unexpectedly replayed training data")

    next_tokens = seq * grad_accum
    report = {
        "schema": 1,
        "suite": "helios1b-pilot1m-checkpoint-resume-verify-v1",
        "checkpoint_sha256": actual_sha,
        "sha256_verified": True,
        "parameters": cfg.parameter_count(),
        "stored_updates": stored_updates,
        "stored_tokens": stored_tokens,
        "stored_best_val_ce": stored_best_ce,
        "stored_final_val_ce": stored_final_ce,
        "reproduced_val_ce": reproduced_ce,
        "reproduced_val_ce_abs_delta": ce_delta,
        "optimizer_state": state_summary,
        "pre_resume_data_cursor": pre_cursor,
        "post_one_update_data_cursor": post_cursor,
        "next_update_tokens": next_tokens,
        "next_update_mean_loss": loss_sum / grad_accum,
        "next_update_grad_norm": float(grad_norm),
        "finite_next_update": (
            math.isfinite(loss_sum)
            and math.isfinite(float(grad_norm))
        ),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "checkpoint_mutated": False,
        "pass": True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
