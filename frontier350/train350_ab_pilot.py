from __future__ import annotations

"""Fair bounded real-data learning pilot for Neral 350M Frontier architecture selection.

Deep32 and Wide24 are trained from scratch on the exact same locked BPE corpus, token order,
seed, optimizer and LR schedule. Frozen release benchmarks are never touched. This pilot is
only allowed to compare validation CE/BPB, training stability, throughput and VRAM.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model350 import Neral350Config, Neral350M

CANDIDATES = {
    "deep32": Neral350Config(vocab_size=16384, d_model=1024, n_layers=32, n_heads=16, n_kv_heads=2, ffn_hidden=2624),
    "wide24": Neral350Config(vocab_size=16384, d_model=1152, n_layers=24, n_heads=18, n_kv_heads=2, ffn_hidden=3136),
}


def load_partition(root: Path, name: str) -> np.ndarray:
    files = sorted((root / name).glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"no {name} shards under {root}")
    arrays = [np.fromfile(path, dtype=np.uint16) for path in files]
    data = np.concatenate(arrays)
    if data.size < 10_000:
        raise RuntimeError(f"{name} partition too small")
    return data


def make_batch(data: np.ndarray, seq: int, batch: int, cursor: int, device: torch.device):
    span = seq + 1
    starts = [(cursor + i * span) % (len(data) - span) for i in range(batch)]
    block = np.stack([data[s:s + span] for s in starts]).astype(np.int64, copy=False)
    t = torch.from_numpy(block).to(device=device, non_blocking=True)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model: Neral350M, data: np.ndarray, *, seq: int, batch: int, batches: int, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    losses = []
    stride = max(seq + 1, (len(data) - (seq + 1)) // max(1, batches))
    for i in range(batches):
        x, y = make_batch(data, seq, batch, i * stride, device)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        losses.append(float(loss))
    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=3_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=24)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 required")
    if args.target_tokens < 1_000_000:
        raise SystemExit("quality pilot too small")

    seed = 3502026
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    root = Path(args.data_root)
    train = load_partition(root, "train")
    valid = load_partition(root, "valid")
    cfg = CANDIDATES[args.candidate]
    model = Neral350M(cfg).to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(False)
    params = model.parameter_report()
    if not 345_000_000 <= params["total"] <= 360_000_000:
        raise RuntimeError(f"parameter contract failed: {params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    tokens_per_step = args.seq * args.microbatch
    steps = math.ceil(args.target_tokens / tokens_per_step)
    milestones = sorted(set([0, max(1, steps // 6), max(1, steps // 3), max(1, 2 * steps // 3), steps]))
    metrics: list[dict] = []
    cursor = 0
    measured_tokens = 0
    measured_time = 0.0
    torch.cuda.reset_peak_memory_stats()

    initial_val = evaluate(model, valid, seq=args.seq, batch=1, batches=args.eval_batches, device=device)
    metrics.append({"step": 0, "tokens": 0, "validation_ce": initial_val, "validation_bpb": initial_val / math.log(2.0)})
    print(json.dumps({"candidate": args.candidate, **metrics[-1]}), flush=True)

    model.train()
    for step in range(1, steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, steps - args.warmup_steps)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = make_batch(train, args.seq, args.microbatch, cursor, device)
        cursor = (cursor + tokens_per_step) % max(1, len(train) - (args.seq + 1) * args.microbatch)
        torch.cuda.synchronize(); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(grad)):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        optimizer.step()
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        measured_time += elapsed; measured_tokens += tokens_per_step

        if step % 50 == 0 or step == 1:
            print(json.dumps({"candidate": args.candidate, "step": step, "tokens": min(measured_tokens, args.target_tokens), "train_loss": float(loss), "grad_norm": float(grad), "lr": lr, "tokens_per_second": measured_tokens / measured_time, "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9}), flush=True)
        if step in milestones and step != steps:
            val = evaluate(model, valid, seq=args.seq, batch=1, batches=args.eval_batches, device=device)
            row = {"step": step, "tokens": min(step * tokens_per_step, args.target_tokens), "validation_ce": val, "validation_bpb": val / math.log(2.0), "train_loss": float(loss)}
            metrics.append(row); print(json.dumps({"candidate": args.candidate, **row}), flush=True)

    final_val = evaluate(model, valid, seq=args.seq, batch=1, batches=args.eval_batches, device=device)
    final = {
        "candidate": args.candidate,
        "parameters": params,
        "config": cfg.to_dict(),
        "seed": seed,
        "target_tokens": args.target_tokens,
        "processed_tokens": measured_tokens,
        "seq": args.seq,
        "microbatch": args.microbatch,
        "tokens_per_second": measured_tokens / measured_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9,
        "initial_validation_ce": initial_val,
        "final_validation_ce": final_val,
        "final_validation_bpb": final_val / math.log(2.0),
        "validation_ce_improvement_pct": 100.0 * (initial_val - final_val) / initial_val,
        "finite": math.isfinite(final_val),
        "curve": [*metrics, {"step": steps, "tokens": min(measured_tokens, args.target_tokens), "validation_ce": final_val, "validation_bpb": final_val / math.log(2.0)}],
        "selection_scope": "architecture learning pilot only; no release benchmark claim",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2), encoding="utf8")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
