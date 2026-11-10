from __future__ import annotations

"""Checkpoint-native Stage-7 A/B gate for Arctus 350M execution speed.

This is deliberately report-only. It loads one immutable unfused resume checkpoint,
converts model + AdamW state in memory to the fused QKV/gate-up execution layout, and
compares the real training path on identical data/seed. It never writes a checkpoint and
never promotes the fused layout automatically.
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

from convert_checkpoint350_fast import convert_optimizer
from model350 import FRONTIER_350, Neral350M
from model350_fast import Neral350MFast, convert_unfused_state_dict

PARAMETERS = 349_825_152
TOKENS_PER_UPDATE = 16_384


class MMapPool:
    def __init__(self, root: Path):
        paths = sorted(root.glob("*.bin"))
        if not paths:
            raise FileNotFoundError(f"no *.bin shards in {root}")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]

    def batch(self, batch: int, seq: int, rng: random.Random, device: torch.device):
        rows = np.empty((batch, seq + 1), dtype=np.int64)
        for i in range(batch):
            a = rng.choice(self.arrays)
            if len(a) < seq + 1:
                raise ValueError("speed shard too short")
            start = rng.randrange(0, len(a) - seq)
            rows[i] = a[start : start + seq + 1]
        if rows.max(initial=0) >= FRONTIER_350.vocab_size:
            raise ValueError("token outside locked vocabulary")
        whole = torch.from_numpy(rows).pin_memory().to(device, non_blocking=True)
        return whole[:, :-1], whole[:, 1:]


def _configure_cuda() -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _make_model(layout: str, model_state: dict, device: torch.device):
    if layout == "unfused":
        model = Neral350M()
    elif layout == "fused":
        model = Neral350MFast()
    else:
        raise ValueError(layout)
    model.load_state_dict(model_state, strict=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(False)
    if model.parameter_report()["total"] != PARAMETERS:
        raise RuntimeError("Arctus parameter contract changed")
    return model


def _eval_signature(layout: str, state: dict, pool: MMapPool, *, mb: int, seq: int,
                    seed: int, device: torch.device) -> dict:
    model = _make_model(layout, state, device)
    model.eval()
    rng = random.Random(seed)
    x, y = pool.batch(mb, seq, rng, device)
    with torch.no_grad():
        logits, loss = model(x, y)
        # Last-token logits are enough to expose execution-layout numerical drift while
        # avoiding a >100 MB full-logit host copy.
        tail = logits[:, -1, :].float().cpu()
        loss_value = float(loss)
    del logits, loss, x, y, model
    torch.cuda.empty_cache()
    gc.collect()
    return {"loss": loss_value, "tail_logits": tail}


def _run_training_case(layout: str, model_state: dict, optimizer_state: dict, pool: MMapPool,
                       *, mb: int, seq: int, grad_accum: int, warmup: int, updates: int,
                       seed: int, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed + 1)
    raw = _make_model(layout, model_state, device)
    raw.train()
    opt = torch.optim.AdamW(
        raw.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.1, fused=True,
    )
    opt.load_state_dict(optimizer_state)
    compiled = torch.compile(raw, backend="inductor", mode="default", fullgraph=True, dynamic=False)

    def run(n: int):
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        last_grad = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                x, y = pool.batch(mb, seq, rng, device)
                _, loss = compiled(x, y)
                if loss is None:
                    raise RuntimeError("missing loss")
                (loss / grad_accum).backward()
                loss_sum.add_(loss.detach().float())
            last_grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
        return loss_sum, last_grad

    torch.cuda.reset_peak_memory_stats()
    warm_loss, warm_grad = run(warmup)
    torch.cuda.synchronize()
    warm_loss_value = float(warm_loss / (warmup * grad_accum))
    warm_grad_value = float(warm_grad)
    if not math.isfinite(warm_loss_value) or not math.isfinite(warm_grad_value):
        raise RuntimeError(f"{layout} non-finite warmup")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    measured_loss, measured_grad = run(updates)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    loss_value = float(measured_loss / (updates * grad_accum))
    grad_value = float(measured_grad)
    if not math.isfinite(loss_value) or not math.isfinite(grad_value):
        raise RuntimeError(f"{layout} non-finite measured values")

    tps = TOKENS_PER_UPDATE * updates / seconds
    row = {
        "layout": layout,
        "status": "ok",
        "microbatch": mb,
        "grad_accum": grad_accum,
        "seq": seq,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "compile_mode": "default",
        "fullgraph": True,
        "warmup_updates": warmup,
        "measured_updates": updates,
        "warmup_loss": warm_loss_value,
        "measured_loss": loss_value,
        "last_grad_norm": grad_value,
        "e2e_tokens_per_second": tps,
        "measured_seconds": seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }
    del compiled, opt, raw
    torch.cuda.empty_cache()
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--microbatch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--updates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=35700)
    args = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 GPU required")
    if min(args.seq, args.microbatch, args.warmup, args.updates) <= 0:
        raise SystemExit("positive dimensions required")
    per_micro = args.seq * args.microbatch
    if TOKENS_PER_UPDATE % per_micro:
        raise SystemExit("microbatch/seq must exactly divide 16,384 tokens/update")
    grad_accum = TOKENS_PER_UPDATE // per_micro

    _configure_cuda()
    device = torch.device("cuda")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint is not locked Frontier 350M")
    if "model" not in ck or "optimizer" not in ck:
        raise RuntimeError("resume checkpoint requires model + optimizer state")

    unfused_model = ck["model"]
    unfused_opt = ck["optimizer"]
    fused_model = convert_unfused_state_dict(unfused_model)
    fused_opt = convert_optimizer(unfused_opt)
    pool = MMapPool(Path(args.train_data))

    # Numerical signature at the immutable source checkpoint before either A/B arm trains.
    base_sig = _eval_signature("unfused", unfused_model, pool, mb=args.microbatch,
                               seq=args.seq, seed=args.seed + 99, device=device)
    fast_sig = _eval_signature("fused", fused_model, pool, mb=args.microbatch,
                               seq=args.seq, seed=args.seed + 99, device=device)
    a = base_sig.pop("tail_logits")
    b = fast_sig.pop("tail_logits")
    diff = (a - b).abs()
    denom = torch.linalg.vector_norm(a.flatten()) * torch.linalg.vector_norm(b.flatten())
    cosine = float(torch.dot(a.flatten(), b.flatten()) / denom) if float(denom) > 0 else 1.0
    parity = {
        "unfused_loss": base_sig["loss"],
        "fused_loss": fast_sig["loss"],
        "loss_abs_diff": abs(base_sig["loss"] - fast_sig["loss"]),
        "tail_logits_max_abs_diff": float(diff.max()),
        "tail_logits_mean_abs_diff": float(diff.mean()),
        "tail_logits_cosine": cosine,
    }
    del a, b, diff

    baseline = _run_training_case(
        "unfused", unfused_model, unfused_opt, pool,
        mb=args.microbatch, seq=args.seq, grad_accum=grad_accum,
        warmup=args.warmup, updates=args.updates, seed=args.seed, device=device,
    )
    fused = _run_training_case(
        "fused", fused_model, fused_opt, pool,
        mb=args.microbatch, seq=args.seq, grad_accum=grad_accum,
        warmup=args.warmup, updates=args.updates, seed=args.seed, device=device,
    )
    speedup_pct = (fused["e2e_tokens_per_second"] / baseline["e2e_tokens_per_second"] - 1.0) * 100.0

    report = {
        "schema": 1,
        "architecture": "frontier350-v1",
        "parameters": PARAMETERS,
        "source_checkpoint_tokens": int(ck.get("processed_tokens", -1)),
        "source_checkpoint_step": int(ck.get("step", -1)),
        "checkpoint_mutated": False,
        "optimizer_state_converted": True,
        "parity": parity,
        "unfused": baseline,
        "fused": fused,
        "fused_speedup_pct": speedup_pct,
        "promotion_allowed": False,
        "note": "Report-only A/B. Promotion requires explicit numerical tolerances, fixed validation A/B, resume round-trip validation, and a separate user-approved training request.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
