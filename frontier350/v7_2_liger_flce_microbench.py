from __future__ import annotations

"""Benchmark-only fused LM-head + cross-entropy candidate for Neral 350M.

Requires liger-kernel==0.8.2 on PYTHONPATH.  The benchmark never reads/writes Neral
checkpoints, corpora, optimizer state, or continuity journals.  It compares the current
proven turbo50 full-logits path against a hidden-state backbone + Liger fused-linear CE
path on the exact 48x512 physical batch and checks representative trainable gradients.
"""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from model350 import FRONTIER_350, Neral350M, _rope_table
from protection_v9 import configure_trainable

BATCH = 48
SEQ = 512
REPLAY_ROWS = 12
REPLAY_BOOST = 1.5
COMPILE_MODE = "max-autotune-no-cudagraphs"


def prewarm_rope() -> None:
    _rope_table(FRONTIER_350.max_seq_len, FRONTIER_350.head_dim,
                FRONTIER_350.rope_theta, torch.device("cuda"), torch.bfloat16)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(72101)
    x = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y[:, ::41] = -100
    return x, y


def weighted_torch_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ry, my = y[:REPLAY_ROWS], y[REPLAY_ROWS:]
    rloss = F.cross_entropy(logits[:REPLAY_ROWS].reshape(-1, logits.size(-1)),
                            ry.reshape(-1), ignore_index=-100, reduction="sum")
    mloss = F.cross_entropy(logits[REPLAY_ROWS:].reshape(-1, logits.size(-1)),
                            my.reshape(-1), ignore_index=-100, reduction="sum")
    rv = ry.ne(-100).sum(dtype=torch.float32)
    mv = my.ne(-100).sum(dtype=torch.float32)
    return (rloss * REPLAY_BOOST + mloss) / (mv + REPLAY_BOOST * rv)


def representative_grads(model: Neral350M) -> torch.Tensor:
    names = (
        "blocks.20.attn.q.weight",
        "blocks.23.attn.q.weight",
        "blocks.23.ffn.down.weight",
    )
    named = dict(model.named_parameters())
    pieces = []
    for name in names:
        grad = named[name].grad
        if grad is None:
            raise RuntimeError(f"missing representative gradient: {name}")
        pieces.append(grad.detach().float().reshape(-1).cpu())
    return torch.cat(pieces)


def fidelity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    an = float(a.norm())
    bn = float(b.norm())
    cos = float(F.cosine_similarity(a, b, dim=0))
    rel = float((a - b).norm()) / max(an, 1e-12)
    norm_rel = abs(an - bn) / max(an, 1e-12)
    return {"grad_cosine": cos, "grad_rel_l2": rel, "grad_norm_rel_diff": norm_rel}


def baseline_once(model: Neral350M, run, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = run(x, None)
        loss = weighted_torch_loss(logits, y)
    loss.backward()
    return loss


def make_backbone(model: Neral350M):
    def backbone(xb: torch.Tensor) -> torch.Tensor:
        h = model.tok_emb(xb)
        for block in model.blocks:
            h = block(h)
        return model.norm(h)
    return backbone


def liger_once(model: Neral350M, run_backbone, loss_fn,
               x: torch.Tensor, y: torch.Tensor, denominator: float) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = run_backbone(x)
        h = hidden.reshape(-1, hidden.size(-1))
        split = REPLAY_ROWS * SEQ
        target = y.reshape(-1)
        rloss = loss_fn(model.lm_head.weight, h[:split], target[:split])
        mloss = loss_fn(model.lm_head.weight, h[split:], target[split:])
        loss = (rloss * REPLAY_BOOST + mloss) / denominator
    loss.backward()
    return loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--measured", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=1.08)
    ap.add_argument("--max-loss-diff", type=float, default=2e-4)
    ap.add_argument("--min-grad-cosine", type=float, default=0.999)
    ap.add_argument("--max-grad-rel-l2", type=float, default=0.01)
    a = ap.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 4070" not in gpu:
        raise RuntimeError(f"expected RTX 4070, got {gpu}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(72100); torch.cuda.manual_seed_all(72100)
    model = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    configure_trainable(model, 4)
    prewarm_rope()
    x, y = make_inputs()
    valid_tokens = int(y.ne(-100).sum().item())
    rv = int(y[:REPLAY_ROWS].ne(-100).sum().item())
    mv = int(y[REPLAY_ROWS:].ne(-100).sum().item())
    denominator = float(mv + REPLAY_BOOST * rv)

    baseline_run = torch.compile(model, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    backbone_run = torch.compile(make_backbone(model), fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum", accum_dtype=torch.float32)

    # One fixed-input fidelity pass before speed timing.
    baseline_loss = baseline_once(model, baseline_run, x, y)
    torch.cuda.synchronize()
    baseline_loss_value = float(baseline_loss.detach())
    baseline_grad = representative_grads(model)

    liger_loss = liger_once(model, backbone_run, loss_fn, x, y, denominator)
    torch.cuda.synchronize()
    liger_loss_value = float(liger_loss.detach())
    liger_grad = representative_grads(model)
    grad_report = fidelity(baseline_grad, liger_grad)
    loss_diff = abs(baseline_loss_value - liger_loss_value)

    def measure(kind: str) -> dict:
        rates: list[float] = []
        losses: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.warmup + a.measured):
            torch.cuda.synchronize(); start = time.perf_counter()
            if kind == "baseline":
                loss = baseline_once(model, baseline_run, x, y)
            else:
                loss = liger_once(model, backbone_run, loss_fn, x, y, denominator)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                           error_if_nonfinite=True)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            if i >= a.warmup:
                rates.append(valid_tokens / elapsed)
                losses.append(float(loss.detach()))
            del loss
        return {
            "median_tps": statistics.median(rates),
            "mean_tps": statistics.fmean(rates),
            "loss": losses[-1],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1073741824,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1073741824,
        }

    baseline = measure("baseline")
    candidate = measure("liger")
    ratio = candidate["median_tps"] / baseline["median_tps"]
    failures: list[str] = []
    if ratio < a.min_ratio:
        failures.append("speed_ratio")
    if loss_diff > a.max_loss_diff:
        failures.append("loss_fidelity")
    if grad_report["grad_cosine"] < a.min_grad_cosine:
        failures.append("grad_cosine")
    if grad_report["grad_rel_l2"] > a.max_grad_rel_l2:
        failures.append("grad_rel_l2")
    if not math.isfinite(candidate["loss"]):
        failures.append("nonfinite_loss")

    report = {
        "schema": 1,
        "gpu": gpu,
        "torch": torch.__version__,
        "liger": "0.8.2",
        "batch": BATCH,
        "seq": SEQ,
        "valid_tokens_per_step": valid_tokens,
        "baseline": {"profile": "turbo50-full-logits", **baseline},
        "candidate": {"profile": "liger-fused-linear-ce", **candidate},
        "speed_ratio": ratio,
        "fidelity": {"baseline_loss": baseline_loss_value, "candidate_loss": liger_loss_value,
                     "loss_abs_diff": loss_diff, **grad_report},
        "limits": {"min_ratio": a.min_ratio, "max_loss_diff": a.max_loss_diff,
                   "min_grad_cosine": a.min_grad_cosine, "max_grad_rel_l2": a.max_grad_rel_l2},
        "promote_candidate": not failures,
        "failures": failures,
        "note": "benchmark only; no checkpoint/data/continuity state touched",
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
