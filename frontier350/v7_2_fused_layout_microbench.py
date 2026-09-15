from __future__ import annotations

"""Benchmark-only QKV + SwiGLU fusion experiment for Neral 350M.

The candidate is checkpoint-convertible and preserves the parameter count and learned
function.  This script uses deterministic synthetic inputs, exact weight conversion and
mapped gradient checks.  It never reads or writes Neral checkpoints, corpora, optimizer
state, or continuity journals.
"""

import argparse
import gc
import json
import statistics
import time

import torch
import torch.nn.functional as F

from model350 import FRONTIER_350, Neral350M, _rope_table
from model350_fast import Neral350MFast, convert_unfused_state_dict
from protection_v9 import configure_trainable

BATCH = 48
SEQ = 512
REPLAY_ROWS = 12
REPLAY_BOOST = 1.5
COMPILE_MODE = "max-autotune-no-cudagraphs"
SEED_MODEL = 72300


def prewarm_rope() -> None:
    _rope_table(FRONTIER_350.max_seq_len, FRONTIER_350.head_dim,
                FRONTIER_350.rope_theta, torch.device("cuda"), torch.bfloat16)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(72301)
    x = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y[:, ::41] = -100
    return x, y


def weighted_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ry, my = y[:REPLAY_ROWS], y[REPLAY_ROWS:]
    rloss = F.cross_entropy(logits[:REPLAY_ROWS].reshape(-1, logits.size(-1)),
                            ry.reshape(-1), ignore_index=-100, reduction="sum")
    mloss = F.cross_entropy(logits[REPLAY_ROWS:].reshape(-1, logits.size(-1)),
                            my.reshape(-1), ignore_index=-100, reduction="sum")
    rv = ry.ne(-100).sum(dtype=torch.float32)
    mv = my.ne(-100).sum(dtype=torch.float32)
    return (rloss * REPLAY_BOOST + mloss) / (mv + REPLAY_BOOST * rv)


def configure_fast_trainable(model: Neral350MFast) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[torch.nn.Parameter] = []
    for i in range(FRONTIER_350.n_layers - 4, FRONTIER_350.n_layers):
        for p in model.blocks[i].parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


def base_grad_vector(model: Neral350M) -> torch.Tensor:
    n = dict(model.named_parameters())
    pieces: list[torch.Tensor] = []
    for i in (20, 23):
        p = f"blocks.{i}."
        for name in (
            p + "attn.q.weight", p + "attn.k.weight", p + "attn.v.weight",
            p + "attn.o.weight", p + "ffn.gate.weight", p + "ffn.up.weight",
            p + "ffn.down.weight",
        ):
            g = n[name].grad
            if g is None:
                raise RuntimeError(f"missing baseline gradient {name}")
            pieces.append(g.detach().float().reshape(-1).cpu())
    return torch.cat(pieces)


def fast_grad_vector(model: Neral350MFast) -> torch.Tensor:
    n = dict(model.named_parameters())
    pieces: list[torch.Tensor] = []
    qn = FRONTIER_350.d_model
    kv = FRONTIER_350.kv_dim
    ff = FRONTIER_350.ffn_hidden
    for i in (20, 23):
        p = f"blocks.{i}."
        qkv = n[p + "attn.qkv.weight"].grad
        gu = n[p + "ffn.gate_up.weight"].grad
        if qkv is None or gu is None:
            raise RuntimeError(f"missing fused gradient block {i}")
        pieces.extend([
            qkv[:qn].detach().float().reshape(-1).cpu(),
            qkv[qn:qn + kv].detach().float().reshape(-1).cpu(),
            qkv[qn + kv:qn + 2 * kv].detach().float().reshape(-1).cpu(),
        ])
        for name in (p + "attn.o.weight",):
            g = n[name].grad
            if g is None:
                raise RuntimeError(f"missing fused gradient {name}")
            pieces.append(g.detach().float().reshape(-1).cpu())
        pieces.extend([
            gu[:ff].detach().float().reshape(-1).cpu(),
            gu[ff:2 * ff].detach().float().reshape(-1).cpu(),
        ])
        g = n[p + "ffn.down.weight"].grad
        if g is None:
            raise RuntimeError(f"missing fused gradient {p}ffn.down.weight")
        pieces.append(g.detach().float().reshape(-1).cpu())
    return torch.cat(pieces)


def grad_fidelity(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    an = float(a.norm())
    cos = float(F.cosine_similarity(a, b, dim=0))
    rel = float((a - b).norm()) / max(an, 1e-12)
    norm_rel = abs(an - float(b.norm())) / max(an, 1e-12)
    return {"grad_cosine": cos, "grad_rel_l2": rel, "grad_norm_rel_diff": norm_rel}


def run_step(model, compiled, params, x, y) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = compiled(x, None)
        loss = weighted_loss(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    return loss


def measure(model, compiled, params, x, y, warmup: int, measured: int,
            valid_tokens: int) -> dict:
    rates: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + measured):
        torch.cuda.synchronize(); start = time.perf_counter()
        loss = run_step(model, compiled, params, x, y)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        if i >= warmup:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measured", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.10)
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
    x, y = make_inputs()
    valid_tokens = int(y.ne(-100).sum().item())

    # Baseline. Re-seeding later recreates exactly the same unfused FP32 initialization
    # used as the source for the fused state conversion.
    torch.manual_seed(SEED_MODEL); torch.cuda.manual_seed_all(SEED_MODEL)
    base = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    base_named = configure_trainable(base, 4)
    base_params = [p for _, p in base_named]
    if sum(p.numel() for p in base.parameters()) != 349_825_152:
        raise RuntimeError("baseline parameter contract changed")
    prewarm_rope()
    base_run = torch.compile(base, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    base_fidelity_loss = run_step(base, base_run, base_params, x, y)
    torch.cuda.synchronize()
    base_loss_value = float(base_fidelity_loss.detach())
    base_grad = base_grad_vector(base)
    baseline = measure(base, base_run, base_params, x, y, a.warmup, a.measured, valid_tokens)
    del base_fidelity_loss, base_run, base, base_params, base_named
    gc.collect(); torch.cuda.empty_cache()

    # Recreate the identical source weights on CPU, convert them by concatenation, then
    # benchmark only the converted fast layout on CUDA.
    torch.manual_seed(SEED_MODEL)
    source = Neral350M().to(dtype=torch.bfloat16)
    converted = convert_unfused_state_dict(source.state_dict())
    fast = Neral350MFast().to(dtype=torch.bfloat16)
    missing, unexpected = fast.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"fused conversion mismatch missing={missing} unexpected={unexpected}")
    del converted, source
    fast = fast.to("cuda").train()
    fast_params = configure_fast_trainable(fast)
    total = sum(p.numel() for p in fast.parameters())
    trainable = sum(p.numel() for p in fast_params)
    if total != 349_825_152:
        raise RuntimeError(f"fused parameter contract changed: {total}")
    if trainable != 55_158_272:
        raise RuntimeError(f"fused trainable parameter contract changed: {trainable}")
    prewarm_rope()
    fast_run = torch.compile(fast, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    fast_fidelity_loss = run_step(fast, fast_run, fast_params, x, y)
    torch.cuda.synchronize()
    fast_loss_value = float(fast_fidelity_loss.detach())
    fast_grad = fast_grad_vector(fast)
    candidate = measure(fast, fast_run, fast_params, x, y, a.warmup, a.measured, valid_tokens)

    fidelity = grad_fidelity(base_grad, fast_grad)
    loss_diff = abs(base_loss_value - fast_loss_value)
    ratio = candidate["median_tps"] / baseline["median_tps"]
    failures: list[str] = []
    if ratio < a.min_ratio:
        failures.append("speed_ratio")
    if loss_diff > a.max_loss_diff:
        failures.append("loss_fidelity")
    if fidelity["grad_cosine"] < a.min_grad_cosine:
        failures.append("grad_cosine")
    if fidelity["grad_rel_l2"] > a.max_grad_rel_l2:
        failures.append("grad_rel_l2")

    report = {
        "schema": 1,
        "gpu": gpu,
        "torch": torch.__version__,
        "batch": BATCH,
        "seq": SEQ,
        "valid_tokens_per_step": valid_tokens,
        "parameters": total,
        "trainable_parameters": trainable,
        "baseline": {"profile": "turbo50-unfused", **baseline},
        "candidate": {"profile": "fused-qkv-gateup-native-rmsnorm", **candidate},
        "speed_ratio": ratio,
        "fidelity": {"baseline_loss": base_loss_value, "candidate_loss": fast_loss_value,
                     "loss_abs_diff": loss_diff, **fidelity},
        "limits": {"min_ratio": a.min_ratio, "max_loss_diff": a.max_loss_diff,
                   "min_grad_cosine": a.min_grad_cosine, "max_grad_rel_l2": a.max_grad_rel_l2},
        "promote_candidate": not failures,
        "failures": failures,
        "note": "benchmark only; exact deterministic weight conversion; no checkpoint/data/continuity state touched",
    }
    from pathlib import Path
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
