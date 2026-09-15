from __future__ import annotations

"""Benchmark frozen-prefix-only QKV/SwiGLU fusion for Neral 350M.

Only blocks 0..19 are converted to fused projection layouts.  The four trainable blocks
20..23, final norm and tied LM head retain the release layout.  This targets execution
speed where gradients are not learned while minimizing numerical drift in trainable
kernels.  Benchmark only: no checkpoint/data/continuity state is touched.
"""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model350 import FRONTIER_350, Block, Neral350M, RMSNorm, _rope_table, rotate_half
from protection_v9 import configure_trainable

BATCH = 48
SEQ = 512
REPLAY_ROWS = 12
REPLAY_BOOST = 1.5
FROZEN_FUSED_BLOCKS = 20
COMPILE_MODE = "max-autotune-no-cudagraphs"
SEED_MODEL = 72500
SEED_INPUT = 72501


class FrozenFusedGQA(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = FRONTIER_350
        self.cfg = cfg
        self.q_size = cfg.d_model
        self.kv_size = cfg.kv_dim
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * cfg.kv_dim, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qn = RMSNorm(cfg.head_dim, cfg.qk_rms_eps)
        self.kn = RMSNorm(cfg.head_dim, cfg.qk_rms_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        c, s = _rope_table(
            self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, q.device, q.dtype
        )
        c = c[..., :t, :]
        s = s[..., :t, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        try:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                enable_gqa=True,
            )
        except (TypeError, RuntimeError):
            k = k.repeat_interleave(self.cfg.gqa_groups, 1)
            v = v.repeat_interleave(self.cfg.gqa_groups, 1)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))


class FrozenFusedSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = FRONTIER_350
        self.hidden = cfg.ffn_hidden
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class FrozenFusedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = FRONTIER_350
        self.n1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = FrozenFusedGQA()
        self.n2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = FrozenFusedSwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class HybridPrefixModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = FRONTIER_350
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [FrozenFusedBlock() for _ in range(FROZEN_FUSED_BLOCKS)]
            + [Block(cfg) for _ in range(FROZEN_FUSED_BLOCKS, cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def convert_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {
        "tok_emb.weight": state["tok_emb.weight"],
        "lm_head.weight": state.get("lm_head.weight", state["tok_emb.weight"]),
        "norm.weight": state["norm.weight"],
    }
    for i in range(FRONTIER_350.n_layers):
        p = f"blocks.{i}."
        if i < FROZEN_FUSED_BLOCKS:
            out[p + "n1.weight"] = state[p + "n1.weight"]
            out[p + "attn.qkv.weight"] = torch.cat(
                [state[p + "attn.q.weight"], state[p + "attn.k.weight"], state[p + "attn.v.weight"]], dim=0
            )
            out[p + "attn.o.weight"] = state[p + "attn.o.weight"]
            out[p + "attn.qn.weight"] = state[p + "attn.qn.weight"]
            out[p + "attn.kn.weight"] = state[p + "attn.kn.weight"]
            out[p + "n2.weight"] = state[p + "n2.weight"]
            out[p + "ffn.gate_up.weight"] = torch.cat(
                [state[p + "ffn.gate.weight"], state[p + "ffn.up.weight"]], dim=0
            )
            out[p + "ffn.down.weight"] = state[p + "ffn.down.weight"]
        else:
            for suffix in (
                "n1.weight", "attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.o.weight",
                "attn.qn.weight", "attn.kn.weight", "n2.weight", "ffn.gate.weight",
                "ffn.up.weight", "ffn.down.weight",
            ):
                out[p + suffix] = state[p + suffix]
    return out


def configure_hybrid_trainable(model: HybridPrefixModel) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for block in model.blocks[-4:]:
        block.float()
        for p in block.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


def make_inputs():
    torch.manual_seed(SEED_INPUT)
    torch.cuda.manual_seed_all(SEED_INPUT)
    x = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, FRONTIER_350.vocab_size, (BATCH, SEQ), device="cuda")
    y[:, ::41] = -100
    return x, y


def weighted_loss(logits, y):
    ry, my = y[:REPLAY_ROWS], y[REPLAY_ROWS:]
    rloss = F.cross_entropy(logits[:REPLAY_ROWS].reshape(-1, logits.size(-1)), ry.reshape(-1), ignore_index=-100, reduction="sum")
    mloss = F.cross_entropy(logits[REPLAY_ROWS:].reshape(-1, logits.size(-1)), my.reshape(-1), ignore_index=-100, reduction="sum")
    rv = ry.ne(-100).sum(dtype=torch.float32)
    mv = my.ne(-100).sum(dtype=torch.float32)
    return (rloss * REPLAY_BOOST + mloss) / (mv + REPLAY_BOOST * rv)


def prewarm(model):
    _rope_table(FRONTIER_350.max_seq_len, FRONTIER_350.head_dim, FRONTIER_350.rope_theta,
                next(model.parameters()).device, torch.bfloat16)


def grad_vector(model) -> torch.Tensor:
    named = dict(model.named_parameters())
    pieces = []
    for i in (20, 23):
        p = f"blocks.{i}."
        for suffix in (
            "attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.o.weight",
            "ffn.gate.weight", "ffn.up.weight", "ffn.down.weight",
        ):
            g = named[p + suffix].grad
            if g is None:
                raise RuntimeError(f"missing gradient {p + suffix}")
            pieces.append(g.detach().float().reshape(-1).cpu())
    return torch.cat(pieces)


def run_step(model, compiled, params, x, y):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = compiled(x, None)
        loss = weighted_loss(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    return loss


def measure(model, compiled, params, x, y, warmup, measured, valid_tokens):
    rates, losses = [], []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measured", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.03)
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

    torch.manual_seed(SEED_MODEL); torch.cuda.manual_seed_all(SEED_MODEL)
    base = Neral350M().to("cuda", dtype=torch.bfloat16).train()
    base_params = [p for _, p in configure_trainable(base, 4)]
    prewarm(base)
    base_run = torch.compile(base, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    base_fidelity_loss = run_step(base, base_run, base_params, x, y)
    torch.cuda.synchronize()
    base_loss_value = float(base_fidelity_loss.detach())
    base_grad = grad_vector(base)
    baseline = measure(base, base_run, base_params, x, y, a.warmup, a.measured, valid_tokens)
    source_state = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    del base_fidelity_loss, base_run, base, base_params
    gc.collect(); torch.cuda.empty_cache()

    hybrid = HybridPrefixModel().to(dtype=torch.bfloat16)
    converted = convert_prefix(source_state)
    missing, unexpected = hybrid.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"hybrid conversion mismatch missing={missing} unexpected={unexpected}")
    del converted, source_state
    hybrid = hybrid.to("cuda").train()
    hybrid_params = configure_hybrid_trainable(hybrid)
    total = sum(p.numel() for p in hybrid.parameters())
    trainable = sum(p.numel() for p in hybrid_params)
    if total != 349_825_152 or trainable != 55_158_272:
        raise RuntimeError(f"parameter contract changed total={total} trainable={trainable}")
    prewarm(hybrid)
    hybrid_run = torch.compile(hybrid, fullgraph=True, dynamic=False, mode=COMPILE_MODE)
    hybrid_fidelity_loss = run_step(hybrid, hybrid_run, hybrid_params, x, y)
    torch.cuda.synchronize()
    hybrid_loss_value = float(hybrid_fidelity_loss.detach())
    hybrid_grad = grad_vector(hybrid)
    candidate = measure(hybrid, hybrid_run, hybrid_params, x, y, a.warmup, a.measured, valid_tokens)

    an = float(base_grad.norm())
    grad_cosine = float(F.cosine_similarity(base_grad, hybrid_grad, dim=0))
    grad_rel_l2 = float((base_grad - hybrid_grad).norm()) / max(an, 1e-12)
    grad_norm_rel_diff = abs(an - float(hybrid_grad.norm())) / max(an, 1e-12)
    loss_diff = abs(base_loss_value - hybrid_loss_value)
    ratio = candidate["median_tps"] / baseline["median_tps"]
    failures = []
    if ratio < a.min_ratio: failures.append("speed_ratio")
    if loss_diff > a.max_loss_diff: failures.append("loss_fidelity")
    if grad_cosine < a.min_grad_cosine: failures.append("grad_cosine")
    if grad_rel_l2 > a.max_grad_rel_l2: failures.append("grad_rel_l2")

    report = {
        "schema": 1,
        "gpu": gpu,
        "torch": torch.__version__,
        "batch": BATCH,
        "seq": SEQ,
        "frozen_fused_blocks": FROZEN_FUSED_BLOCKS,
        "trainable_release_layout_blocks": 4,
        "parameters": total,
        "trainable_parameters": trainable,
        "baseline": {"profile": "release-layout", **baseline},
        "candidate": {"profile": "fused-frozen-prefix-only", **candidate},
        "speed_ratio": ratio,
        "fidelity": {
            "baseline_loss": base_loss_value,
            "candidate_loss": hybrid_loss_value,
            "loss_abs_diff": loss_diff,
            "grad_cosine": grad_cosine,
            "grad_rel_l2": grad_rel_l2,
            "grad_norm_rel_diff": grad_norm_rel_diff,
        },
        "limits": {
            "min_ratio": a.min_ratio,
            "max_loss_diff": a.max_loss_diff,
            "min_grad_cosine": a.min_grad_cosine,
            "max_grad_rel_l2": a.max_grad_rel_l2,
        },
        "promote_candidate": not failures,
        "failures": failures,
        "note": "benchmark only; trainable blocks retain release layout; no continuity state touched",
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
