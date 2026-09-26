from __future__ import annotations

"""Disposable Neral-v2 ~1B RTX 4070 feasibility benchmark.

This is NOT productive training. It initializes random Neral-owned weights, uses
synthetic token IDs, performs a few real forward/backward/optimizer updates and
writes aggregate performance telemetry only. No checkpoint is saved.

Candidate architecture:
- own 16K-vocab contract
- 32 decoder blocks
- d_model 1536
- 24 query heads / 8 KV heads (head dim 64)
- SwiGLU hidden 5120
- tied input/output embeddings
- RoPE + RMSNorm + GQA
Exact unique parameter count: 981,566,976.
"""

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16_384
    d_model: int = 1_536
    n_layers: int = 32
    n_heads: int = 24
    n_kv_heads: int = 8
    ffn_hidden: int = 5_120
    max_seq_len: int = 2_048
    rope_theta: float = 100_000.0
    rms_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.n_heads // self.n_kv_heads


CFG = Config()
EXPECTED_PARAMS = 981_566_976


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


class GQA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, cfg.kv_dim, bias=False)
        self.v = nn.Linear(d, cfg.kv_dim, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.cfg
        q = self.q(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        except (TypeError, RuntimeError):
            k = k.repeat_interleave(c.gqa_groups, dim=1)
            v = v.repeat_interleave(c.gqa_groups, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, c.d_model))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = GQA(cfg)
        self.n2 = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), cos, sin)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class NeralV2OneB(nn.Module):
    def __init__(self, cfg: Config, gradient_checkpointing: bool):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = gradient_checkpointing
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        pos = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        phase = torch.outer(pos, inv).repeat_interleave(2, dim=-1)
        self.register_buffer("rope_cos", phase.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", phase.sin()[None, None], persistent=False)
        self.apply(self._init)
        self._scale_residuals()

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _scale_residuals(self) -> None:
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, 0.0, std)
            nn.init.normal_(b.down.weight, 0.0, std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid input")
        t = idx.size(1)
        cos = self.rope_cos[:, :, :t].to(dtype=self.tok_emb.weight.dtype)
        sin = self.rope_sin[:, :, :t].to(dtype=self.tok_emb.weight.dtype)
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def zeropower_newton_schulz5(grad: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the matrix zeroth power used by Muon.

    The iteration intentionally runs in BF16: that is the efficient/stable regime
    used by the published Muon algorithm for transformer hidden matrices.
    """
    if grad.ndim != 2:
        raise ValueError("Muon requires 2-D hidden matrices")
    if not 1 <= steps < 100:
        raise ValueError("invalid Newton-Schulz step count")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.to(dtype=torch.bfloat16, copy=True)
    x.div_(x.norm().clamp_min(eps))
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    if transposed:
        x = x.mT
    return x


class LocalMuon(torch.optim.Optimizer):
    """Dependency-free Muon fallback for PyTorch releases without torch.optim.Muon."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("invalid Muon hyperparameters")
        defaults = dict(
            lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay),
            nesterov=bool(nesterov), ns_steps=int(ns_steps)
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError("LocalMuon received non-matrix parameter")
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    # Keeping momentum in the parameter/gradient dtype is essential
                    # to make a ~1B model feasible on a 12-GB consumer GPU.
                    buf = state["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_newton_schulz5(update, steps=ns_steps)

                # match_rms_adamw scaling from the scalable-Muon recipe.
                adjusted_lr = lr * 0.2 * math.sqrt(max(p.shape))
                if weight_decay:
                    p.mul_(1.0 - adjusted_lr * weight_decay)
                p.add_(update.to(dtype=p.dtype), alpha=-adjusted_lr)
        return loss


def optimizer_self_test(device: torch.device) -> None:
    p = nn.Parameter(torch.randn(16, 8, device=device, dtype=torch.bfloat16))
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt = LocalMuon([p], lr=0.02, momentum=0.95, weight_decay=0.1)
    opt.step()
    if not bool(torch.isfinite(p).all()) or bool(torch.equal(before, p)):
        raise RuntimeError("local Muon self-test failed")
    state = opt.state[p].get("momentum_buffer")
    if state is None or state.dtype != torch.bfloat16 or state.shape != p.shape:
        raise RuntimeError("local Muon momentum-state contract failed")
    del opt, p, before, state


def build_optimizers(model: nn.Module):
    matrix_params, other_params = [], []
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        # Muon is for hidden 2-D matrices; tied token embeddings stay on AdamW.
        if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name:
            matrix_params.append(p)
        else:
            other_params.append(p)

    if hasattr(torch.optim, "Muon"):
        try:
            muon = torch.optim.Muon(
                matrix_params, lr=0.02, momentum=0.95, weight_decay=0.1,
                adjust_lr_fn="match_rms_adamw"
            )
            muon_impl = "torch.optim.Muon"
        except TypeError:
            muon = LocalMuon(matrix_params, lr=0.02, momentum=0.95, weight_decay=0.1)
            muon_impl = "local_muon_v1"
    else:
        muon = LocalMuon(matrix_params, lr=0.02, momentum=0.95, weight_decay=0.1)
        muon_impl = "local_muon_v1"

    adamw = torch.optim.AdamW(
        other_params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.01, fused=True
    )
    return (
        muon, adamw,
        sum(p.numel() for p in matrix_params),
        sum(p.numel() for p in other_params),
        muon_impl,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--gradient-checkpointing", choices=("on", "off"), default="off")
    ap.add_argument("--compile-mode", choices=("none", "default"), default="none")
    ap.add_argument("--warmup-updates", type=int, default=1)
    ap.add_argument("--measure-updates", type=int, default=2)
    args = ap.parse_args()
    if not args.execute:
        ap.error("--execute required")
    if args.out.exists():
        raise RuntimeError("output is create-only")
    if not (1 <= args.seq <= CFG.max_seq_len and 1 <= args.microbatch <= 8):
        raise ValueError("invalid benchmark shape")
    if min(args.warmup_updates, args.measure_updates) <= 0:
        raise ValueError("positive update counts required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU required")

    torch.manual_seed(12026)
    torch.cuda.manual_seed_all(12026)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer_self_test(device)
    torch.cuda.empty_cache()

    model = NeralV2OneB(CFG, args.gradient_checkpointing == "on").to(device=device, dtype=torch.bfloat16)
    params = param_count(model)
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"parameter contract changed: {params:,} != {EXPECTED_PARAMS:,}")
    muon, adamw, muon_params, adamw_params, muon_impl = build_optimizers(model)

    train_model = model
    if args.compile_mode == "default":
        train_model = torch.compile(model, mode="default", fullgraph=False, dynamic=False)

    gen = torch.Generator(device=device)
    gen.manual_seed(12026)
    x = torch.randint(0, CFG.vocab_size, (args.microbatch, args.seq), generator=gen, device=device)
    y = torch.randint(0, CFG.vocab_size, (args.microbatch, args.seq), generator=gen, device=device)

    def update() -> float:
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss = train_model(x, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        muon.step()
        adamw.step()
        return float(loss.detach())

    model.train()
    warmup_seconds = 0.0
    for _ in range(args.warmup_updates):
        t0 = time.perf_counter()
        update()
        torch.cuda.synchronize()
        warmup_seconds += time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for _ in range(args.measure_updates):
        t0 = time.perf_counter()
        losses.append(update())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    measured_seconds = sum(times)
    tokens_per_update = args.seq * args.microbatch
    measured_tokens = tokens_per_update * args.measure_updates
    props = torch.cuda.get_device_properties(0)
    summary = {
        "schema": "neral_v2_1b_rtx4070_feasibility_v1",
        "classification": "synthetic_disposable_benchmark_no_productive_training",
        "architecture": asdict(CFG),
        "parameters": params,
        "muon_parameters": muon_params,
        "adamw_parameters": adamw_params,
        "parameter_dtype": "bfloat16",
        "gradient_checkpointing": args.gradient_checkpointing == "on",
        "compile_mode": args.compile_mode,
        "seq": args.seq,
        "microbatch": args.microbatch,
        "tokens_per_update": tokens_per_update,
        "warmup_seconds": warmup_seconds,
        "measured_updates": args.measure_updates,
        "measured_seconds": measured_seconds,
        "tokens_per_second": measured_tokens / measured_seconds,
        "seconds_per_update": measured_seconds / args.measure_updates,
        "mean_loss": sum(losses) / len(losses),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "memory_total_mib": props.total_memory / 1024**2,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "native_muon_available": hasattr(torch.optim, "Muon"),
        "muon_implementation": muon_impl,
        "muon_momentum_dtype": "bfloat16" if muon_impl == "local_muon_v1" else "implementation_default",
        "checkpoint_written": False,
        "productive_training_tokens": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)

    del train_model, model, muon, adamw, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
