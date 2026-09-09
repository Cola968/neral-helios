from __future__ import annotations

"""Dense Neral 600M architecture-search model family.

This is intentionally a clean from-scratch implementation.  It reuses no foreign
weights/checkpoints and keeps Neral's own 16K Byte-BPE contract.  The execution layout
uses fused QKV and fused SwiGLU gate/up projections from the beginning because those
fusions do not change the mathematical function or parameter count.
"""

from dataclasses import replace
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from architecture_search600 import Arch600, CANDIDATES, VOCAB_SIZE

_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def rope_table(max_seq_len: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index, str(dtype), max_seq_len, dim, float(theta))
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    phase = torch.outer(pos, inv).repeat_interleave(2, -1)[None, None]
    out = phase.cos().to(dtype), phase.sin().to(dtype)
    _ROPE_CACHE[key] = out
    return out


def apply_rope(q: torch.Tensor, k: torch.Tensor, cfg: Arch600, offset: int = 0):
    t = q.size(-2)
    c, s = rope_table(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, q.device, q.dtype)
    c = c[..., offset:offset + t, :]
    s = s[..., offset:offset + t, :]
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


class FusedGQA(nn.Module):
    def __init__(self, cfg: Arch600):
        super().__init__()
        self.cfg = cfg
        self.q_size = cfg.d_model
        self.kv_size = cfg.kv_dim
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * cfg.kv_dim, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, 1e-6)
        self.k_norm = RMSNorm(cfg.head_dim, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        packed = self.qkv(x)
        q, k, v = torch.split(packed, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, self.cfg)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))


class FusedSwiGLU(nn.Module):
    def __init__(self, cfg: Arch600):
        super().__init__()
        self.hidden = cfg.ffn_hidden
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: Arch600):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = FusedGQA(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = FusedSwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Neral600M(nn.Module):
    def __init__(self, cfg: Arch600, checkpoint_every: int = 0):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.checkpoint_every = int(checkpoint_every)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        self._scale_residual_outputs()
        actual = sum(p.numel() for p in self.parameters())
        expected = cfg.parameter_breakdown()["total"]
        if actual != expected:
            raise RuntimeError(f"parameter contract mismatch actual={actual:,} expected={expected:,}")

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_outputs(self) -> None:
        # GPT-style residual branch scaling stabilizes pre-norm training as depth grows.
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2 or idx.size(1) <= 0 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid input shape/context")
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            use_ckpt = self.training and self.checkpoint_every > 0 and (i % self.checkpoint_every == 0)
            x = checkpoint(block, x, use_reentrant=False) if use_ckpt else block(x)
        h = self.norm(x)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_report(self) -> dict:
        return self.cfg.parameter_breakdown()


def make_model(candidate: str, *, checkpoint_every: int = 0, max_seq_len: Optional[int] = None) -> Neral600M:
    if candidate not in CANDIDATES:
        raise KeyError(f"unknown candidate {candidate!r}; choices={sorted(CANDIDATES)}")
    cfg = CANDIDATES[candidate]
    if max_seq_len is not None:
        cfg = replace(cfg, max_seq_len=int(max_seq_len))
    return Neral600M(cfg, checkpoint_every=checkpoint_every)
