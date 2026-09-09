from __future__ import annotations

"""Neral 350M Frontier v1.

From-scratch byte-level decoder-only LM derived from Neral Helix principles, but sized
for ~350M parameters and optimized for RTX-class training plus compact mobile inference.
No foreign weights are loaded.
"""

from dataclasses import dataclass, asdict
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


@dataclass(frozen=True)
class Neral350Config:
    vocab_size: int = 260
    d_model: int = 896
    n_layers: int = 30
    n_heads: int = 14
    n_kv_heads: int = 2
    ffn_hidden: int = 2432
    max_seq_len: int = 8192
    rope_theta: float = 250_000.0
    rms_eps: float = 1e-5
    qk_rms_eps: float = 1e-6
    dropout: float = 0.0
    initializer_range: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head dimension")
        if min(self.n_layers, self.ffn_hidden, self.max_seq_len) <= 0:
            raise ValueError("positive dimensions required")

    def to_dict(self) -> dict:
        return asdict(self)


FRONTIER_350 = Neral350Config()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def rope(q: torch.Tensor, k: torch.Tensor, theta: float, offset: int = 0):
    t, d = q.size(-2), q.size(-1)
    pos = torch.arange(offset, offset + t, device=q.device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
    phase = torch.outer(pos, inv).repeat_interleave(2, -1)[None, None]
    c, s = phase.cos().to(q.dtype), phase.sin().to(q.dtype)
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


class GQA(nn.Module):
    def __init__(self, cfg: Neral350Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, cfg.kv_dim, bias=False)
        self.v = nn.Linear(d, cfg.kv_dim, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.qn = RMSNorm(cfg.head_dim, cfg.qk_rms_eps)
        self.kn = RMSNorm(cfg.head_dim, cfg.qk_rms_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        q, k = rope(q, k, self.cfg.rope_theta)
        try:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0, enable_gqa=True)
        except (TypeError, RuntimeError):
            k = k.repeat_interleave(self.cfg.gqa_groups, 1)
            v = v.repeat_interleave(self.cfg.gqa_groups, 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Neral350Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Neral350Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GQA(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Neral350M(nn.Module):
    def __init__(self, cfg: Optional[Neral350Config] = None):
        super().__init__()
        self.cfg = cfg or FRONTIER_350
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm = RMSNorm(self.cfg.d_model, self.cfg.rms_eps)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = False
        self.apply(self._init)
        self._residual_scale()

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, self.cfg.initializer_range)

    def _residual_scale(self) -> None:
        std = self.cfg.initializer_range / math.sqrt(2 * self.cfg.n_layers)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, 0.0, std)
            nn.init.normal_(b.ffn.down.weight, 0.0, std)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.cfg.max_seq_len:
            raise ValueError("invalid input shape/context")
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = activation_checkpoint(b, x, use_reentrant=False) if self.training and self.gradient_checkpointing else b(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "millions": total / 1e6}

    def estimated_kv_cache_bytes(self, sequence_length: int, batch_size: int = 1, bytes_per_value: int = 2) -> int:
        return batch_size * self.cfg.n_layers * 2 * self.cfg.n_kv_heads * sequence_length * self.cfg.head_dim * bytes_per_value
