from __future__ import annotations

"""Arctus Prime 350M research candidate.

This is a parameter-matched, quality-oriented successor candidate to Frontier 350 v1.
It is intentionally isolated from the production/checkpoint-compatible v1 path.
No foreign weights, tokenizer vocabulary, or merges are used.

Design goals:
- stay at ~350M parameters;
- keep Tensor-Core-friendly head_dim=64;
- increase transformer depth from 24 -> 28;
- relax aggressive 9:1 GQA to 4:1 GQA;
- keep Neral's locked 16K BPE, Pre-RMSNorm, Q/K RMSNorm, RoPE and SwiGLU;
- use fused QKV and gate/up execution layouts from day one.

IMPORTANT: this model is NOT shape-compatible with existing Frontier 350 v1 checkpoints.
It must win a matched architecture bake-off before any promotion is considered.
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class ArctusPrime350Config:
    vocab_size: int = 16_384
    d_model: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_hidden: int = 3008
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
        if self.head_dim != 64:
            raise ValueError("Prime 350 contract requires head_dim=64")
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head dimension")
        if min(self.n_layers, self.ffn_hidden, self.max_seq_len) <= 0:
            raise ValueError("positive dimensions required")

    def to_dict(self) -> dict:
        return asdict(self)

    def analytical_parameter_count(self) -> int:
        # Tied token embedding / LM head means vocab*d is counted once.
        d = self.d_model
        hd = self.head_dim
        kv = self.kv_dim
        per_layer = (
            2 * d * d               # Q + output projection
            + 2 * d * kv            # K + V projections
            + 3 * d * self.ffn_hidden  # SwiGLU gate/up/down
            + 2 * d                 # block RMSNorm weights
            + 2 * hd                # Q/K RMSNorm weights
        )
        return self.vocab_size * d + self.n_layers * per_layer + d


# Exact analytical count: 348,975,616 parameters.
PRIME_350 = ArctusPrime350Config()
PRIME_350_PARAMETER_COUNT = 348_975_616
assert PRIME_350.analytical_parameter_count() == PRIME_350_PARAMETER_COUNT


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def _rope_table(max_seq_len: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index, str(dtype), int(max_seq_len), int(dim), float(theta))
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        phase = torch.outer(pos, inv).repeat_interleave(2, -1)[None, None]
        result = phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)
    _ROPE_CACHE[key] = result
    return result


def rope(q: torch.Tensor, k: torch.Tensor, theta: float, max_seq_len: int, offset: int = 0):
    t = q.size(-2)
    end = offset + t
    if end > max_seq_len:
        raise ValueError("RoPE position exceeds max_seq_len")
    c, s = _rope_table(max_seq_len, q.size(-1), theta, q.device, q.dtype)
    c = c[..., offset:end, :]
    s = s[..., offset:end, :]
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


class FusedGQA(nn.Module):
    def __init__(self, cfg: ArctusPrime350Config):
        super().__init__()
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
        q, k = rope(q, k, self.cfg.rope_theta, self.cfg.max_seq_len)
        try:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                enable_gqa=True,
            )
        except (TypeError, RuntimeError):
            k = k.repeat_interleave(self.cfg.gqa_groups, 1)
            v = v.repeat_interleave(self.cfg.gqa_groups, 1)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))


class FusedSwiGLU(nn.Module):
    def __init__(self, cfg: ArctusPrime350Config):
        super().__init__()
        self.hidden = cfg.ffn_hidden
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class PrimeBlock(nn.Module):
    def __init__(self, cfg: ArctusPrime350Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = FusedGQA(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = FusedSwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class ArctusPrime350M(nn.Module):
    def __init__(self, cfg: Optional[ArctusPrime350Config] = None):
        super().__init__()
        self.cfg = cfg or PRIME_350
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([PrimeBlock(self.cfg) for _ in range(self.cfg.n_layers)])
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
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight, 0.0, std)
            nn.init.normal_(block.ffn.down.weight, 0.0, std)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2 or not 0 < idx.size(1) <= self.cfg.max_seq_len:
            raise ValueError("invalid input shape/context")
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = (
                activation_checkpoint(block, x, use_reentrant=False)
                if self.training and self.gradient_checkpointing
                else block(x)
            )
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "millions": total / 1e6,
            "analytical": self.cfg.analytical_parameter_count(),
        }

    def estimated_kv_cache_bytes(
        self,
        sequence_length: int,
        batch_size: int = 1,
        bytes_per_value: int = 2,
    ) -> int:
        return (
            batch_size
            * self.cfg.n_layers
            * 2
            * self.cfg.n_kv_heads
            * sequence_length
            * self.cfg.head_dim
            * bytes_per_value
        )
