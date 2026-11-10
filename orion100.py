from __future__ import annotations

"""Neral Orion: hardware-efficient ~100M from-scratch architecture family.

Orion is the model-generation name; Helix remains the architecture research lineage.
No foreign weights are loaded. The implementation targets Ada/RTX 4070 efficiency with:
- fused QKV projection for compact GQA/MQA,
- fused SwiGLU gate+up projection,
- tensor-core-friendly 64-wide attention heads in the performance candidates,
- compact KV caches,
- optional activation checkpointing,
- cached RoPE tables.

The fast candidates are *not* promoted by parameter count or theoretical speed. They must win
the GPU probe plus the frozen quality gates before any expensive pretraining is authorized.
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_ROPE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class OrionConfig:
    vocab_size: int = 260
    d_model: int = 768
    n_layers: int = 14
    n_heads: int = 12
    n_kv_heads: int = 2
    ffn_hidden: int = 2496
    max_seq_len: int = 8192
    rope_theta: float = 100_000.0
    dropout: float = 0.0
    rms_eps: float = 1e-5
    qk_rms_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if min(self.d_model, self.n_layers, self.n_heads, self.n_kv_heads, self.ffn_hidden, self.max_seq_len) <= 0:
            raise ValueError("positive Orion dimensions required")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head_dim")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    def to_dict(self) -> dict:
        return asdict(self)


# A3: throughput-first. 14 large blocks reduce launch depth and use head_dim=64.
ORION_A3 = OrionConfig()
# A4: slightly deeper balance candidate while keeping 64-wide heads and fused projections.
ORION_A4 = OrionConfig(n_layers=16, ffn_hidden=2112)


class OrionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prefer the native kernel when available; preserve a stable FP32 fallback.
        rms_norm = getattr(F, "rms_norm", None)
        if rms_norm is not None:
            return rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _rope_table(length: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index, str(dtype), int(length), int(dim), float(theta))
    cached = _ROPE_TABLES.get(key)
    if cached is not None:
        return cached
    with torch.inference_mode(False), torch.no_grad():
        pos = torch.arange(length, device=device, dtype=torch.float32)
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        phase = torch.outer(pos, inv)
        phase = torch.repeat_interleave(phase, 2, dim=-1)[None, None, :, :]
        result = phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)
    _ROPE_TABLES[key] = result
    return result


def apply_rope(q: torch.Tensor, k: torch.Tensor, cfg: OrionConfig, offset: int = 0):
    t = q.size(-2)
    required = offset + t
    if required > cfg.max_seq_len:
        raise ValueError("Orion RoPE position exceeds max_seq_len")
    cos, sin = _rope_table(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, q.device, q.dtype)
    cos = cos[..., offset:required, :]
    sin = sin[..., offset:required, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class OrionGQA(nn.Module):
    """Fused-projection grouped-query attention.

    Compared with three separate q/k/v Linear calls, one qkv projection reduces launch
    overhead while retaining compact K/V dimensions and the same mathematical operation.
    """

    def __init__(self, cfg: OrionConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv_proj = nn.Linear(cfg.d_model, cfg.qkv_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = OrionRMSNorm(cfg.head_dim, cfg.qk_rms_eps)
        self.k_norm = OrionRMSNorm(cfg.head_dim, cfg.qk_rms_eps)

    def _project(self, x: torch.Tensor, offset: int = 0):
        b, t, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, (self.cfg.q_dim, self.cfg.kv_dim, self.cfg.kv_dim), dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, self.cfg, offset)
        return q, k, v

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
        dropout = self.cfg.dropout if self.training else 0.0
        if q.size(1) == k.size(1):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=causal)
        try:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout, is_causal=causal, enable_gqa=True
            )
        except (TypeError, RuntimeError):
            groups = self.cfg.gqa_groups
            return F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(groups, dim=1),
                v.repeat_interleave(groups, dim=1),
                dropout_p=dropout,
                is_causal=causal,
            )

    def _merge(self, y: torch.Tensor, b: int, t: int):
        y = y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._project(x)
        return self._merge(self._sdpa(q, k, v, True), b, t)

    def forward_cached(self, x: torch.Tensor, past=None):
        b, t, _ = x.shape
        if past is None:
            q, k, v = self._project(x)
            return self._merge(self._sdpa(q, k, v, True), b, t), (k, v)
        if t != 1:
            raise ValueError("cached continuation accepts one byte token")
        pk, pv = past
        if pk.shape != pv.shape or pk.ndim != 4:
            raise ValueError("invalid Orion KV cache")
        offset = pk.size(-2)
        if offset >= self.cfg.max_seq_len:
            raise ValueError("Orion KV cache is full")
        q, kn, vn = self._project(x, offset)
        k = torch.cat((pk, kn), dim=-2)
        v = torch.cat((pv, vn), dim=-2)
        return self._merge(self._sdpa(q, k, v, False), b, 1), (k, v)


class OrionSwiGLU(nn.Module):
    """SwiGLU with one fused gate/up GEMM instead of two separate GEMMs."""

    def __init__(self, cfg: OrionConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class OrionBlock(nn.Module):
    def __init__(self, cfg: OrionConfig):
        super().__init__()
        self.attn_norm = OrionRMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = OrionGQA(cfg)
        self.ffn_norm = OrionRMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = OrionSwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_cached(self, x: torch.Tensor, past=None):
        a, cache = self.attn.forward_cached(self.attn_norm(x), past)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, cache


class NeralOrion(nn.Module):
    def __init__(self, cfg: Optional[OrionConfig] = None):
        super().__init__()
        self.cfg = cfg or ORION_A3
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([OrionBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm = OrionRMSNorm(self.cfg.d_model, self.cfg.rms_eps)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def _scale_residual_projections(self):
        std = self.cfg.initializer_range / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=std)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2 or idx.size(1) <= 0 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid Orion input")
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "millions": total / 1e6}

    def estimated_kv_cache_bytes(self, sequence_length: int, batch_size: int = 1, bytes_per_value: int = 2) -> int:
        if not 0 < sequence_length <= self.cfg.max_seq_len:
            raise ValueError("sequence length outside Orion context")
        return batch_size * self.cfg.n_layers * 2 * self.cfg.n_kv_heads * sequence_length * self.cfg.head_dim * bytes_per_value


def expected_parameters(cfg: OrionConfig) -> int:
    """Closed-form count used to detect silent architecture drift."""
    d, f = cfg.d_model, cfg.ffn_hidden
    attention = d * cfg.qkv_dim + d * d + 2 * cfg.head_dim
    mlp = 3 * d * f
    block_norms = 2 * d
    return cfg.vocab_size * d + cfg.n_layers * (attention + mlp + block_norms) + d


def architecture_report(cfg: OrionConfig) -> dict:
    params = expected_parameters(cfg)
    # Legacy Helix A2 uses 7 linear GEMM calls/block (Q,K,V,out,gate,up,down).
    fused_linear_launches = cfg.n_layers * 4
    legacy_a2_linear_launches = 24 * 7
    return {
        "name": "Neral Orion",
        "parameters": params,
        "millions": params / 1e6,
        "config": cfg.to_dict(),
        "head_dim": cfg.head_dim,
        "qkv_projection_width": cfg.qkv_dim,
        "linear_gemm_calls_per_block": 4,
        "linear_gemm_calls_per_forward": fused_linear_launches,
        "legacy_helix_a2_linear_gemm_calls_per_forward": legacy_a2_linear_launches,
        "linear_launch_reduction_vs_legacy_a2": 1.0 - fused_linear_launches / legacy_a2_linear_launches,
        "kv_cache_bf16_4096_bytes": NeralOrion(cfg).estimated_kv_cache_bytes(4096),
        "kv_cache_bf16_8192_bytes": NeralOrion(cfg).estimated_kv_cache_bytes(8192),
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"a3": architecture_report(ORION_A3), "a4": architecture_report(ORION_A4)}, indent=2))
