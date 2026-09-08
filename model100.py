from __future__ import annotations

"""Neral-100M Helix: a from-scratch small-LLM architecture.

This is intentionally separate from the legacy 20M/40M/50M implementation.  It keeps
Neral's own byte vocabulary, but reallocates the parameter budget toward a deep/thin
backbone and grouped-query attention (GQA).  No foreign weights are loaded or assumed.
"""

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_ROPE100_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class Neral100Config:
    vocab_size: int = 260
    d_model: int = 512
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_hidden: int = 1600
    max_seq_len: int = 8192
    rope_theta: float = 100_000.0
    dropout: float = 0.0
    rms_eps: float = 1e-5
    qk_rms_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.ffn_hidden <= 0:
            raise ValueError("positive Helix dimensions required")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("positive attention head counts required")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.max_seq_len <= 0:
            raise ValueError("positive max_seq_len required")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)


HELIX_100 = Neral100Config()


class HelixRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _rope_table(length: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index, str(dtype), int(length), int(dim), float(theta))
    cached = _ROPE100_TABLES.get(key)
    if cached is not None:
        return cached
    with torch.inference_mode(False), torch.no_grad():
        pos = torch.arange(length, device=device, dtype=torch.float32)
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        phase = torch.outer(pos, inv)
        phase = torch.repeat_interleave(phase, 2, dim=-1)[None, None, :, :]
        result = phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)
    _ROPE100_TABLES[key] = result
    return result


def apply_helix_rope(q: torch.Tensor, k: torch.Tensor, theta: float, offset: int, max_seq_len: int):
    t = q.size(-2)
    required = offset + t
    if required > max_seq_len:
        raise ValueError("Helix RoPE position exceeds max_seq_len")
    cos, sin = _rope_table(max_seq_len, q.size(-1), theta, q.device, q.dtype)
    cos = cos[..., offset:required, :]
    sin = sin[..., offset:required, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class HelixGQA(nn.Module):
    """Grouped-query causal attention with compact K/V projections and cache."""

    def __init__(self, cfg: Neral100Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(d, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(d, cfg.kv_dim, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        # Shared-across-heads Q/K RMS scales add almost no parameters and stabilize deep attention.
        self.q_norm = HelixRMSNorm(cfg.head_dim, cfg.qk_rms_eps)
        self.k_norm = HelixRMSNorm(cfg.head_dim, cfg.qk_rms_eps)

    def _project(self, x: torch.Tensor, offset: int = 0):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_helix_rope(q, k, self.cfg.rope_theta, offset, self.cfg.max_seq_len)
        return q, k, v

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
        dropout = self.cfg.dropout if self.training else 0.0
        if q.size(1) == k.size(1):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=causal)
        # Newer PyTorch builds perform GQA without materialising repeated K/V heads.
        try:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout, is_causal=causal, enable_gqa=True
            )
        except (TypeError, RuntimeError):
            # Compatibility fallback. The stored KV cache remains compact even if an older
            # runtime needs temporary repeated views for the attention kernel.
            groups = self.cfg.gqa_groups
            kr = k.repeat_interleave(groups, dim=1)
            vr = v.repeat_interleave(groups, dim=1)
            return F.scaled_dot_product_attention(q, kr, vr, dropout_p=dropout, is_causal=causal)

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
            q, k, v = self._project(x, 0)
            y = self._sdpa(q, k, v, True)
            return self._merge(y, b, t), (k, v)
        if t != 1:
            raise ValueError("cached continuation accepts exactly one byte token")
        pk, pv = past
        expected = (b, self.cfg.n_kv_heads, pk.size(-2), self.cfg.head_dim)
        if pk.shape != expected or pv.shape != expected:
            raise ValueError("invalid Helix compact KV cache")
        offset = pk.size(-2)
        if offset >= self.cfg.max_seq_len:
            raise ValueError("Helix KV cache is full")
        q, kn, vn = self._project(x, offset)
        k = torch.cat((pk, kn), dim=-2)
        v = torch.cat((pv, vn), dim=-2)
        y = self._sdpa(q, k, v, False)
        return self._merge(y, b, 1), (k, v)


class HelixSwiGLU(nn.Module):
    def __init__(self, cfg: Neral100Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HelixBlock(nn.Module):
    def __init__(self, cfg: Neral100Config):
        super().__init__()
        self.attn_norm = HelixRMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = HelixGQA(cfg)
        self.ffn_norm = HelixRMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = HelixSwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_cached(self, x: torch.Tensor, past=None):
        a, cache = self.attn.forward_cached(self.attn_norm(x), past)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, cache


class Neral100M(nn.Module):
    def __init__(self, cfg: Optional[Neral100Config] = None):
        super().__init__()
        self.cfg = cfg or HELIX_100
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([HelixBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm = HelixRMSNorm(self.cfg.d_model, self.cfg.rms_eps)
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
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch,time]")
        if idx.size(1) <= 0 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid Helix sequence length")
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

    def forward_cached(self, idx: torch.Tensor, past_key_values=None):
        if idx.ndim != 2 or idx.size(1) <= 0 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid cached Helix input")
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks) or idx.size(1) != 1:
                raise ValueError("invalid cached continuation")
        x = self.tok_emb(idx)
        caches = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, cache = block.forward_cached(x, past)
            caches.append(cache)
        x = self.final_norm(x)
        return self.lm_head(x), caches

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: int):
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), 1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: Optional[int] = 2,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        if max_new_tokens <= 0:
            return idx
        if not use_cache:
            for _ in range(max_new_tokens):
                ctx = idx[:, -self.cfg.max_seq_len:]
                logits, _ = self(ctx)
                nxt = self._sample_next(logits[:, -1], temperature, top_k)
                idx = torch.cat((idx, nxt), dim=1)
                if eos_id is not None and bool((nxt == eos_id).all()):
                    break
            return idx
        ctx = idx[:, -self.cfg.max_seq_len:]
        logits, cache = self.forward_cached(ctx)
        for step in range(max_new_tokens):
            nxt = self._sample_next(logits[:, -1], temperature, top_k)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if step + 1 >= max_new_tokens:
                break
            if cache[0][0].size(-2) >= self.cfg.max_seq_len:
                logits, cache = self.forward_cached(idx[:, -self.cfg.max_seq_len:])
            else:
                logits, cache = self.forward_cached(nxt, cache)
        return idx

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "millions": total / 1e6}

    def estimated_kv_cache_bytes(self, sequence_length: int, batch_size: int = 1, bytes_per_value: int = 2) -> int:
        if not 0 < sequence_length <= self.cfg.max_seq_len:
            raise ValueError("sequence_length outside Helix context")
        return (
            batch_size
            * self.cfg.n_layers
            * 2
            * self.cfg.n_kv_heads
            * sequence_length
            * self.cfg.head_dim
            * bytes_per_value
        )
