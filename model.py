from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NeralConfig


# Parameter-free runtime cache. Tables are shared by compatible model instances and are
# deliberately excluded from state_dict/checkpoints. This avoids rebuilding sin/cos tables
# on every decode step without changing any trainable parameter or model weight.
_ROPE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rope_table(
    length: int,
    dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length <= 0:
        raise ValueError("RoPE length must be positive")
    key = (device.type, device.index, str(dtype), int(dim), float(theta), int(length))
    cached = _ROPE_TABLES.get(key)
    if cached is not None:
        return cached
    pos = torch.arange(length, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)[None, None, :, :]
    value = (emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype))
    _ROPE_TABLES[key] = value
    return value


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    theta: float,
    position_offset: int = 0,
    table_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k with an optional absolute position offset.

    During training ``position_offset`` is zero. Cached decoding passes the number of
    already-cached tokens so the new token receives exactly the same RoPE position it would
    have received during a full forward pass.
    """
    t = q.size(-2)
    d = q.size(-1)
    if position_offset < 0:
        raise ValueError("position_offset must be non-negative")
    required = position_offset + t
    length = max(required, table_length or required)
    cos, sin = _rope_table(length, d, theta, q.device, q.dtype)
    cos = cos[..., position_offset:required, :]
    sin = sin[..., position_offset:required, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NeralConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.theta = cfg.rope_theta
        self.dropout = cfg.dropout
        self.max_seq_len = cfg.max_seq_len
        d = cfg.d_model
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def _project(self, x: torch.Tensor, position_offset: int = 0):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.theta, position_offset, self.max_seq_len)
        return q, k, v

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    def _merge(self, y: torch.Tensor, batch: int, time: int, channels: int):
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self._project(x)
        return self._merge(self._sdpa(q, k, v, True), b, t, c)

    def forward_cached(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward with a KV cache for autoregressive inference.

        The initial call may contain a full prompt. Follow-up calls are intentionally
        restricted to one token. With a single query token all concatenated keys are valid
        past/current positions, so SDPA must *not* use its upper-left causal mask.
        """
        b, t, c = x.shape
        if past is None:
            q, k, v = self._project(x, 0)
            y = self._sdpa(q, k, v, True)
            return self._merge(y, b, t, c), (k, v)
        if t != 1:
            raise ValueError("Cached continuation accepts exactly one token per step")
        past_k, past_v = past
        if past_k.ndim != 4 or past_v.ndim != 4 or past_k.shape != past_v.shape:
            raise ValueError("Invalid KV cache")
        if past_k.size(0) != b or past_k.size(1) != self.n_heads or past_k.size(-1) != self.head_dim:
            raise ValueError("KV cache shape does not match attention module")
        offset = past_k.size(-2)
        if offset >= self.max_seq_len:
            raise ValueError("KV cache is full; rebuild from the sliding context window")
        q, k_new, v_new = self._project(x, offset)
        k = torch.cat((past_k, k_new), dim=-2)
        v = torch.cat((past_v, v_new), dim=-2)
        y = self._sdpa(q, k, v, False)
        return self._merge(y, b, 1, c), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: NeralConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: NeralConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_cached(self, x: torch.Tensor, past=None):
        a, cache = self.attn.forward_cached(self.norm1(x), past)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, cache


class Neral20M(nn.Module):
    def __init__(self, cfg: Optional[NeralConfig] = None):
        super().__init__()
        self.cfg = cfg or NeralConfig()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm = RMSNorm(self.cfg.d_model, self.cfg.rms_eps)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, time]")
        if idx.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {idx.size(1)} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_cached(self, idx: torch.Tensor, past_key_values=None):
        """Inference-only forward returning per-layer KV state.

        No cache tensor is a parameter or checkpoint field. The method is deliberately
        separate from ``forward`` so training semantics remain unchanged.
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, time]")
        if idx.size(1) <= 0 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid cached-forward sequence length")
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("KV cache layer count mismatch")
            if idx.size(1) != 1:
                raise ValueError("Cached continuation accepts one token")
        x = self.tok_emb(idx)
        new_cache = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, cache = block.forward_cached(x, past)
            new_cache.append(cache)
        x = self.final_norm(x)
        return self.lm_head(x), new_cache

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: int):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

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
                next_id = self._sample_next(logits[:, -1, :], temperature, top_k)
                idx = torch.cat((idx, next_id), dim=1)
                if eos_id is not None and bool((next_id == eos_id).all()):
                    break
            return idx

        ctx = idx[:, -self.cfg.max_seq_len:]
        logits, cache = self.forward_cached(ctx)
        for step in range(max_new_tokens):
            next_id = self._sample_next(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
            if step + 1 >= max_new_tokens:
                break
            cache_len = cache[0][0].size(-2) if cache else 0
            if cache_len >= self.cfg.max_seq_len:
                # Preserve the legacy sliding-window semantics exactly when the cache fills.
                ctx = idx[:, -self.cfg.max_seq_len:]
                logits, cache = self.forward_cached(ctx)
            else:
                logits, cache = self.forward_cached(next_id, cache)
        return idx

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "millions": total / 1e6}
