from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Helios180Config

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)

class RotaryCache(nn.Module):
    """Compute RoPE once per model forward, not once per layer."""
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv = self.inv_freq.to(device=device)
        freqs = torch.outer(pos, inv)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)[None, None, :, :]
        return emb.cos().to(dtype), emb.sin().to(dtype)

def apply_rope(q, k, cos, sin):
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: Helios180Config):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps) if cfg.qk_norm else nn.Identity()

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        if self.cfg.native_gqa:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                enable_gqa=True,
            )
        else:
            n = self.cfg.groups_per_kv_head
            y = F.scaled_dot_product_attention(
                q, k.repeat_interleave(n, dim=1), v.repeat_interleave(n, dim=1),
                is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        return self.out_proj(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))

@dataclass
class Helios180Output:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    cross_entropy: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None

class Helios180ForCausalLM(nn.Module):
    def __init__(self, cfg: Optional[Helios180Config] = None):
        super().__init__()
        self.cfg = cfg or Helios180Config()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.rope = RotaryCache(self.cfg.head_dim, self.cfg.rope_theta)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm = RMSNorm(self.cfg.d_model, self.cfg.rms_eps)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        self._scale_residuals()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _scale_residuals(self):
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for b in self.blocks:
            nn.init.normal_(b.attn.out_proj.weight, 0.0, std)
            nn.init.normal_(b.mlp.down.weight, 0.0, std)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch,time]")
        if input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = self.tok_emb(input_ids)
        cos, sin = self.rope(input_ids.size(1), x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return Helios180Output(logits)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        z = torch.logsumexp(logits.float(), dim=-1).square().mean() if self.cfg.z_loss_coeff > 0 else logits.new_zeros(())
        return Helios180Output(logits, ce + self.cfg.z_loss_coeff * z, ce, z)

    def parameter_report(self):
        actual = sum(p.numel() for p in self.parameters())
        return {"actual": actual, "analytical": self.cfg.parameter_count(), "millions": actual / 1e6}
