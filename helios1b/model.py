from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import HeliosConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, dim = q.size(-2), q.size(-1)
    if dim % 2:
        raise ValueError("RoPE head_dim must be even")
    positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)[None, None, :, :]
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: HeliosConfig, layer_index: int):
        super().__init__()
        self.cfg = cfg
        self.use_rope = cfg.layer_uses_rope(layer_index)
        self.q_proj = nn.Linear(
            cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False
        )
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.out_proj = nn.Linear(
            cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False
        )
        self.q_norm = (
            RMSNorm(cfg.head_dim, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(cfg.head_dim, cfg.rms_eps) if cfg.qk_norm else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(
            b, t, self.cfg.n_heads, self.cfg.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            b, t, self.cfg.n_kv_heads, self.cfg.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            b, t, self.cfg.n_kv_heads, self.cfg.head_dim
        ).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, k, self.cfg.rope_theta)

        if self.cfg.groups_per_kv_head > 1:
            k = k.repeat_interleave(self.cfg.groups_per_kv_head, dim=1)
            v = v.repeat_interleave(self.cfg.groups_per_kv_head, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(
            b, t, self.cfg.n_heads * self.cfg.head_dim
        )
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: HeliosConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HeliosBlock(nn.Module):
    def __init__(self, cfg: HeliosConfig, layer_index: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GroupedQueryAttention(cfg, layer_index)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


@dataclass
class HeliosOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    cross_entropy: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None


class HeliosForCausalLM(nn.Module):
    def __init__(self, cfg: Optional[HeliosConfig] = None):
        super().__init__()
        self.cfg = cfg or HeliosConfig()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            [HeliosBlock(self.cfg, i) for i in range(self.cfg.n_layers)]
        )
        self.final_norm = RMSNorm(self.cfg.d_model, self.cfg.rms_eps)
        self.lm_head = nn.Linear(
            self.cfg.d_model, self.cfg.vocab_size, bias=False
        )
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        self._scale_residual_projections()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=std)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> HeliosOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")

        x = self.tok_emb(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return HeliosOutput(logits=logits)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        if self.cfg.z_loss_coeff > 0:
            z = torch.logsumexp(logits.float(), dim=-1).square().mean()
            loss = ce + self.cfg.z_loss_coeff * z
        else:
            z = logits.new_zeros(())
            loss = ce

        return HeliosOutput(
            logits=logits,
            loss=loss,
            cross_entropy=ce,
            z_loss=z,
        )

    def parameter_report(self) -> dict:
        actual = sum(p.numel() for p in self.parameters())
        return {
            "actual": actual,
            "analytical": self.cfg.parameter_count(),
            "millions": actual / 1e6,
        }
