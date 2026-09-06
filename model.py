from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NeralConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype=x.dtype) * self.weight)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [B, H, T, D]
    t = q.size(-2)
    d = q.size(-1)
    device = q.device
    pos = torch.arange(t, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)[None, None, :, :]
    cos = emb.cos().to(dtype=q.dtype)
    sin = emb.sin().to(dtype=q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NeralConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.theta = cfg.rope_theta
        self.dropout = cfg.dropout
        d = cfg.d_model
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.theta)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(y)


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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_id: Optional[int] = 2,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    cutoff = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "millions": total / 1e6}
