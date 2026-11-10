from __future__ import annotations

"""Checkpoint-convertible fast execution layout for Neral 350M Frontier.

The learned function and total parameter count are unchanged. This layout fuses the
Q/K/V projections into one GEMM and the SwiGLU gate/up projections into one GEMM, while
keeping the same GQA/RoPE/RMSNorm/SwiGLU mathematics. It is an experimental execution
layout until the RTX 4070 speed/equivalence gates pass; it must not silently replace the
locked release model.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from model350 import FRONTIER_350, Neral350Config, _rope_table, rotate_half


class FastRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch's native RMSNorm gives torch.compile/Inductor a much cleaner fusion
        # opportunity than the explicit float -> square -> mean -> rsqrt sequence.
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rope_fast(q: torch.Tensor, k: torch.Tensor, theta: float, max_seq_len: int, offset: int = 0):
    t = q.size(-2)
    end = offset + t
    if end > max_seq_len:
        raise ValueError("RoPE position exceeds max_seq_len")
    c, s = _rope_table(max_seq_len, q.size(-1), theta, q.device, q.dtype)
    c = c[..., offset:end, :]
    s = s[..., offset:end, :]
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


class FusedGQA(nn.Module):
    def __init__(self, cfg: Neral350Config):
        super().__init__()
        self.cfg = cfg
        self.q_size = cfg.d_model
        self.kv_size = cfg.kv_dim
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * cfg.kv_dim, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qn = FastRMSNorm(cfg.head_dim, cfg.qk_rms_eps)
        self.kn = FastRMSNorm(cfg.head_dim, cfg.qk_rms_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        q, k = rope_fast(q, k, self.cfg.rope_theta, self.cfg.max_seq_len)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            enable_gqa=True,
        )
        return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))


class FusedSwiGLU(nn.Module):
    def __init__(self, cfg: Neral350Config):
        super().__init__()
        self.hidden = cfg.ffn_hidden
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class FastBlock(nn.Module):
    def __init__(self, cfg: Neral350Config):
        super().__init__()
        self.n1 = FastRMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = FusedGQA(cfg)
        self.n2 = FastRMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = FusedSwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Neral350MFast(nn.Module):
    def __init__(self, cfg: Optional[Neral350Config] = None):
        super().__init__()
        self.cfg = cfg or FRONTIER_350
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([FastBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm = FastRMSNorm(self.cfg.d_model, self.cfg.rms_eps)
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
        for block in self.blocks:
            x = activation_checkpoint(block, x, use_reentrant=False) if self.training and self.gradient_checkpointing else block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "millions": total / 1e6}


def convert_unfused_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert model350.py weights to the fused layout without changing values."""
    out: dict[str, torch.Tensor] = {}
    out["tok_emb.weight"] = state["tok_emb.weight"]
    out["lm_head.weight"] = state.get("lm_head.weight", state["tok_emb.weight"])
    out["norm.weight"] = state["norm.weight"]
    for i in range(FRONTIER_350.n_layers):
        p = f"blocks.{i}."
        out[p + "n1.weight"] = state[p + "n1.weight"]
        out[p + "attn.qkv.weight"] = torch.cat(
            [state[p + "attn.q.weight"], state[p + "attn.k.weight"], state[p + "attn.v.weight"]],
            dim=0,
        )
        out[p + "attn.o.weight"] = state[p + "attn.o.weight"]
        out[p + "attn.qn.weight"] = state[p + "attn.qn.weight"]
        out[p + "attn.kn.weight"] = state[p + "attn.kn.weight"]
        out[p + "n2.weight"] = state[p + "n2.weight"]
        out[p + "ffn.gate_up.weight"] = torch.cat(
            [state[p + "ffn.gate.weight"], state[p + "ffn.up.weight"]], dim=0
        )
        out[p + "ffn.down.weight"] = state[p + "ffn.down.weight"]
    return out


def unfused_parameter_names() -> list[str]:
    names = ["tok_emb.weight"]
    for i in range(FRONTIER_350.n_layers):
        p = f"blocks.{i}."
        names += [
            p + "n1.weight",
            p + "attn.q.weight", p + "attn.k.weight", p + "attn.v.weight",
            p + "attn.o.weight", p + "attn.qn.weight", p + "attn.kn.weight",
            p + "n2.weight",
            p + "ffn.gate.weight", p + "ffn.up.weight", p + "ffn.down.weight",
        ]
    names.append("norm.weight")
    return names


def fused_parameter_names() -> list[str]:
    names = ["tok_emb.weight"]
    for i in range(FRONTIER_350.n_layers):
        p = f"blocks.{i}."
        names += [
            p + "n1.weight", p + "attn.qkv.weight", p + "attn.o.weight",
            p + "attn.qn.weight", p + "attn.kn.weight", p + "n2.weight",
            p + "ffn.gate_up.weight", p + "ffn.down.weight",
        ]
    names.append("norm.weight")
    return names
