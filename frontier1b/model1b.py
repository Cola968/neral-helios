from __future__ import annotations

"""Neral Nova 1B experimental hybrid model.

Weights are initialized from scratch. Mamba-3 is an execution/math dependency only;
Nova does not load any upstream checkpoint, tokenizer, vocabulary, or model weights.
The upstream package is pinned by the runtime workflow and licensed separately under
Apache-2.0.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from architecture1b import CANDIDATES, NOVA_1B, Nova1BConfig, VOCAB_SIZE

_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def _rope_table(max_seq_len: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index, str(dtype), max_seq_len, dim, float(theta))
    hit = _ROPE_CACHE.get(key)
    if hit is not None:
        return hit
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    phase = torch.outer(pos, inv).repeat_interleave(2, -1)[None, None]
    out = phase.cos().to(dtype), phase.sin().to(dtype)
    _ROPE_CACHE[key] = out
    return out


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cfg: Nova1BConfig):
    t = q.size(-2)
    c, s = _rope_table(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, q.device, q.dtype)
    c, s = c[..., :t, :], s[..., :t, :]
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


class SwiGLU(nn.Module):
    def __init__(self, cfg: Nova1BConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class GatedGQA(nn.Module):
    """Exact causal GQA anchor with Q/K norm and a learned per-head output gate."""

    def __init__(self, cfg: Nova1BConfig):
        super().__init__()
        self.cfg = cfg
        self.q_size = cfg.d_model
        self.kv_size = cfg.kv_dim
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model + 2 * cfg.kv_dim, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(cfg.head_dim, eps=1e-6)
        self.head_gate = nn.Linear(cfg.d_model, cfg.n_heads, bias=False) if cfg.gated_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        packed = self.qkv(x)
        q, k, v = torch.split(packed, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.cfg.anchor_position == "rope":
            q, k = _apply_rope(q, k, self.cfg)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0, enable_gqa=True)
        if self.head_gate is not None:
            g = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            y = y * g
        y = y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model)
        return self.out(y)


class Mamba3Mixer(nn.Module):
    """Thin wrapper around the pinned upstream Mamba-3 MIMO mixer."""

    def __init__(self, cfg: Nova1BConfig, layer_idx: int):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba3 import Mamba3
        except Exception as exc:  # pragma: no cover - runtime dependency gate
            raise RuntimeError(
                "Nova Mamba-3 runtime missing. Use the pinned dedicated runtime; "
                "do not silently substitute another mixer."
            ) from exc
        self.mixer = Mamba3(
            d_model=cfg.d_model,
            d_state=cfg.mamba_state,
            expand=cfg.mamba_expand,
            headdim=cfg.head_dim,
            ngroups=cfg.mamba_groups,
            rope_fraction=cfg.mamba_rope_fraction,
            is_outproj_norm=cfg.mamba_outproj_norm,
            is_mimo=True,
            mimo_rank=cfg.mamba_mimo_rank,
            chunk_size=max(8, 64 // cfg.mamba_mimo_rank),
            layer_idx=layer_idx,
            n_layer=cfg.n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class AdaptiveMemoryWeave(nn.Module):
    """Neral-specific low-rank bridge from recurrent stream into an exact anchor.

    AMW is an experimental novelty hypothesis, not a claimed globally novel invention.
    It must beat the no-AMW ablation at matched tokens/FLOPs before promotion.
    """

    def __init__(self, cfg: Nova1BConfig):
        super().__init__()
        self.down = nn.Linear(cfg.d_model, cfg.amw_rank, bias=False)
        self.up = nn.Linear(cfg.amw_rank, cfg.d_model, bias=False)
        self.inject_gate = nn.Parameter(torch.full((cfg.d_model,), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge = self.up(F.silu(self.down(x)))
        return x + torch.sigmoid(self.inject_gate) * bridge


class NovaBlock(nn.Module):
    def __init__(self, cfg: Nova1BConfig, kind: str, layer_idx: int):
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(cfg.d_model)
        self.mix = Mamba3Mixer(cfg, layer_idx) if kind == "mamba3" else GatedGQA(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))


class NeralNova1B(nn.Module):
    def __init__(self, cfg: Nova1BConfig = NOVA_1B, *, checkpoint_blocks: bool = True):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.blocks = nn.ModuleList([NovaBlock(cfg, kind, i) for i, kind in enumerate(cfg.schedule())])
        self.bridges = nn.ModuleList([AdaptiveMemoryWeave(cfg) for _ in range(cfg.cycles)]) if cfg.use_amw else None
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        self._scale_residual_outputs()
        self._verify_parameter_contract()

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_outputs(self) -> None:
        std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            if isinstance(block.mix, GatedGQA):
                nn.init.normal_(block.mix.out.weight, mean=0.0, std=std)
            else:
                out_proj = getattr(block.mix.mixer, "out_proj", None)
                if out_proj is not None:
                    nn.init.normal_(out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=std)

    def _verify_parameter_contract(self) -> None:
        actual = sum(p.numel() for p in self.parameters())
        expected = self.cfg.parameter_breakdown()["total"]
        if actual != expected:
            raise RuntimeError(
                f"Nova instantiated parameter contract mismatch: actual={actual:,}, expected={expected:,}. "
                "Check the pinned Mamba-3 revision before any run."
            )

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if idx.ndim != 2 or idx.size(1) <= 0 or idx.size(1) > self.cfg.max_seq_len:
            raise ValueError("invalid input shape/context")
        x = self.tok_emb(idx)
        cycle = 0
        for i, block in enumerate(self.blocks):
            # AMW is injected immediately before each exact-attention anchor.
            if block.kind == "attention" and self.bridges is not None:
                x = self.bridges[cycle](x)
            if self.training and self.checkpoint_blocks:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if block.kind == "attention":
                cycle += 1
        h = self.norm(x)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_report(self) -> dict[str, int]:
        return self.cfg.parameter_breakdown()


def make_model(candidate: str = NOVA_1B.name, *, checkpoint_blocks: bool = True) -> NeralNova1B:
    if candidate not in CANDIDATES:
        raise KeyError(f"unknown Nova candidate {candidate!r}; choices={sorted(CANDIDATES)}")
    return NeralNova1B(CANDIDATES[candidate], checkpoint_blocks=checkpoint_blocks)
