from __future__ import annotations

"""Checkpoint-compatible Arctus execution variant using PyTorch native RMSNorm.

This keeps the locked Frontier 350M architecture, separate Q/K/V projections and separate
SwiGLU gate/up projections exactly as in model350.py. Only the RMSNorm implementation is
swapped from the manual fp32 formulation to torch.nn.functional.rms_norm so Stage-3 can
measure that change in isolation after the fused QKV/SwiGLU path failed to improve speed.

State-dict key names and parameter shapes stay compatible with Neral350M. Numerical
promotion still requires a dedicated equivalence/quality gate because native kernels may
round differently from the reference implementation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model350 import FRONTIER_350, Neral350Config, Neral350M


class NativeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Neral350MNativeNorm(Neral350M):
    """Locked Arctus model with native RMSNorm and otherwise unchanged math/layout."""

    def __init__(self, cfg: Optional[Neral350Config] = None):
        super().__init__(cfg or FRONTIER_350)
        c = self.cfg
        self.norm = NativeRMSNorm(c.d_model, c.rms_eps)
        for block in self.blocks:
            block.n1 = NativeRMSNorm(c.d_model, c.rms_eps)
            block.n2 = NativeRMSNorm(c.d_model, c.rms_eps)
            block.attn.qn = NativeRMSNorm(c.head_dim, c.qk_rms_eps)
            block.attn.kn = NativeRMSNorm(c.head_dim, c.qk_rms_eps)
