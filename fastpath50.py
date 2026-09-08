from __future__ import annotations

"""Exact-weights runtime fast path for the existing Neral-50M lineage.

The current 50M model already uses a fused QKV projection. Its remaining easy launch-overhead
win is SwiGLU: legacy blocks call gate(x) and up(x) separately. FusedSwiGLU50 concatenates
those two weight matrices and computes both in one Linear call.

Important resume rule:
- load a legacy checkpoint first;
- fuse only after checkpoint loading;
- do NOT attempt to load a legacy AdamW optimizer state into the fused parameter layout.
Therefore the current interrupted I1 stage remains on the legacy layout. This module is for
inference/evaluation and future stages that intentionally create a fresh optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model50 import Neral50M


class FusedSwiGLU50(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    @classmethod
    def from_legacy(cls, legacy: nn.Module) -> "FusedSwiGLU50":
        if not all(hasattr(legacy, name) for name in ("gate", "up", "down")):
            raise TypeError("expected legacy SwiGLU with gate/up/down")
        d_model = legacy.gate.in_features
        hidden = legacy.gate.out_features
        if legacy.up.in_features != d_model or legacy.up.out_features != hidden:
            raise ValueError("legacy gate/up shapes differ")
        fused = cls(d_model, hidden).to(device=legacy.gate.weight.device, dtype=legacy.gate.weight.dtype)
        with torch.no_grad():
            fused.gate_up.weight.copy_(torch.cat((legacy.gate.weight, legacy.up.weight), dim=0))
            fused.down.weight.copy_(legacy.down.weight)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def fuse_swiglu_inplace(model: Neral50M) -> Neral50M:
    """Fuse all 50M SwiGLU blocks without changing represented weights or parameter count."""
    for block in model.blocks:
        if isinstance(block.mlp, FusedSwiGLU50):
            continue
        block.mlp = FusedSwiGLU50.from_legacy(block.mlp)
    return model


def linear_gemm_report(model: Neral50M) -> dict:
    layers = len(model.blocks)
    legacy = layers * 5  # qkv, out, gate, up, down
    fused = layers * 4   # qkv, out, gate_up, down
    return {
        "layers": layers,
        "legacy_linear_gemm_calls_per_forward": legacy,
        "fused_linear_gemm_calls_per_forward": fused,
        "linear_launch_reduction": 1.0 - fused / legacy,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def verify_equivalence(model: Neral50M, tokens: torch.Tensor, atol: float = 2e-3, rtol: float = 2e-3) -> dict:
    """Compare a legacy model with a fused copy on the same device/dtype.

    BF16 GEMM grouping may change rounding slightly, so this is numerical-equivalence rather
    than bitwise-equivalence. The represented linear weights are exactly preserved.
    """
    import copy

    model.eval()
    fused = copy.deepcopy(model)
    fuse_swiglu_inplace(fused)
    fused.eval()
    a, _ = model(tokens)
    b, _ = fused(tokens)
    max_abs = float((a.float() - b.float()).abs().max())
    ok = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    return {"allclose": ok, "max_abs_logit_delta": max_abs, **linear_gemm_report(fused)}
