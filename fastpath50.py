from __future__ import annotations

"""Exact-weights runtime/training fast paths for the existing Neral-50M lineage.

The current 50M model already uses a fused QKV projection. Its remaining easy launch-overhead
win is SwiGLU: legacy blocks call gate(x) and up(x) separately. FusedSwiGLU50 concatenates
those two weight matrices and computes both in one Linear call.

Important resume rule:
- load a legacy checkpoint first;
- fuse only after checkpoint loading;
- do NOT attempt to load a legacy AdamW optimizer state into the fused parameter layout.
The current interrupted I1 stage therefore remains on the legacy layout. Chat V2 begins only
after D3 and creates a fresh optimizer, so it may use the fused representation safely.

A fused Chat-V2 checkpoint can be exported back to the canonical legacy state-dict key layout.
The represented weights are unchanged, which keeps the normal Neral benchmark/runtime loaders
usable even though training used fewer GEMM launches.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RMSNorm
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


class NativeRMSNorm50(nn.Module):
    """RMSNorm with the same single weight parameter and state-dict key as legacy RMSNorm."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    @classmethod
    def from_legacy(cls, norm: nn.Module) -> "NativeRMSNorm50":
        if not hasattr(norm, "weight") or not hasattr(norm, "eps"):
            raise TypeError("expected RMSNorm-compatible module")
        out = cls(norm.weight.numel(), norm.eps).to(device=norm.weight.device, dtype=norm.weight.dtype)
        with torch.no_grad():
            out.weight.copy_(norm.weight)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = getattr(F, "rms_norm", None)
        if native is not None:
            return native(x, (x.shape[-1],), self.weight, self.eps)
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight


def fuse_swiglu_inplace(model: Neral50M) -> Neral50M:
    """Fuse all 50M SwiGLU blocks without changing represented weights or parameter count."""
    for block in model.blocks:
        if isinstance(block.mlp, FusedSwiGLU50):
            continue
        block.mlp = FusedSwiGLU50.from_legacy(block.mlp)
    return model


def use_native_rmsnorm_inplace(model: Neral50M) -> Neral50M:
    for block in model.blocks:
        if not isinstance(block.norm1, NativeRMSNorm50):
            block.norm1 = NativeRMSNorm50.from_legacy(block.norm1)
        if not isinstance(block.norm2, NativeRMSNorm50):
            block.norm2 = NativeRMSNorm50.from_legacy(block.norm2)
    if not isinstance(model.final_norm, NativeRMSNorm50):
        model.final_norm = NativeRMSNorm50.from_legacy(model.final_norm)
    return model


def upgrade_runtime_inplace(model: Neral50M, *, fused_mlp: bool = True, native_rmsnorm: bool = True) -> Neral50M:
    if fused_mlp:
        fuse_swiglu_inplace(model)
    if native_rmsnorm:
        use_native_rmsnorm_inplace(model)
    return model


def canonical_legacy_state_dict(model: Neral50M) -> dict[str, torch.Tensor]:
    """Return canonical legacy Neral-50M state keys from either legacy or fused runtime layout."""
    state = model.state_dict()
    if not any(".mlp.gate_up.weight" in key for key in state):
        return {k: v.detach().clone() for k, v in state.items()}
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if ".mlp.gate_up.weight" not in key:
            out[key] = value.detach().clone()
            continue
        prefix = key.removesuffix("gate_up.weight")
        hidden = value.shape[0] // 2
        if value.shape[0] != hidden * 2:
            raise ValueError("invalid fused gate_up shape")
        out[prefix + "gate.weight"] = value[:hidden].detach().clone()
        out[prefix + "up.weight"] = value[hidden:].detach().clone()
    return out


def load_fused_state_dict(model: Neral50M, state: dict[str, torch.Tensor]) -> Neral50M:
    """Upgrade a fresh 50M model and load a previously saved fused Chat-V2 state dict."""
    upgrade_runtime_inplace(model, fused_mlp=True, native_rmsnorm=True)
    model.load_state_dict(state, strict=True)
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
    """Compare a legacy model with a fused/native-runtime copy on the same device/dtype."""
    model.eval()
    fused = copy.deepcopy(model)
    upgrade_runtime_inplace(fused)
    fused.eval()
    a, _ = model(tokens)
    b, _ = fused(tokens)
    max_abs = float((a.float() - b.float()).abs().max())
    ok = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    return {"allclose": ok, "max_abs_logit_delta": max_abs, **linear_gemm_report(fused)}
