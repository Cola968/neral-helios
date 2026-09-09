from __future__ import annotations

"""Experimental, parameter-free attention fast paths for the existing Neral-50M weights.

The trainable architecture is unchanged. Neral-50M has a 52-wide attention head. Padding Q/K/V
with zeros and passing the original 1/sqrt(52) scale is mathematically equivalent to the legacy
attention operation, but different padded widths may dispatch substantially different CUDA SDPA
kernels on Ada. The current production path pads 52->56. This module adds 52->64 and optional
backend forcing so the RTX-4070 runner can choose from measured evidence rather than assumptions.

No candidate is promoted merely because a backend name contains 'flash'. The probe must establish
end-to-end BF16 forward+backward+optimizer throughput, memory and numerical equivalence first.
"""

import copy
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import CausalSelfAttention
from model50 import Neral50M

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - old torch fallback
    SDPBackend = None
    sdpa_kernel = None


class ExactPaddedAttention50(CausalSelfAttention):
    def __init__(self, cfg, *, padded_head_dim: int = 64, backend: str = "auto"):
        super().__init__(cfg)
        if padded_head_dim < self.head_dim or padded_head_dim % 8:
            raise ValueError("padded_head_dim must be >= native head_dim and divisible by 8")
        if backend not in {"auto", "flash", "efficient", "math"}:
            raise ValueError("backend must be auto/flash/efficient/math")
        self.padded_head_dim = int(padded_head_dim)
        self.backend = backend

    def _backend_context(self):
        if self.backend == "auto" or sdpa_kernel is None or SDPBackend is None:
            return nullcontext()
        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        return sdpa_kernel(backends=[mapping[self.backend]])

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        pad = self.padded_head_dim - self.head_dim
        if pad:
            q, k, v = (F.pad(a, (0, pad)) for a in (q, k, v))
        with self._backend_context():
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        return y[..., : self.head_dim]


def use_exact_padded_attention_inplace(
    model: Neral50M,
    *,
    padded_head_dim: int = 64,
    backend: str = "auto",
) -> Neral50M:
    """Replace attention runtime modules while preserving their qkv/out weights exactly."""
    for block in model.blocks:
        old = block.attn
        new = ExactPaddedAttention50(model.cfg, padded_head_dim=padded_head_dim, backend=backend)
        new = new.to(device=old.qkv.weight.device, dtype=old.qkv.weight.dtype)
        new.load_state_dict(old.state_dict(), strict=True)
        block.attn = new
    return model


@torch.no_grad()
def verify_runtime_equivalence(
    model: Neral50M,
    tokens: torch.Tensor,
    *,
    padded_head_dim: int = 64,
    backend: str = "auto",
    atol: float = 3e-3,
    rtol: float = 3e-3,
) -> dict:
    model.eval()
    candidate = copy.deepcopy(model)
    use_exact_padded_attention_inplace(candidate, padded_head_dim=padded_head_dim, backend=backend)
    candidate.eval()
    a, _ = model(tokens)
    b, _ = candidate(tokens)
    af, bf = a.float(), b.float()
    return {
        "allclose": bool(torch.allclose(af, bf, atol=atol, rtol=rtol)),
        "max_abs_logit_delta": float((af - bf).abs().max()),
        "mean_abs_logit_delta": float((af - bf).abs().mean()),
        "native_head_dim": int(model.cfg.head_dim),
        "padded_head_dim": int(padded_head_dim),
        "backend": backend,
        "parameter_count": sum(p.numel() for p in candidate.parameters()),
    }
