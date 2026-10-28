"""Restricted deserialization of Neral-owned legacy Torch checkpoints.

Older 350M foundation resumes store NumPy's MT19937 RNG array. A plain
weights_only=True load rejects that array; unrestricted pickle isn't required.
Only NumPy array/dtype primitives are allowed. Architecture/tokenizer identity
is validated separately by the existing 350M contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: str | Path) -> dict:
    try:
        from numpy._core.multiarray import _reconstruct, scalar
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import _reconstruct, scalar

    allowed = [
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        (scalar, "numpy._core.multiarray.scalar"),
        (scalar, "numpy.core.multiarray.scalar"),
        np.ndarray,
        np.dtype,
        type(np.dtype("uint32")),
        type(np.dtype("int64")),
        type(np.dtype("float64")),
        type(np.dtype("bool")),
    ]
    # An explicitly scoped allowlist does not change future unpickling policy.
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Neral checkpoint must contain a model tensor dictionary")
    return payload

