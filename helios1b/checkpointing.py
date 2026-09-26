from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _numpy_state_to_safe(state: tuple) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "bit_generator": str(name),
        "keys": np.asarray(keys, dtype=np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_state_from_safe(state: dict[str, Any]) -> tuple:
    return (
        str(state["bit_generator"]),
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_state_to_safe(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(_numpy_state_from_safe(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> str:
    """Atomically replace a checkpoint and write its SHA-256 sidecar.

    Training only calls this after a complete optimizer update. A crash while
    writing cannot replace the previous good checkpoint with a partial file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    sha_tmp = path.with_name(f".{path.name}.sha256.tmp-{os.getpid()}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        torch.save(payload, tmp)
        digest = sha256_file(tmp)
        sha_tmp.write_text(digest + "\n", encoding="ascii")
        os.replace(tmp, path)
        os.replace(sha_tmp, sidecar)
        return digest
    finally:
        for leftover in (tmp, sha_tmp):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    verify_sha256: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if verify_sha256:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.is_file():
            expected = sidecar.read_text(encoding="ascii").strip()
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"checkpoint SHA-256 mismatch for {path}: "
                    f"expected {expected}, got {actual}"
                )

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a dictionary")
    if int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Helios checkpoint schema")
    return payload
