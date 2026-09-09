from __future__ import annotations

"""Crash-safe local checkpoint IO contract for future Neral training phases.

This module does not upload checkpoints and never trusts an unverified external checkpoint.
It is designed to wrap 50M/ORION trainers incrementally without changing model mathematics.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

FORMAT = "neral-resume-v3"
REQUIRED_META = {
    "step", "token_count", "curriculum_phase", "dataset_cursor", "config_hash",
    "architecture_hash", "source_commit", "dataset_manifest_hash",
}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_resume_metadata(meta: dict) -> dict:
    if not isinstance(meta, dict):
        raise ValueError("resume metadata must be a dict")
    missing = sorted(REQUIRED_META - set(meta))
    if missing:
        raise ValueError("missing resume metadata: " + ", ".join(missing))
    if int(meta["step"]) < 0 or int(meta["token_count"]) < 0:
        raise ValueError("negative resume counters")
    if not isinstance(meta["dataset_cursor"], dict):
        raise ValueError("dataset_cursor must be structured")
    for key in ("curriculum_phase", "config_hash", "architecture_hash", "source_commit", "dataset_manifest_hash"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"invalid {key}")
    if len(meta["config_hash"]) < 32 or len(meta["architecture_hash"]) < 32 or len(meta["dataset_manifest_hash"]) < 32:
        raise ValueError("resume hashes look truncated")
    return meta


def capture_rng_state() -> dict:
    state: dict[str, Any] = {"python_random": random.getstate()}
    try:
        import numpy as np
        state["numpy_random"] = np.random.get_state()
    except Exception:
        state["numpy_random"] = None
    try:
        import torch
        state["torch_cpu"] = torch.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except Exception:
        state["torch_cpu"] = None
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: dict) -> None:
    if not isinstance(state, dict):
        raise ValueError("rng state missing")
    random.setstate(state["python_random"])
    if state.get("numpy_random") is not None:
        import numpy as np
        np.random.set_state(state["numpy_random"])
    if state.get("torch_cpu") is not None:
        import torch
        torch.set_rng_state(state["torch_cpu"])
        cuda = state.get("torch_cuda") or []
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all(cuda)


def make_resume_payload(
    *,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    step: int,
    token_count: int,
    curriculum_phase: str,
    dataset_cursor: dict,
    config: dict,
    architecture_contract: dict,
    source_commit: str,
    dataset_manifest: dict,
    rng_state: dict | None = None,
    extra: dict | None = None,
) -> dict:
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("model state required")
    if not isinstance(optimizer_state, dict):
        raise ValueError("optimizer state required for exact resume")
    if not isinstance(scheduler_state, dict):
        raise ValueError("scheduler state required for exact resume")
    meta = {
        "step": int(step),
        "token_count": int(token_count),
        "curriculum_phase": curriculum_phase,
        "dataset_cursor": dataset_cursor,
        "config_hash": stable_hash(config),
        "architecture_hash": stable_hash(architecture_contract),
        "source_commit": source_commit,
        "dataset_manifest_hash": stable_hash(dataset_manifest),
    }
    validate_resume_metadata(meta)
    return {
        "format": FORMAT,
        "metadata": meta,
        "config": config,
        "architecture_contract": architecture_contract,
        "dataset_manifest": dataset_manifest,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "rng": rng_state if rng_state is not None else capture_rng_state(),
        "extra": extra or {},
    }


def validate_resume_payload(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("unsupported resume payload")
    meta = validate_resume_metadata(payload.get("metadata"))
    for key in ("config", "architecture_contract", "dataset_manifest", "model", "optimizer", "scheduler", "rng"):
        if key not in payload:
            raise ValueError(f"resume payload missing {key}")
    if stable_hash(payload["config"]) != meta["config_hash"]:
        raise ValueError("config hash mismatch")
    if stable_hash(payload["architecture_contract"]) != meta["architecture_hash"]:
        raise ValueError("architecture hash mismatch")
    if stable_hash(payload["dataset_manifest"]) != meta["dataset_manifest_hash"]:
        raise ValueError("dataset manifest hash mismatch")
    return meta


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_write_json(path: Path, value: dict) -> None:
    _atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf8"))


def save_checkpoint_atomic(payload: dict, path: str | Path) -> dict:
    """Serialize to a same-directory temp file, fsync, verify, rename, then advance last_good.

    The torch pickle is only ever loaded after its SHA-256 sidecar has been verified. This does not
    make pickle safe for untrusted input; callers must keep checkpoints local/trusted.
    """
    validate_resume_payload(payload)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to serialize Neral checkpoints") from exc

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".partial-", suffix=".pt", dir=str(target.parent))
    os.close(fd)
    temp = Path(temp_name)
    try:
        torch.save(payload, temp)
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        digest = sha256_file(temp)
        if temp.stat().st_size <= 0:
            raise IOError("empty checkpoint")
        os.replace(temp, target)
        final_digest = sha256_file(target)
        if final_digest != digest:
            raise IOError("checkpoint hash changed during atomic rename")
        sidecar = target.with_suffix(target.suffix + ".sha256.json")
        _atomic_write_json(sidecar, {"sha256": digest, "bytes": target.stat().st_size, "format": FORMAT})
        pointer = target.parent / "last_good_checkpoint.json"
        _atomic_write_json(pointer, {"path": target.name, "sha256": digest, "step": payload["metadata"]["step"], "token_count": payload["metadata"]["token_count"]})
        return {"path": str(target), "sha256": digest, "bytes": target.stat().st_size}
    finally:
        if temp.exists():
            temp.unlink()


def verify_checkpoint_file(path: str | Path) -> dict:
    target = Path(path).resolve()
    sidecar = target.with_suffix(target.suffix + ".sha256.json")
    if not target.is_file() or not sidecar.is_file():
        raise FileNotFoundError("checkpoint or checksum sidecar missing")
    meta = json.loads(sidecar.read_text(encoding="utf8"))
    actual = sha256_file(target)
    if actual != meta.get("sha256"):
        raise ValueError("checkpoint SHA-256 mismatch")
    if int(meta.get("bytes", -1)) != target.stat().st_size:
        raise ValueError("checkpoint size mismatch")
    return {"path": str(target), "sha256": actual, "bytes": target.stat().st_size}


def load_verified_checkpoint(path: str | Path, *, trusted_local: bool = False, map_location: str = "cpu") -> dict:
    if not trusted_local:
        raise ValueError("refusing pickle deserialization without trusted_local=True")
    verify_checkpoint_file(path)
    import torch
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    validate_resume_payload(payload)
    return payload


def resolve_last_good(directory: str | Path) -> Path:
    root = Path(directory).resolve()
    pointer = root / "last_good_checkpoint.json"
    if not pointer.is_file():
        raise FileNotFoundError("last_good_checkpoint.json missing")
    row = json.loads(pointer.read_text(encoding="utf8"))
    candidate = (root / row["path"]).resolve()
    if candidate.parent != root:
        raise ValueError("last_good pointer escapes checkpoint directory")
    verified = verify_checkpoint_file(candidate)
    if verified["sha256"] != row.get("sha256"):
        raise ValueError("last_good pointer hash mismatch")
    return candidate
