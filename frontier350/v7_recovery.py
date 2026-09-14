from __future__ import annotations

"""Compact exact-recovery states for Scale V7.

Only the four trainable blocks, AdamW state and logical RNG/data-chain state are stored.
Frozen weights stay referenced by the immutable stage anchor.  This makes sub-5M recovery
practical without writing the full 350M model every interval.
"""

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from v7_policy import semantic_resume_contract

SCHEMA = 7


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def atomic_torch_save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    # os.replace is atomic on the same volume. A half-written .tmp is never resumable.
    os.replace(tmp, path)


def capture_exact_recovery(
    *,
    named: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    contract: dict,
    anchor_sha256: str,
    source_processed_tokens: int,
    done: int,
    step: int,
    chain: str,
    counts: dict,
) -> dict:
    if done <= 0 or step <= 0:
        raise ValueError("recovery state requires positive progress")
    masters = {name: p.detach().cpu().clone() for name, p in named}
    return {
        "schema": SCHEMA,
        "kind": "neral350-v7-exact-recovery",
        "anchor_sha256": anchor_sha256,
        "source_processed_tokens": int(source_processed_tokens),
        "processed_tokens": int(source_processed_tokens) + int(done),
        "new_tokens": int(done),
        "updates": int(step),
        "semantic_contract": semantic_resume_contract(contract),
        "masters": masters,
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "sampler_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "batch_chain": chain,
        "source_tokens": dict(counts),
    }


def save_exact_recovery(path: Path, **kwargs) -> dict:
    state = capture_exact_recovery(**kwargs)
    atomic_torch_save(state, path)
    return state


def load_exact_recovery(
    path: Path,
    *,
    expected_anchor_sha256: str,
    contract: dict,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("kind") != "neral350-v7-exact-recovery":
        raise ValueError("not a V7 exact recovery state")
    if state.get("anchor_sha256") != expected_anchor_sha256:
        raise ValueError("recovery anchor differs")
    if state.get("semantic_contract") != semantic_resume_contract(contract):
        raise ValueError("recovery semantic contract differs")
    if int(state.get("new_tokens", 0)) <= 0:
        raise ValueError("invalid recovery token count")
    if not state.get("masters") or not state.get("optimizer"):
        raise ValueError("incomplete recovery state")
    return state


def restore_exact_recovery(
    state: dict,
    *,
    named: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> tuple[int, int, str, dict]:
    named_list = list(named)
    by_name = dict(named_list)
    if set(state["masters"]) != set(by_name):
        raise ValueError("recovery trainable parameter set differs")
    with torch.no_grad():
        for name, value in state["masters"].items():
            p = by_name[name]
            p.copy_(value.to(device=p.device, dtype=p.dtype))
    optimizer.load_state_dict(state["optimizer"])
    rng.setstate(state["sampler_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return (
        int(state["new_tokens"]),
        int(state["updates"]),
        str(state["batch_chain"]),
        dict(state["source_tokens"]),
    )


def materialize_checkpoint(anchor: Path, recovery: Path, out: Path) -> int:
    """Build an evaluator-compatible full checkpoint only when a recovery probe is needed."""

    anchor_ck = torch.load(anchor, map_location="cpu", weights_only=False)
    state = torch.load(recovery, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("kind") != "neral350-v7-exact-recovery":
        raise ValueError("invalid V7 recovery snapshot")
    from recovery_identity import file_sha
    if file_sha(anchor) != state.get("anchor_sha256"):
        raise ValueError("recovery snapshot belongs to a different anchor")
    model = dict(anchor_ck["model"])
    for name, value in state["masters"].items():
        if name not in model:
            raise ValueError(f"unknown trainable parameter: {name}")
        model[name] = value.to(dtype=model[name].dtype)
    ck = dict(anchor_ck)
    ck["model"] = model
    ck["processed_tokens"] = int(state["processed_tokens"])
    ck["step"] = int(anchor_ck.get("step", 0)) + int(state["updates"])
    ck["best_validation"] = float("inf")
    ck["validation_identity"] = ""
    ck["optimizer"] = {**anchor_ck.get("optimizer", {}), "state": {}}
    ck.pop("v9_training_state", None)
    ck["v7_recovery_provenance"] = {
        "anchor_sha256": state["anchor_sha256"],
        "recovery_state": str(recovery),
        "new_training_tokens": int(state["new_tokens"]),
        "batch_chain": state["batch_chain"],
    }
    atomic_torch_save(ck, out)
    return int(ck["processed_tokens"])


def list_recovery_states(directory: Path) -> list[Path]:
    rows: list[tuple[int, Path]] = []
    for path in directory.glob("v7_recovery_*.pt"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if obj.get("schema") == SCHEMA and obj.get("kind") == "neral350-v7-exact-recovery":
                rows.append((int(obj["new_tokens"]), path))
        except Exception:
            continue
    return [path for _, path in sorted(rows, reverse=True)]
