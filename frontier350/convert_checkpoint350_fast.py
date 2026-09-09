from __future__ import annotations

"""Convert an unfused Neral 350M resume checkpoint to the fused execution layout.

This is a layout conversion, not a training step: Q/K/V and gate/up weights plus their
AdamW moments are concatenated along output dimension. Scalar optimizer state is required
to agree across merged tensors. The original checkpoint is never modified.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from model350 import FRONTIER_350
from model350_fast import convert_unfused_state_dict, fused_parameter_names, unfused_parameter_names


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _equal_scalar(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return bool(torch.equal(a, b))
    return a == b


def merge_optimizer_states(parts: list[dict]) -> dict:
    if not parts:
        return {}
    keys = set(parts[0])
    if any(set(p) != keys for p in parts[1:]):
        raise RuntimeError("optimizer state fields differ across tensors being fused")
    out = {}
    for key in keys:
        vals = [p[key] for p in parts]
        first = vals[0]
        if torch.is_tensor(first):
            if first.ndim == 0:
                if not all(_equal_scalar(first, v) for v in vals[1:]):
                    raise RuntimeError(f"scalar optimizer state {key} differs across fused tensors")
                out[key] = first.clone()
            else:
                if not all(torch.is_tensor(v) and v.ndim == first.ndim for v in vals):
                    raise RuntimeError(f"optimizer tensor state {key} has incompatible ranks")
                if not all(v.shape[1:] == first.shape[1:] for v in vals[1:]):
                    raise RuntimeError(f"optimizer tensor state {key} has incompatible trailing shapes")
                out[key] = torch.cat(vals, dim=0)
        else:
            if not all(_equal_scalar(first, v) for v in vals[1:]):
                raise RuntimeError(f"optimizer scalar state {key} differs across fused tensors")
            out[key] = first
    return out


def source_names_for_fused(name: str) -> list[str]:
    if name.endswith("attn.qkv.weight"):
        base = name[:-len("qkv.weight")]
        return [base + "q.weight", base + "k.weight", base + "v.weight"]
    if name.endswith("ffn.gate_up.weight"):
        base = name[:-len("gate_up.weight")]
        return [base + "gate.weight", base + "up.weight"]
    return [name]


def convert_optimizer(opt: dict) -> dict:
    groups = opt.get("param_groups")
    states = opt.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise RuntimeError("converter currently requires the single AdamW parameter group used by train350.py")
    old_ids = list(groups[0].get("params", []))
    old_names = unfused_parameter_names()
    new_names = fused_parameter_names()
    if len(old_ids) != len(old_names):
        raise RuntimeError(f"optimizer parameter count mismatch: checkpoint={len(old_ids)} expected={len(old_names)}")
    old_name_to_id = dict(zip(old_names, old_ids))
    new_state = {}
    for new_id, name in enumerate(new_names):
        src_names = source_names_for_fused(name)
        src_ids = [old_name_to_id[n] for n in src_names]
        src_states = [states.get(i, {}) for i in src_ids]
        if any(bool(s) for s in src_states) and not all(bool(s) for s in src_states):
            raise RuntimeError(f"partial optimizer state for fused tensor {name}")
        new_state[new_id] = merge_optimizer_states(src_states) if src_states and src_states[0] else {}
    new_group = dict(groups[0])
    new_group["params"] = list(range(len(new_names)))
    return {"state": new_state, "param_groups": [new_group]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    source = Path(args.input)
    dest = Path(args.output)
    if source.resolve() == dest.resolve():
        raise SystemExit("refusing in-place checkpoint conversion")
    ck = torch.load(source, map_location="cpu", weights_only=False)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise RuntimeError("checkpoint architecture is not the locked Frontier 350M config")
    if "model" not in ck or "optimizer" not in ck:
        raise RuntimeError("resume checkpoint must contain model and optimizer state")
    converted_model = convert_unfused_state_dict(ck["model"])
    converted_opt = convert_optimizer(ck["optimizer"])
    out = dict(ck)
    out["model"] = converted_model
    out["optimizer"] = converted_opt
    out["execution_layout"] = "fused-qkv-gateup-v1"
    out["converted_from_execution_layout"] = ck.get("execution_layout", "unfused-v1")
    atomic_save(out, dest)

    report = {
        "source": str(source),
        "output": str(dest),
        "processed_tokens": int(ck.get("processed_tokens", -1)),
        "step": int(ck.get("step", -1)),
        "old_parameter_tensors": len(unfused_parameter_names()),
        "new_parameter_tensors": len(fused_parameter_names()),
        "total_parameters_unchanged": 349_825_152,
        "optimizer_state_converted": True,
        "training_performed": False,
        "execution_layout": "fused-qkv-gateup-v1",
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
