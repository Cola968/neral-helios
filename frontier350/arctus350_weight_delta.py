"""Read-only Arctus350 true weight deltas: 1B SAFE vs 2B and 1B-based SFT.

Local-only CPU PyTorch. Private checkpoints are neither modified nor uploaded.
The output contains aggregate per-family counts, never tensor values/paths.
This is evidence for parameter updates, NOT for answer quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


BASE_SHA = "ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c"
TWO_B_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
HISTORICAL_SFT_SHA = "408998c0d0816527341253fe37c2d9e2f1216c5cad8500dee247d11c72cc8d7b"
BF16_BIN_BOUNDARY = 2 ** -8  # 0.00390625; classification uses actual baseline abs values


def classify_weight(name: str) -> str:
    if name == "lm_head.weight":
        return "tied_embedding_alias"
    if name == "tok_emb.weight":
        return "embedding"
    if name == "norm.weight" or re.fullmatch(
        r"blocks\.\d+\.(?:n1|n2|attn\.(?:qn|kn))\.weight", name
    ):
        return "rmsnorm"
    m = re.fullmatch(r"blocks\.\d+\.(attn|ffn)\.([a-z]+)\.weight", name)
    if m:
        if m.group(1) == "attn":
            return "attention_output" if m.group(2) == "o" else (
                "attention_qkv" if m.group(2) in ("q", "k", "v") else "unknown"
            )
        return "ffn_down" if m.group(2) == "down" else (
            "ffn_gate_up" if m.group(2) in ("gate", "up") else "unknown"
        )
    return "unknown"


def summarize_group(row: dict) -> dict:
    needed = ("total", "changed", "low_total", "low_changed", "high_total",
              "high_changed", "baseline_sq", "delta_sq")
    if not isinstance(row, dict) or any(k not in row for k in needed):
        raise ValueError("missing group evidence")
    for k in needed[:6]:
        if type(row[k]) is not int or row[k] < 0:
            raise ValueError("invalid group count: " + k)
    for k in needed[6:]:
        if type(row[k]) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0:
            raise ValueError("invalid group squared norm: " + k)
    n = row["total"]
    if (n < 1 or row["low_total"] + row["high_total"] != n or
        row["low_changed"] > row["low_total"] or
        row["high_changed"] > row["high_total"] or
        row["changed"] != row["low_changed"] + row["high_changed"]):
        raise ValueError("contradictory group counts")
    return {
        "parameter_count": n,
        "changed_count": row["changed"],
        "fraction_changed": row["changed"] / n,
        "fraction_changed_low_magnitude":
            row["low_changed"] / row["low_total"] if row["low_total"] else None,
        "fraction_changed_high_magnitude":
            row["high_changed"] / row["high_total"] if row["high_total"] else None,
        "low_magnitude_parameter_count": row["low_total"],
        "high_magnitude_parameter_count": row["high_total"],
        "relative_l2":
            math.sqrt(row["delta_sq"] / row["baseline_sq"]) if row["baseline_sq"] else None,
    }


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _blank() -> dict:
    return dict(total=0, changed=0, low_total=0, low_changed=0,
                high_total=0, high_changed=0, baseline_sq=0.0, delta_sq=0.0)


def _load(path, expected_digest):
    import torch  # no torch required to unit-test pure contracts
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .model350 import FRONTIER_350
    from .secure_checkpoint350 import load_checkpoint

    if digest(path) != expected_digest:
        raise ValueError("pinned checkpoint SHA256 mismatch")
    ck = load_checkpoint(path)
    if ck.get("config") != FRONTIER_350.to_dict():
        raise ValueError("native architecture mismatch")
    if ck.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("16K BPE lineage mismatch")
    weights = ck.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("checkpoint model state missing")
    if any(not isinstance(v, torch.Tensor) for v in weights.values()):
        raise ValueError("non-tensor model state entry")
    return weights


def compare_states(base, other):
    import torch
    if base.keys() != other.keys():
        raise ValueError("model state key mismatch")
    grouped = {}
    for key in sorted(base):
        family = classify_weight(key)
        if family == "tied_embedding_alias":
            # model350 ties tok_emb/lm_head; do not count the same parameter twice
            if not torch.equal(base[key], base["tok_emb.weight"]) or not torch.equal(
                other[key], other["tok_emb.weight"]
            ):
                raise ValueError("tied embedding alias does not match")
            continue
        if family == "unknown":
            raise ValueError("unexpected model parameter name")
        a, b = base[key], other[key]
        if a.shape != b.shape or not a.is_floating_point() or not b.is_floating_point():
            raise ValueError("parameter shape or type mismatch")
        # Keep only a single tensor's FP32 converted data at a time on CPU.
        x = a.detach().float()
        y = b.detach().float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("nonfinite checkpoint parameter")
        changed = x != y
        low = x.abs() < BF16_BIN_BOUNDARY
        delta = y - x
        group = grouped.setdefault(family, _blank())
        group["total"] += x.numel()
        group["changed"] += int(changed.sum().item())
        group["low_total"] += int(low.sum().item())
        group["low_changed"] += int((low & changed).sum().item())
        high = ~low
        group["high_total"] += int(high.sum().item())
        group["high_changed"] += int((high & changed).sum().item())
        group["baseline_sq"] += float(x.double().square().sum().item())
        group["delta_sq"] += float(delta.double().square().sum().item())
        del x, y, changed, low, high, delta
    if not grouped:
        raise ValueError("no trainable parameters found")
    return {name: summarize_group(item) for name, item in sorted(grouped.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--two-b", type=Path, required=True)
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError("output exists: never overwrite a previous audit")
    if len({a.base.resolve(), a.two_b.resolve(), a.sft.resolve()}) != 3:
        raise ValueError("three distinct pinned files required")
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint_ids = {"1b": BASE_SHA, "2b": TWO_B_SHA, "1b_sft": HISTORICAL_SFT_SHA}
    base = _load(a.base, BASE_SHA)
    second = _load(a.two_b, TWO_B_SHA)
    comparison_2b = compare_states(base, second)
    del second
    sft = _load(a.sft, HISTORICAL_SFT_SHA)
    comparison_sft = compare_states(base, sft)
    del sft, base
    for key, path in (("1b", a.base), ("2b", a.two_b), ("1b_sft", a.sft)):
        if digest(path) != checkpoint_ids[key]:
            raise ValueError("source checkpoint mutated during audit")
    report = {
        "schema": "arctus350-real-checkpoint-weight-delta-v1",
        "status": "measured",
        "checkpoint_sha256": checkpoint_ids,
        "dtype_note": "float32 numeric differences computed from actual stored original weights",
        "magnitude_cutoff_abs": BF16_BIN_BOUNDARY,
        "comparisons": {"1b_vs_2b": comparison_2b,
                        "1b_vs_historical_1b_sft": comparison_sft},
        "limitations": "Parameter changes alone do not establish causality, accuracy, "
                       "optimizer quality, or readiness for release.",
        "weights_modified": False, "model_training": False,
        "private_tensor_values_exported": False,
    }
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
