"""Read-only V9 A/B JSONL analysis: detect drift BEFORE a resume boundary.

The first three steps must match before testing resumed continuation. Step metrics
cannot establish exact model equality, model quality or accepted throughput.
No torch, GPU, pickle or training required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _pairs(entries):
    result = {}
    for key, value in entries:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_train_rows(raw: bytes, expected_steps: int, tokens_per_step: int):
    if not 1 <= expected_steps <= 1024 or not 1 <= tokens_per_step <= 10_000_000:
        raise ValueError("bounded expected steps and token count required")
    rows = []
    for line in raw.decode("utf-8-sig").splitlines():
        if line.strip():
            obj = json.loads(line, object_pairs_hook=_pairs, parse_constant=_bad_constant)
            if not isinstance(obj, dict):
                raise ValueError("metric entry must be an object")
            if obj.get("event") == "train":
                rows.append(obj)
    if len(rows) != expected_steps:
        raise ValueError(f"expected {expected_steps} completed train entries, got {len(rows)}")
    source_baseline = None
    for index, row in enumerate(rows, start=1):
        for key, value in (("step", index), ("new_tokens", index * tokens_per_step),
                           ("tokens_per_update", tokens_per_step)):
            if type(row.get(key)) is not int or row[key] != value:
                raise ValueError(f"step {index}: invalid or discontinuous {key}")
        if type(row.get("processed_tokens")) is not int:
            raise ValueError(f"step {index}: invalid processed_tokens")
        baseline = row["processed_tokens"] - row["new_tokens"]
        if baseline < 0 or (source_baseline is not None and baseline != source_baseline):
            raise ValueError("inconsistent source checkpoint token identity")
        source_baseline = baseline
        if not isinstance(row.get("batch_chain"), str) or len(row["batch_chain"]) != 64 or any(c not in "0123456789abcdef" for c in row["batch_chain"]):
            raise ValueError(f"step {index}: invalid batch hash chain")
        for key in ("loss", "grad_norm", "lr", "seconds", "tokens_per_second"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"step {index}: invalid {key}")
        if row["seconds"] <= 0 or row["tokens_per_second"] <= 0:
            raise ValueError("positive timing required")
    return rows, source_baseline


def diagnose(a_bytes: bytes, b_bytes: bytes, *, stop_after: int = 3,
             expected_steps: int = 8, tokens_per_step: int = 24576) -> dict:
    if not 0 < stop_after < expected_steps:
        raise ValueError("resume boundary must be inside the stage")
    a, anchor_a = read_train_rows(a_bytes, expected_steps, tokens_per_step)
    b, anchor_b = read_train_rows(b_bytes, expected_steps, tokens_per_step)
    if anchor_a != anchor_b:
        raise ValueError("A/B source processed-token identity differs")
    chain_mismatch = next((i for i, (x, y) in enumerate(zip(a, b), 1)
                           if x["batch_chain"] != y["batch_chain"]), None)
    tracked = ("loss", "grad_norm", "lr")
    first = next(((i, key) for i, (x, y) in enumerate(zip(a, b), 1)
                  for key in tracked if x[key] != y[key]), None)
    first_loss = next((i for i, (x, y) in enumerate(zip(a, b), 1)
                       if x["loss"] != y["loss"]), None)
    baseline_parity = chain_mismatch is None or chain_mismatch > stop_after
    baseline_parity = baseline_parity and (first is None or first[0] > stop_after)
    return {
        "schema": 1,
        "classification": (
            "input_chain_diverged" if chain_mismatch is not None else
            "pre_resume_numeric_divergence" if not baseline_parity else
            "post_resume_numeric_divergence_possible" if first is not None else
            "recorded_metrics_equal_state_equality_unverified"
        ),
        "source_processed_tokens": anchor_a,
        "training_tokens_each_arm": expected_steps * tokens_per_step,
        "interruption_after_step": stop_after,
        "all_batch_hashes_equal": chain_mismatch is None,
        "first_batch_hash_mismatch_step": chain_mismatch,
        "first_numeric_difference": ({"step": first[0], "field": first[1]} if first else None),
        "first_loss_difference_step": first_loss,
        "numeric_prefix_equal_before_interruption": baseline_parity,
        "eligible_for_production": False,
        "exact_resume_verified": False,
        "training_quality_improvement_verified": False,
        "warning": "Step logs are diagnostics only. A/B must have equal PRE-resume states before a continuation can be attributed to resume; strict full-state checkpoint comparison and frozen quality gates remain mandatory.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-a", type=Path, required=True)
    p.add_argument("--arm-b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stop-after", type=int, default=3)
    p.add_argument("--expected-steps", type=int, default=8)
    p.add_argument("--tokens-per-step", type=int, default=24576)
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite an existing report")
    report = diagnose(args.arm_a.read_bytes(), args.arm_b.read_bytes(),
                      stop_after=args.stop_after, expected_steps=args.expected_steps,
                      tokens_per_step=args.tokens_per_step)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["numeric_prefix_equal_before_interruption"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
