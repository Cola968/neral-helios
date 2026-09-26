"""Plan exact-resume-safe Phoenix foundation quality gates without training.

Converts cumulative historical+new token milestones into new-lineage token counters and,
for intermediate stops, rounds down to a complete optimizer-update boundary.
"""
from __future__ import annotations

import argparse
import json

HISTORICAL_TOKENS = 2_073_161_216
CEILING = 10_000_000_000
GATES = (
    2_500_000_000,
    3_000_000_000,
    4_000_000_000,
    5_500_000_000,
    7_000_000_000,
    8_500_000_000,
    10_000_000_000,
)


def plan(seq: int, microbatch: int, grad_accum: int, tokens_per_second: float = 0.0) -> dict:
    if min(seq, microbatch, grad_accum) <= 0:
        raise ValueError("positive update geometry required")
    if seq > 8192:
        raise ValueError("sequence exceeds Frontier context")
    if tokens_per_second < 0:
        raise ValueError("tokens_per_second cannot be negative")

    update = seq * microbatch * grad_accum
    rows = []
    previous = 0
    for cumulative in GATES:
        raw_new = cumulative - HISTORICAL_TOKENS
        if cumulative == CEILING:
            stop_new = raw_new
            exact_resume_boundary = raw_new % update == 0
        else:
            stop_new = (raw_new // update) * update
            exact_resume_boundary = True
        delta = stop_new - previous
        if delta <= 0:
            raise ValueError("quality gates collapsed under update geometry")
        row = {
            "cumulative_quality_gate_tokens": cumulative,
            "raw_new_lineage_tokens": raw_new,
            "planned_stop_new_lineage_tokens": stop_new,
            "planned_cumulative_tokens": HISTORICAL_TOKENS + stop_new,
            "undershoot_vs_named_gate_tokens": raw_new - stop_new,
            "delta_from_previous_stop_tokens": delta,
            "complete_update_boundary": exact_resume_boundary,
        }
        if tokens_per_second > 0:
            row["pure_compute_hours_from_previous_stop"] = delta / tokens_per_second / 3600
        rows.append(row)
        previous = stop_new

    return {
        "schema": 1,
        "classification": "planning_only_no_training",
        "historical_anchor_tokens": HISTORICAL_TOKENS,
        "ceiling_tokens": CEILING,
        "remaining_to_ceiling_tokens": CEILING - HISTORICAL_TOKENS,
        "seq": seq,
        "microbatch": microbatch,
        "grad_accum": grad_accum,
        "tokens_per_optimizer_update": update,
        "planning_tokens_per_second": tokens_per_second or None,
        "gates": rows,
        "rule": "Intermediate stops are rounded down to complete AdamW updates; the final terminal target may contain a remainder because no exact resume follows it.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--microbatch", type=int, required=True)
    p.add_argument("--grad-accum", type=int, required=True)
    p.add_argument("--tokens-per-second", type=float, default=0.0)
    args = p.parse_args()
    print(json.dumps(plan(args.seq, args.microbatch, args.grad_accum, args.tokens_per_second),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
