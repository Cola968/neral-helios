"""Evidence-only admission for NextGen V1. Never starts training or promotes a checkpoint.

Inputs are externally produced measured evidence, not independently authenticated facts.
Use a separate frozen release holdout after choosing a candidate on development data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STAGE_TOKENS = 5_000_000
MIN_STAGES = 3
TARGET_TPS = 60_000.0
STRETCH_TPS = 70_000.0


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: number required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: non-finite")
    return result


def validate_evidence(record: dict) -> dict:
    """Fail closed; never infer missing evaluations or count rejected work as progress."""
    failures: list[str] = []
    if record.get("evidence_version") != 1:
        raise ValueError("evidence_version must be 1")
    if record.get("source") != "self_hosted_rtx4070":
        raise ValueError("evidence must identify the tested GPU")
    if record.get("weights_origin") != "neral_own_weights":
        raise ValueError("foreign/unknown weights are not an accepted experiment")
    if record.get("checkpoint_isolated") is not True or record.get("production_unchanged") is not True:
        raise ValueError("experiment must be isolated from production")
    if record.get("same_data_order") is not True or record.get("same_token_budget") is not True:
        raise ValueError("A/B needs matching data order and token budgets")
    if record.get("safety_limits_unchanged") is not True:
        raise ValueError("thermal/VRAM/SAFE limits must remain unchanged")

    stages = record.get("stages")
    if not isinstance(stages, list) or len(stages) < MIN_STAGES:
        raise ValueError("at least three complete stage records required")
    total_tokens = 0
    total_seconds = 0.0
    rates = []
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"stage {i}: object required")
        accepted = finite_number(stage.get("accepted_tokens"), f"stage {i} accepted_tokens")
        elapsed = finite_number(stage.get("wall_seconds"), f"stage {i} wall_seconds")
        rejected = finite_number(stage.get("rejected_tokens"), f"stage {i} rejected_tokens")
        if accepted < STAGE_TOKENS or int(accepted) != accepted or elapsed <= 0 or rejected != 0:
            failures.append(f"stage_{i}_incomplete_or_rejected")
        if stage.get("includes_eval_checkpoint_recovery") is not True:
            failures.append(f"stage_{i}_incomplete_wall_time")
        if stage.get("health_pass") is not True:
            failures.append(f"stage_{i}_health_failed")
        rate = accepted / elapsed
        rates.append(rate)
        total_tokens += int(accepted)
        total_seconds += elapsed
        if rate < TARGET_TPS:
            failures.append(f"stage_{i}_below_60k")
    combined = total_tokens / total_seconds
    if combined < TARGET_TPS:
        failures.append("combined_below_60k")

    base = record.get("baseline")
    candidate = record.get("candidate")
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        raise ValueError("both quality reports required")
    for key in ("tokenizer_sha256", "validation_sha256", "dev_suite_sha256", "eval_protocol_sha256"):
        if not isinstance(base.get(key), str) or len(base[key]) != 64 or base.get(key) != candidate.get(key):
            raise ValueError(f"baseline/candidate {key} mismatch or missing")
    if not isinstance(candidate.get("checkpoint_sha256"), str) or len(candidate["checkpoint_sha256"]) != 64:
        raise ValueError("candidate checkpoint identity required")
    for key in ("validation_loss", "repeated_trigram_pct"):
        bv = finite_number(base.get(key), f"baseline {key}")
        cv = finite_number(candidate.get(key), f"candidate {key}")
        if bv < 0 or cv < 0 or cv > bv:
            failures.append(f"regression_{key}")
    bc, cc = base.get("dev_categories"), candidate.get("dev_categories")
    if not isinstance(bc, dict) or not bc or not isinstance(cc, dict) or bc.keys() != cc.keys():
        raise ValueError("same nonempty dev-category names required")
    for name in bc:
        bv = finite_number(bc[name], f"baseline {name}")
        cv = finite_number(cc[name], f"candidate {name}")
        if not (0 <= bv <= 100 and 0 <= cv <= 100) or cv < bv:
            failures.append(f"regression_dev_{name}")

    phase = record.get("phase")
    if phase not in ("foundation", "post_training"):
        raise ValueError("phase must be foundation or post_training")
    if phase == "post_training":
        bg, cg = base.get("generation"), candidate.get("generation")
        if not isinstance(bg, dict) or not isinstance(cg, dict):
            raise ValueError("post-training needs both generation reports")
        if bg.get("protocol_sha256") != cg.get("protocol_sha256") or not isinstance(bg.get("protocol_sha256"), str) or len(bg["protocol_sha256"]) != 64:
            raise ValueError("generation protocol mismatch")
        if finite_number(bg.get("cases"), "baseline generation cases") < 24 or bg["cases"] != cg.get("cases"):
            raise ValueError("matching generation cases >=24 required")
        for metric in ("exact_correct", "eos_correct"):
            bv = finite_number(bg.get(metric), "baseline " + metric)
            cv = finite_number(cg.get(metric), "candidate " + metric)
            if not (0 <= bv <= bg["cases"] and 0 <= cv <= bg["cases"]) or cv <= bv:
                failures.append(f"no_post_training_gain_{metric}")
    return {
        "admitted": not failures,
        "failures": failures,
        "accepted_tokens": total_tokens,
        "wall_seconds": total_seconds,
        "stage_accepted_tps": rates,
        "combined_accepted_tps": combined,
        "stretch_70k_verified": not failures and all(rate >= STRETCH_TPS for rate in rates) and combined >= STRETCH_TPS,
        "warning": "Reports are claims until original self-hosted artifacts, hashes and frozen holdout are independently checked; not a production promotion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate_evidence(json.loads(args.evidence.read_text(encoding="utf-8")))
    if args.out.exists():
        parser.error("refusing to overwrite an existing report")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
