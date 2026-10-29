"""CPU-only compute-order-of-magnitude facts for native Arctus350.

No model access, imports of torch, training, evaluation, cloud, network or files.
6*N*D is a heuristic for dense decoder-only Transformer pretraining FLOPs,
NOT observed computation, throughput, energy, GPU memory or quality.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json

NATIVE_PARAMS = 349_825_152
TRAINED_TOKENS = 2_073_161_216
MAX_TARGET_TOKENS = 10**14
Q4 = Decimal("0.0001")


def _four(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise ValueError("non-positive denominator")
    return format((Decimal(numerator) / Decimal(denominator)).quantize(Q4), "f")


def budget_report(planned_tokens: int, *, model_parameters: int = NATIVE_PARAMS) -> dict:
    """Estimate relative-scale compute, fail closed on any non-native identity.

    This is a projection ONLY: tokens are supplied, not measured.
    Historical token count is independently reported by a prior native run.
    A comparison to 4T published training tokens is not a target recommendation.
    """
    if type(model_parameters) is not int or model_parameters != NATIVE_PARAMS:
        raise ValueError("not the native Arctus350 architecture")
    if type(planned_tokens) is not int or not 1 <= planned_tokens <= MAX_TARGET_TOKENS:
        raise ValueError("planned tokens must be a bounded positive integer")
    return {
        "schema": "arctus350-compute-reality-v1",
        "status": "THEORETICAL_ESTIMATE_ONLY",
        "model_family": "neral-arctus350",
        "native_parameter_count": NATIVE_PARAMS,
        "historical_tokens": TRAINED_TOKENS,
        "planned_tokens": planned_tokens,
        "tokens_per_parameter": _four(planned_tokens, NATIVE_PARAMS),
        "historical_work_multiplier": _four(planned_tokens, TRAINED_TOKENS),
        "flops_rule": "approx_6_times_parameters_times_tokens",
        "approx_training_flops": 6 * NATIVE_PARAMS * planned_tokens,
        "approx_training_exaflop_operations": _four(
            6 * NATIVE_PARAMS * planned_tokens, 10**18),
        "started_training": False,
        "measured_model_quality": False,
        "sft_or_pretraining_recommendation": "NONE_NO_PERFORMANCE_INFERENCE",
        "limitations": (
            "6ND is a heuristic; this is not measured work. Real computation "
            "includes attention, activation recomputation, optimizer and padding; "
            "hardware runtime, energy, memory and throughput cannot be inferred. "
            "Four trillion tokens in another model is not a required training target "
            "for Arctus350. Increasing training FLOPs does not guarantee quality."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only native Arctus350 approximate compute scale; no training")
    parser.add_argument("--planned-tokens", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        data = budget_report(args.planned_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
