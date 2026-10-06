"""Compare two independent synthetic v2 compute probes, NEVER accepted training.

This is a strict, CPU-only comparability gate. A successful report never
establishes quality, numerical/gradient parity, real-data or production readiness.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

TARGET = 50_000.0
SAME = ("gpu", "model_sha256", "parameters", "trainable_parameters", "train_blocks",
        "batch", "seq", "warmup_steps", "measured_steps", "timed_synthetic_tokens")


def finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: finite number required")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{field}: nonfinite")
    return x


def verify(report: dict, execution: str) -> float:
    if not isinstance(report, dict) or report.get("evidence_version") != 2:
        raise ValueError("v2 synthetic report required")
    if report.get("classification") != "compute_only_synthetic_unaccepted_tokens":
        raise ValueError("only isolated synthetic reports may be compared")
    if report.get("execution") != execution:
        raise ValueError(f"execution must be {execution}")
    if report.get("eligible_for_production") is not False:
        raise ValueError("synthetic reports must explicitly prohibit promotion")
    if report.get("gpu") != "NVIDIA GeForce RTX 4070":
        raise ValueError("unexpected GPU")
    if report.get("parameters") != 349_825_152 or report.get("train_blocks") != 4:
        raise ValueError("unexpected model/trainable configuration")
    sha = report.get("model_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha):
        raise ValueError("invalid model identity")
    if report.get("batch") not in (8, 16) or report.get("seq") != 512:
        raise ValueError("this A/B supports only fixed tested shapes")
    for key in ("warmup_steps", "measured_steps", "timed_synthetic_tokens"):
        v = finite(report.get(key), key)
        if int(v) != v or v <= 0:
            raise ValueError(f"{key}: positive integer required")
    if report["warmup_steps"] < 5 or report["measured_steps"] < 24:
        raise ValueError("insufficient warmup or measurement")
    if report["timed_synthetic_tokens"] != report["batch"] * report["seq"] * report["measured_steps"]:
        raise ValueError("timed token count inconsistent")
    wall = finite(report.get("measured_wall_seconds"), "measured_wall_seconds")
    setup = finite(report.get("setup_and_warmup_wall_seconds"), "setup_and_warmup_wall_seconds")
    rate = finite(report.get("compute_only_tokens_per_second"), "compute_only_tokens_per_second")
    inclusive = finite(report.get("setup_inclusive_synthetic_tokens_per_second"), "setup_inclusive_synthetic_tokens_per_second")
    slow = finite(report.get("slowest_measured_step_tokens_per_second"), "slowest_measured_step_tokens_per_second")
    if wall <= 0 or setup < 0 or rate <= 0 or slow <= 0:
        raise ValueError("invalid durations/rates")
    if not math.isclose(rate, report["timed_synthetic_tokens"] / wall, rel_tol=1e-6):
        raise ValueError("timed throughput is inconsistent")
    if not math.isclose(inclusive, report["timed_synthetic_tokens"] / (wall + setup), rel_tol=1e-6):
        raise ValueError("setup-inclusive throughput is inconsistent")
    if slow > rate * (1 + 1e-5):
        raise ValueError("slowest step cannot exceed aggregate rate")
    if finite(report.get("peak_temperature_c"), "peak_temperature_c") > 82:
        raise ValueError("temperature limit exceeded")
    if finite(report.get("peak_reserved_gib"), "peak_reserved_gib") > 9:
        raise ValueError("VRAM limit exceeded")
    if report.get("synthetic_50k_target") != TARGET:
        raise ValueError("unexpected target")
    if report.get("synthetic_50k_timed_target_met") is not (rate >= TARGET):
        raise ValueError("reported target flag contradicts measured rate")
    return rate


def compare(baseline: dict, candidate: dict) -> dict:
    base_rate = verify(baseline, "compile-default")
    candidate_rate = verify(candidate, "compile-overhead")
    for key in SAME:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"unmatched A/B field: {key}")
    return {
        "evidence_version": 1,
        "classification": "synthetic_only_comparable_ab_not_training_evidence",
        "batch": baseline["batch"], "seq": baseline["seq"],
        "baseline_synthetic_tps": base_rate,
        "candidate_synthetic_tps": candidate_rate,
        "speedup": candidate_rate / base_rate,
        "candidate_faster": candidate_rate > base_rate,
        "candidate_50k_synthetic_timed": candidate_rate >= TARGET,
        "candidate_50k_setup_inclusive": candidate["setup_inclusive_synthetic_tokens_per_second"] >= TARGET,
        "baseline_setup_seconds": baseline["setup_and_warmup_wall_seconds"],
        "candidate_setup_seconds": candidate["setup_and_warmup_wall_seconds"],
        "numerical_parity_verified": False,
        "end_to_end_accepted_training_verified": False,
        "eligible_for_production": False,
        "warning": "Fixed synthetic token batch; no real weights, source corpus, data pipeline, independent numerical parity, checkpoint, validation or release test. Crossing 50k synthetic is NOT 50k accepted training.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing overwrite")
    result = compare(json.loads(args.baseline.read_text(encoding="utf-8")),
                     json.loads(args.candidate.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".tmp")
    try:
        temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        os.replace(temp, args.out)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
