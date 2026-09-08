"""Post-training gate for Neral-50M M1 RawMath generalization.

M1 must create a real frozen RawMath/transfer gain without trading away any tracked release
metric. Targeted-v1 remains supplemental and is never folded into NIS 1.0. The promotion
policy is Pareto-style: a candidate may improve some metrics and leave saturated metrics equal,
but no tracked metric is allowed to become worse than the source on the identical frozen bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weakness_repair_gate50 import extract

HIGHER_IS_BETTER = (
    "nis",
    "free_eos_pct",
    "utf8_pct",
    "rawmath_exact_pct",
    "rawmath_transfer_pct",
    "rawmath_ood_pct",
    "targeted_math_pct",
    "targeted_code_pct",
    "targeted_de_pct",
    "targeted_reasoning_pct",
    "v4_overall_pct",
    "v4_math_pct",
    "v4_code_pct",
    "context_recall_pct",
    "v5_single_pct",
    "v5_multi_pct",
)
LOWER_IS_BETTER = ("free_repeated_trigram_pct", "v5_bpb_4095")
EPS = 1e-9


def pareto_compare(before: dict, after: dict) -> dict:
    checks: dict[str, bool] = {}
    deltas: dict[str, float | None] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    unchanged: list[str] = []

    for key in HIGHER_IS_BETTER:
        b = before.get(key)
        a = after.get(key)
        ok = b is not None and a is not None and float(a) + EPS >= float(b)
        checks[key] = ok
        if b is None or a is None:
            deltas[key] = None
            regressions.append(key)
            continue
        delta = float(a) - float(b)
        deltas[key] = delta
        if delta > EPS:
            improvements.append(key)
        elif delta < -EPS:
            regressions.append(key)
        else:
            unchanged.append(key)

    for key in LOWER_IS_BETTER:
        b = before.get(key)
        a = after.get(key)
        # If neither side has optional BPB evidence, do not manufacture a failure.
        if b is None and a is None:
            checks[key] = True
            deltas[key] = None
            unchanged.append(key)
            continue
        ok = b is not None and a is not None and float(a) <= float(b) + EPS
        checks[key] = ok
        if b is None or a is None:
            deltas[key] = None
            regressions.append(key)
            continue
        delta = float(a) - float(b)
        deltas[key] = delta
        if delta < -EPS:
            improvements.append(key)
        elif delta > EPS:
            regressions.append(key)
        else:
            unchanged.append(key)

    return {
        "pass": not regressions,
        "checks": checks,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "deltas": deltas,
        "policy": "Every tracked frozen-bundle metric must be non-worse than source; equality is allowed for saturated/unchanged metrics.",
    }


def assess(before: dict, after: dict) -> dict:
    failures: list[str] = []
    before_transfer = (before["rawmath_transfer_pct"] + before["rawmath_ood_pct"]) / 2.0
    after_transfer = (after["rawmath_transfer_pct"] + after["rawmath_ood_pct"]) / 2.0

    pareto = pareto_compare(before, after)
    if not pareto["pass"]:
        failures.extend(f"pareto_regression:{name}" for name in pareto["regressions"])

    checks = {
        "pareto_no_tracked_regression": pareto["pass"],
        "utf8_ge_99": after["utf8_pct"] >= 99.0,
        "rawmath_exact_real_gain": after["rawmath_exact_pct"] >= max(15.0, before["rawmath_exact_pct"] + 5.0),
        "rawmath_transfer_ood_real_gain": after_transfer >= max(5.0, before_transfer + 5.0),
    }
    for name, ok in checks.items():
        if not ok and name != "pareto_no_tracked_regression":
            failures.append(name)

    deltas = {
        "rawmath_exact_pp": after["rawmath_exact_pct"] - before["rawmath_exact_pct"],
        "rawmath_transfer_ood_mean_before": before_transfer,
        "rawmath_transfer_ood_mean_after": after_transfer,
        "rawmath_transfer_ood_mean_pp": after_transfer - before_transfer,
        "nis": after["nis"] - before["nis"],
        "eos_pp": after["free_eos_pct"] - before["free_eos_pct"],
        "repetition_pp_lower_is_better": after["free_repeated_trigram_pct"] - before["free_repeated_trigram_pct"],
        "targeted_math_pp": after["targeted_math_pct"] - before["targeted_math_pct"],
        "targeted_code_pp": after["targeted_code_pct"] - before["targeted_code_pct"],
        "targeted_de_pp": after["targeted_de_pct"] - before["targeted_de_pct"],
        "targeted_reasoning_pp": after["targeted_reasoning_pct"] - before["targeted_reasoning_pct"],
    }
    stretch = {
        "all_tracked_non_worse": pareto["pass"],
        "rawmath_exact_ge_30": after["rawmath_exact_pct"] >= 30.0,
        "transfer_ood_mean_ge_15": after_transfer >= 15.0,
        "rawmath_exact_ge_formal_60": after["rawmath_exact_pct"] >= 60.0,
        "transfer_ood_ge_formal_45": after_transfer >= 45.0,
        "eos_ge_30": after["free_eos_pct"] >= 30.0,
        "repetition_lt_5": after["free_repeated_trigram_pct"] < 5.0,
    }
    return {
        "pass": not failures,
        "failures": failures,
        "checks": checks,
        "pareto": pareto,
        "before": before,
        "after": after,
        "deltas": deltas,
        "stretch": stretch,
        "policy": "Frozen model-only release bundle after training. Promotion is forbidden if any tracked metric regresses. M1 selection itself uses only independent dev/guard sets.",
        "next_action": "candidate eligible for manual promotion review" if not failures else "keep previous selected model",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-dir", required=True)
    ap.add_argument("--after-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = assess(extract(args.before_dir), extract(args.after_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
