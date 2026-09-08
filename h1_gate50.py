"""Frozen post-training acceptance gate for Neral-50M H1.

H1 is a general intensive campaign, not a benchmark-targeted repair. Promotion review therefore
requires strict no-regression across the tracked frozen bundle plus a real NIS improvement. The
frozen bundle is evaluated only after all H1 weight-changing stages have completed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m1_gate50 import pareto_compare
from weakness_repair_gate50 import extract

EPS = 1e-9


def assess(before: dict, after: dict) -> dict:
    pareto = pareto_compare(before, after)
    nis_gain = float(after["nis"]) - float(before["nis"])
    transfer_before = (float(before["rawmath_transfer_pct"]) + float(before["rawmath_ood_pct"])) / 2.0
    transfer_after = (float(after["rawmath_transfer_pct"]) + float(after["rawmath_ood_pct"])) / 2.0

    checks = {
        "pareto_no_tracked_regression": bool(pareto["pass"]),
        "nis_real_gain": nis_gain > 0.25 + EPS,
        "utf8_ge_99": float(after["utf8_pct"]) >= 99.0,
    }
    failures = []
    if not pareto["pass"]:
        failures.extend(f"pareto_regression:{name}" for name in pareto["regressions"])
    for name, ok in checks.items():
        if name != "pareto_no_tracked_regression" and not ok:
            failures.append(name)

    eligible = not failures
    return {
        "pass": eligible,
        "promotion_review_eligible": eligible,
        "failures": failures,
        "checks": checks,
        "pareto": pareto,
        "before": before,
        "after": after,
        "deltas": {
            "nis": nis_gain,
            "rawmath_exact_pp": float(after["rawmath_exact_pct"]) - float(before["rawmath_exact_pct"]),
            "transfer_ood_mean_before": transfer_before,
            "transfer_ood_mean_after": transfer_after,
            "transfer_ood_mean_pp": transfer_after - transfer_before,
            "eos_pp": float(after["free_eos_pct"]) - float(before["free_eos_pct"]),
            "repetition_pp_lower_is_better": float(after["free_repeated_trigram_pct"]) - float(before["free_repeated_trigram_pct"]),
        },
        "stretch": {
            "nis_ge_70": float(after["nis"]) >= 70.0,
            "rawmath_exact_ge_20": float(after["rawmath_exact_pct"]) >= 20.0,
            "transfer_ood_mean_gt_0": transfer_after > 0.0,
            "eos_ge_30": float(after["free_eos_pct"]) >= 30.0,
            "repetition_lt_5": float(after["free_repeated_trigram_pct"]) < 5.0,
        },
        "policy": (
            "General H1 release-review gate: every tracked frozen-bundle metric must be non-worse "
            "than immutable D3/I1 and NIS must improve by >0.25. This is not the formal release target gate."
        ),
        "next_action": (
            "candidate eligible for manual promotion review"
            if eligible
            else "keep D3/I1 selected; inspect H1 regressions before any promotion"
        ),
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
