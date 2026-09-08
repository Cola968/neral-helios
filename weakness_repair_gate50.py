"""Post-training acceptance gate for Neral-50M I2 weakness repair.

This gate compares a source checkpoint and a candidate on the identical model-only bundle.
Targeted-v1 remains supplemental: it can guide diagnosis here but does not become a formal
pretraining-contamination claim or alter NIS 1.0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics as direct_metrics


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def extract(root: str | Path) -> dict:
    root = Path(root)
    free = load(root / "free.json")
    v3 = load(root / "v3.json")
    v4 = load(root / "v4.json")
    raw = load(root / "rawmath.json")
    targeted = load(root / "targeted.json")
    v5 = load(root / "v5.json")
    dm = direct_metrics(free, v3, v4, raw)
    nis = nis_v1(dm)["score"]
    free_s = free["summary"]
    v4_domains = v4["pair_suite"]["domains"]
    v5_bpb = v5["document_bounded_true_byte_bpb"]["macro_bits_per_byte"]
    return {
        "nis": float(nis),
        "free_eos_pct": 100 * float(free_s["eos_fraction"]),
        "free_repeated_trigram_pct": 100 * float(free_s["mean_repeated_trigram_fraction"]),
        "utf8_pct": 100 * float(free_s["valid_utf8_fraction"]),
        "rawmath_exact_pct": float(raw["overall"]["exact_pct"]),
        "rawmath_transfer_pct": float(raw["by_distribution"]["transfer"]["exact_pct"]),
        "rawmath_ood_pct": float(raw["by_distribution"]["ood"]["exact_pct"]),
        "targeted_math_pct": 100 * float(targeted["domains"]["math"]["accuracy"]),
        "targeted_code_pct": 100 * float(targeted["domains"]["code"]["accuracy"]),
        "targeted_de_pct": 100 * float(targeted["domains"]["de_grammar"]["accuracy"]),
        "targeted_reasoning_pct": 100 * float(targeted["domains"]["reasoning"]["accuracy"]),
        "v4_overall_pct": 100 * float(v4["pair_suite"]["accuracy"]),
        "v4_math_pct": 100 * float(v4_domains["math"]["accuracy"]),
        "v4_code_pct": 100 * float(v4_domains["code"]["accuracy"]),
        "context_recall_pct": 100 * float(v4["context_recall"]["accuracy"]),
        "v5_single_pct": 100 * float(v5["single_needle"]["accuracy"]),
        "v5_multi_pct": 100 * float(v5["multi_needle"]["accuracy"]),
        "v5_bpb_4095": float(v5_bpb["4095"]),
    }


def assess(before: dict, after: dict) -> dict:
    failures = []
    checks = {
        "utf8_ge_99": after["utf8_pct"] >= 99.0,
        "eos_improves_5pp_and_ge_18": after["free_eos_pct"] >= max(18.0, before["free_eos_pct"] + 5.0),
        "repetition_improves_and_le_6": after["free_repeated_trigram_pct"] <= min(6.0, before["free_repeated_trigram_pct"] - 0.75),
        "rawmath_exact_not_worse": after["rawmath_exact_pct"] >= before["rawmath_exact_pct"],
        "targeted_math_not_worse": after["targeted_math_pct"] >= before["targeted_math_pct"],
        "targeted_code_not_worse": after["targeted_code_pct"] >= before["targeted_code_pct"],
        "nis_within_1_point": after["nis"] >= before["nis"] - 1.0,
        "v4_overall_protected": after["v4_overall_pct"] >= before["v4_overall_pct"] - 4.0,
        "v4_code_protected": after["v4_code_pct"] >= before["v4_code_pct"] - 10.0,
        "v4_math_protected": after["v4_math_pct"] >= before["v4_math_pct"] - 10.0,
        "context_protected": after["context_recall_pct"] >= before["context_recall_pct"] - 10.0,
        "v5_single_protected": after["v5_single_pct"] >= before["v5_single_pct"] - 10.0,
        "v5_multi_protected": after["v5_multi_pct"] >= before["v5_multi_pct"] - 10.0,
        "long_context_bpb_protected": after["v5_bpb_4095"] <= before["v5_bpb_4095"] * 1.08,
    }
    for name, ok in checks.items():
        if not ok:
            failures.append(name)
    weak_improvements = {
        "eos_pp": after["free_eos_pct"] - before["free_eos_pct"],
        "repetition_pp_lower_is_better": after["free_repeated_trigram_pct"] - before["free_repeated_trigram_pct"],
        "rawmath_exact_pp": after["rawmath_exact_pct"] - before["rawmath_exact_pct"],
        "targeted_math_pp": after["targeted_math_pct"] - before["targeted_math_pct"],
        "targeted_code_pp": after["targeted_code_pct"] - before["targeted_code_pct"],
    }
    stretch = {
        "eos_ge_30": after["free_eos_pct"] >= 30.0,
        "repetition_le_4": after["free_repeated_trigram_pct"] <= 4.0,
        "rawmath_exact_ge_15": after["rawmath_exact_pct"] >= 15.0,
        "targeted_math_ge_50": after["targeted_math_pct"] >= 50.0,
        "targeted_code_ge_55": after["targeted_code_pct"] >= 55.0,
    }
    return {
        "pass": not failures,
        "failures": failures,
        "checks": checks,
        "before": before,
        "after": after,
        "weakness_deltas": weak_improvements,
        "stretch": stretch,
        "next_action": "benchmark-review; candidate may replace I1 only after review" if not failures else "keep I1 source; do not promote I2 candidate",
        "note": "Targeted-v1 is supplemental and was not frozen before B3; formal NIS comparability is unchanged.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-dir", required=True)
    ap.add_argument("--after-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = assess(extract(args.before_dir), extract(args.after_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
