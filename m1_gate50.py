"""Post-training gate for Neral-50M M1 RawMath generalization.

M1 must create a real frozen RawMath/transfer gain without trading away the existing chat-model
capabilities. Targeted-v1 remains supplemental and is never folded into NIS 1.0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weakness_repair_gate50 import extract


def assess(before: dict, after: dict) -> dict:
    failures: list[str] = []
    before_transfer = (before["rawmath_transfer_pct"] + before["rawmath_ood_pct"]) / 2.0
    after_transfer = (after["rawmath_transfer_pct"] + after["rawmath_ood_pct"]) / 2.0
    before_bpb = before.get("v5_bpb_4095")
    after_bpb = after.get("v5_bpb_4095")
    if before_bpb is None:
        bpb_ok = True
        bpb_note = "source 4095-BPB unavailable; needle/context guards remain mandatory"
    elif after_bpb is None:
        bpb_ok = False
        bpb_note = "candidate 4095-BPB missing while source evidence exists"
    else:
        bpb_ok = float(after_bpb) <= float(before_bpb) * 1.08
        bpb_note = "candidate 4095-BPB must remain within +8%"

    checks = {
        "utf8_ge_99": after["utf8_pct"] >= 99.0,
        "rawmath_exact_real_gain": after["rawmath_exact_pct"] >= max(15.0, before["rawmath_exact_pct"] + 5.0),
        "rawmath_transfer_ood_real_gain": after_transfer >= max(5.0, before_transfer + 5.0),
        "targeted_math_protected": after["targeted_math_pct"] >= before["targeted_math_pct"] - 5.0,
        "targeted_code_protected": after["targeted_code_pct"] >= before["targeted_code_pct"] - 5.0,
        "targeted_de_protected": after["targeted_de_pct"] >= before["targeted_de_pct"] - 5.0,
        "targeted_reasoning_protected": after["targeted_reasoning_pct"] >= before["targeted_reasoning_pct"] - 5.0,
        "v4_math_protected": after["v4_math_pct"] >= before["v4_math_pct"] - 10.0,
        "v4_code_protected": after["v4_code_pct"] >= before["v4_code_pct"] - 10.0,
        "v4_overall_protected": after["v4_overall_pct"] >= before["v4_overall_pct"] - 5.0,
        "nis_within_1_5": after["nis"] >= before["nis"] - 1.5,
        "context_protected": after["context_recall_pct"] >= before["context_recall_pct"] - 10.0,
        "single_needle_protected": after["v5_single_pct"] >= before["v5_single_pct"] - 10.0,
        "multi_needle_protected": after["v5_multi_pct"] >= before["v5_multi_pct"] - 10.0,
        "eos_not_materially_worse": after["free_eos_pct"] >= max(10.0, before["free_eos_pct"] - 2.0),
        "repetition_bounded": after["free_repeated_trigram_pct"] <= min(8.5, before["free_repeated_trigram_pct"] + 1.0),
        "long_context_bpb_protected": bpb_ok,
    }
    for name, ok in checks.items():
        if not ok:
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
        "before": before,
        "after": after,
        "deltas": deltas,
        "stretch": stretch,
        "long_context_bpb_evidence": {"source": before_bpb, "candidate": after_bpb, "note": bpb_note},
        "policy": "Frozen model-only release bundle after training. M1 selection itself uses only independent dev/guard sets.",
        "next_action": "candidate eligible for manual promotion review" if not failures else "keep previous selected model",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-dir", required=True)
    ap.add_argument("--after-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = assess(extract(args.before_dir), extract(args.after_dir))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
