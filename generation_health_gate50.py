"""Fail-closed generation-health gate for Neral-50M frozen reports.

This gate exists because lower language-model loss does not imply usable free generation.
It checks termination, repetition and UTF-8 directly from raw-model benchmark outputs. It
never changes decoding, never applies repetition penalties, and never changes model weights.

The thresholds are engineering readiness criteria for allowing another weight-changing
stage; they are not an intelligence score and are deliberately separate from NIS/promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLDS = {
    "utf8_valid_pct_min": 99.0,
    "free_eos_pct_min": 5.0,
    "free_repeated_trigram_pct_max": 10.0,
    "v3_chat_eos_pct_min": 5.0,
    "v3_chat_repeated_trigram_pct_max": 25.0,
    "v4_greedy_eos_pct_min": 5.0,
    "v4_greedy_repeated_trigram_pct_max": 25.0,
}

REQUIRED_METRICS = (
    "utf8_valid_pct",
    "eos_pct",
    "repeated_trigram_pct",
    "v3_chat_eos_pct",
    "v3_chat_repeated_trigram_pct",
    "v4_greedy_eos_pct",
    "v4_greedy_repeated_trigram_pct",
)


def _pct(value) -> float | None:
    return None if value is None else 100.0 * float(value)


def extract_metrics(free_report: dict, v3_report: dict, v4_report: dict) -> dict[str, float | None]:
    free = free_report.get("summary", {})
    chat = v3_report.get("free_generation", {}).get("chat_instruction", {}).get("summary", {})
    greedy = v4_report.get("generation_grid", {}).get("summary", {}).get("greedy", {})
    return {
        "utf8_valid_pct": _pct(free.get("valid_utf8_fraction")),
        "eos_pct": _pct(free.get("eos_fraction")),
        "repeated_trigram_pct": _pct(free.get("mean_repeated_trigram_fraction")),
        "v3_chat_eos_pct": _pct(chat.get("eos_fraction")),
        "v3_chat_repeated_trigram_pct": _pct(chat.get("mean_repeated_trigram_fraction")),
        "v4_greedy_eos_pct": _pct(greedy.get("eos_fraction")),
        "v4_greedy_repeated_trigram_pct": _pct(greedy.get("mean_repeated_trigram_fraction")),
    }


def assess_metrics(metrics: dict, reference: dict | None = None) -> dict:
    missing = [k for k in REQUIRED_METRICS if metrics.get(k) is None]
    if missing:
        return {
            "proceed": False,
            "reason": "missing_metrics",
            "missing": missing,
            "metrics": metrics,
            "thresholds": THRESHOLDS,
            "failures": [],
            "severe": [],
            "next_stage": "repair_generation_evaluation_pipeline",
        }

    failures: list[str] = []
    if metrics["utf8_valid_pct"] < THRESHOLDS["utf8_valid_pct_min"]:
        failures.append("utf8_below_99pct")
    if metrics["eos_pct"] < THRESHOLDS["free_eos_pct_min"]:
        failures.append("free_eos_below_5pct")
    if metrics["repeated_trigram_pct"] > THRESHOLDS["free_repeated_trigram_pct_max"]:
        failures.append("free_repetition_above_10pct")
    if metrics["v3_chat_eos_pct"] < THRESHOLDS["v3_chat_eos_pct_min"]:
        failures.append("chat_eos_below_5pct")
    if metrics["v3_chat_repeated_trigram_pct"] > THRESHOLDS["v3_chat_repeated_trigram_pct_max"]:
        failures.append("chat_repetition_above_25pct")
    if metrics["v4_greedy_eos_pct"] < THRESHOLDS["v4_greedy_eos_pct_min"]:
        failures.append("v4_greedy_eos_below_5pct")
    if metrics["v4_greedy_repeated_trigram_pct"] > THRESHOLDS["v4_greedy_repeated_trigram_pct_max"]:
        failures.append("v4_greedy_repetition_above_25pct")

    severe: list[str] = []
    delta = None
    if reference is not None:
        ref_missing = [k for k in REQUIRED_METRICS if reference.get(k) is None]
        if ref_missing:
            severe.append("reference_missing_generation_metrics")
        else:
            delta = {k: float(metrics[k]) - float(reference[k]) for k in REQUIRED_METRICS}
            if delta["eos_pct"] < -10.0:
                severe.append("free_eos_regressed_gt_10pp")
            if delta["repeated_trigram_pct"] > 5.0:
                severe.append("free_repetition_regressed_gt_5pp")
            if delta["v3_chat_eos_pct"] < -10.0:
                severe.append("chat_eos_regressed_gt_10pp")
            if delta["v4_greedy_repeated_trigram_pct"] > 10.0:
                severe.append("v4_greedy_repetition_regressed_gt_10pp")
            if delta["utf8_valid_pct"] < -1.0:
                severe.append("utf8_regressed_gt_1pp")

    proceed = not failures and not severe
    return {
        "proceed": proceed,
        "reason": "pass" if proceed else "generation_not_ready",
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "failures": failures,
        "severe": severe,
        "delta_vs_reference": delta,
        "next_stage": "normal_benchmark_gated_progression" if proceed else "generation_repair_before_more_specialization",
        "note": "Raw-model generation-health decision only; no decoding penalties, tools, SOLVE, or weight changes.",
    }


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--reference-free")
    ap.add_argument("--reference-v3")
    ap.add_argument("--reference-v4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics = extract_metrics(_load(args.free), _load(args.v3), _load(args.v4))
    reference = None
    refs = (args.reference_free, args.reference_v3, args.reference_v4)
    if any(refs):
        if not all(refs):
            raise ValueError("reference-free/reference-v3/reference-v4 must be provided together")
        reference = extract_metrics(_load(args.reference_free), _load(args.reference_v3), _load(args.reference_v4))
    result = assess_metrics(metrics, reference)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["proceed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
