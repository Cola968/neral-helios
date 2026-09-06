"""Versioned internal Neral Intelligence Score (NIS).

NIS is a project-local composite for comparing Neral checkpoints evaluated with the
same frozen harness. It is NOT IQ, not an external benchmark, and not comparable to
scores from other model families.

NIS 1.0 intentionally excludes speed, parameter count, Helios/SOLVE/security tools and
training loss. It fails closed when any required frozen metric is missing.

A separate NIS Preview 0.1 exists only to summarize the historical 100-prompt dashboard,
which predates v3/v4/RawMath-v1 checkpoint results. Preview 0.1 is not comparable to NIS 1.0.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


NIS_VERSION = "1.0"
PREVIEW_VERSION = "0.1-preview"

FULL_COMPONENTS = {
    "language_knowledge": {
        "weight": 0.20,
        "metrics": ("v3_de_grammar_pct", "v3_en_grammar_pct", "v3_facts_pct", "v3_commonsense_pct"),
    },
    "raw_math": {"weight": 0.20, "metrics": ("rawmath_v1_exact_pct",)},
    "code": {"weight": 0.10, "metrics": ("v3_code_pct", "v4_code_pct")},
    "science_focus": {
        "weight": 0.15,
        "metrics": ("v4_physics_pct", "v4_astronomy_pct", "v4_aviation_pct", "v4_transport_pct"),
    },
    "logic_induction": {
        "weight": 0.15,
        "metrics": ("v4_logic_pct", "v4_commonsense_pct", "v4_induction_pct"),
    },
    "translation": {"weight": 0.10, "metrics": ("v4_translation_pct",)},
    "context_use": {"weight": 0.10, "metrics": ("context_recall_overall_pct",)},
}


def _bounded_pct(value, name: str) -> float:
    if value is None:
        raise ValueError(f"missing metric: {name}")
    value = float(value)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be a percentage in [0,100], got {value}")
    return value


def nis_v1(metrics: dict) -> dict:
    """Compute the full NIS 1.0 score from frozen benchmark percentages."""
    components = {}
    contributions = {}
    for name, spec in FULL_COMPONENTS.items():
        values = [_bounded_pct(metrics.get(key), key) for key in spec["metrics"]]
        score = statistics.fmean(values)
        components[name] = score
        contributions[name] = score * spec["weight"]
    total = sum(contributions.values())
    return {
        "name": "Neral Intelligence Score",
        "version": NIS_VERSION,
        "status": "complete",
        "score": total,
        "scale": "0-100",
        "components": components,
        "weights": {name: spec["weight"] for name, spec in FULL_COMPONENTS.items()},
        "contributions": contributions,
        "scope": "Internal Neral-only frozen-benchmark composite; not IQ and not cross-model comparable.",
        "excluded": ["parameter_count", "decode_speed", "training_loss", "SOLVE", "VERIFY", "Helios security/tool scores"],
    }


def nis_preview_01(*, manual_comprehensibility_pct: float, raw_math_exact_pct: float,
                   non_repetition_trigram_pct: float, eos_pct: float) -> dict:
    """Historical compatibility score for the old 100-prompt dashboard only.

    Formula:
      70% manual comprehensibility
      15% raw math exact
      15% generation health, where health = mean(non-repetition, EOS)

    The manual score came from a small historical subset, so this preview is deliberately
    not promoted to NIS 1.0 and must not be compared numerically with it.
    """
    comp = _bounded_pct(manual_comprehensibility_pct, "manual_comprehensibility_pct")
    math = _bounded_pct(raw_math_exact_pct, "raw_math_exact_pct")
    nonrep = _bounded_pct(non_repetition_trigram_pct, "non_repetition_trigram_pct")
    eos = _bounded_pct(eos_pct, "eos_pct")
    health = statistics.fmean((nonrep, eos))
    score = 0.70 * comp + 0.15 * math + 0.15 * health
    return {
        "name": "Neral Intelligence Score Preview",
        "version": PREVIEW_VERSION,
        "status": "historical-provisional",
        "score": score,
        "scale": "0-100",
        "components": {
            "manual_comprehensibility": comp,
            "raw_math_exact": math,
            "generation_health": health,
            "non_repetition_trigram": nonrep,
            "eos": eos,
        },
        "weights": {"manual_comprehensibility": 0.70, "raw_math_exact": 0.15, "generation_health": 0.15},
        "scope": "Historical 100-prompt compatibility score only; not comparable to NIS 1.0.",
    }


def _load_metrics(path: str, stage: str | None) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    if stage is not None:
        stages = obj.get("stages")
        if not isinstance(stages, dict) or stage not in stages:
            raise ValueError(f"stage {stage!r} not found in comparison JSON")
        obj = stages[stage]
    if not isinstance(obj, dict):
        raise ValueError("metrics JSON must contain an object")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-json", help="Flattened benchmark metrics JSON, or a comparison JSON with --stage")
    ap.add_argument("--stage", help="Stage key inside comparison JSON, e.g. baseline20 or stage_c")
    ap.add_argument("--out")
    args = ap.parse_args()
    if not args.metrics_json:
        raise SystemExit("--metrics-json is required for NIS 1.0")
    report = nis_v1(_load_metrics(args.metrics_json, args.stage))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
