import json
from pathlib import Path

from run_stage_ablation50 import summarize, targeted_accuracy


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf8")


def test_targeted_accuracy_supports_current_schema():
    assert targeted_accuracy({"overall": {"accuracy": 0.5875}}) == 0.5875


def test_targeted_accuracy_supports_legacy_schema():
    assert targeted_accuracy({"overall_accuracy": 0.575}) == 0.575


def test_summarize_current_targeted_schema(tmp_path):
    write(tmp_path / "rawmath.json", {
        "overall": {"exact_pct": 10.0, "answer_nll": 0.9},
        "by_distribution": {"transfer": {"exact_pct": 2.0}, "ood": {"exact_pct": 1.0}},
    })
    write(tmp_path / "free.json", {
        "summary": {"eos_fraction": 0.2, "mean_repeated_trigram_fraction": 0.05},
    })
    write(tmp_path / "targeted.json", {
        "overall": {"accuracy": 0.575},
        "domains": {
            "math": {"accuracy": 0.55},
            "code": {"accuracy": 0.4},
            "de_grammar": {"accuracy": 0.75},
        },
    })
    write(tmp_path / "v4.json", {
        "pair_suite": {
            "accuracy": 0.77,
            "domains": {"math": {"accuracy": 0.83}, "code": {"accuracy": 1.0}},
        },
        "context_recall": {"accuracy": 0.9},
    })
    result = summarize(tmp_path)
    assert result["targeted_overall_pct"] == 57.49999999999999 or result["targeted_overall_pct"] == 57.5
    assert result["targeted_math_pct"] == 55.00000000000001 or result["targeted_math_pct"] == 55.0
    assert result["rawmath_exact_pct"] == 10.0
