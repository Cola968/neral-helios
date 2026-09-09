import json
from pathlib import Path

from eos_math_policy_v4 import (
    E3_OBJECTIVE_MIX,
    MATH_REPAIR_DATA_MIX,
    answer_length_bucket,
    assess_eos_candidate,
    balance_eos_rows,
    eos_bucket_report,
    validate_eos_row,
    validate_policy,
)


def row(i: int, n: int, *, language: str = "de", domain: str = "general") -> dict:
    return {
        "id": str(i),
        "prompt": f"Frage {i}",
        "answer": "x" * n,
        "language": language,
        "domain": domain,
    }


def test_length_buckets_and_row_validation():
    assert answer_length_bucket("x" * 64) == "short"
    assert answer_length_bucket("x" * 65) == "medium"
    assert answer_length_bucket("x" * 193) == "long"
    assert validate_eos_row(row(1, 20))["answer_bytes"] == 20


def test_embedded_turn_marker_is_rejected():
    bad = row(1, 20)
    bad["answer"] = "Eine Antwort.\nUser: neue Frage"
    try:
        validate_eos_row(bad)
    except ValueError as exc:
        assert "dialogue-turn" in str(exc)
    else:
        raise AssertionError("embedded user turn should be rejected")


def test_balancer_increases_terminal_boundary_density_without_duplication():
    rows = [row(i, 40) for i in range(60)] + [row(100 + i, 120) for i in range(60)] + [row(200 + i, 260) for i in range(60)]
    picked = balance_eos_rows(rows, max_examples=100)
    report = eos_bucket_report(picked)
    assert len({r["id"] for r in picked}) == len(picked) == 100
    assert report["shares"]["short"] >= 0.35
    assert report["shares"]["medium"] >= 0.35
    assert report["shares"]["long"] >= 0.15


def test_eos_candidate_requires_free_and_teacher_forced_gain():
    anchor = {
        "free_eos_pct": 30.0,
        "terminal_eos_top1_pct": 42.0,
        "repeated_trigram_pct": 3.0,
        "utf8_pct": 100.0,
        "mean_generated_tokens": 22.0,
    }
    candidate = dict(anchor, free_eos_pct=41.0, terminal_eos_top1_pct=49.0, repeated_trigram_pct=2.8)
    assert assess_eos_candidate(anchor, candidate)["accepted"] is True

    teacher_only = dict(anchor, free_eos_pct=32.0, terminal_eos_top1_pct=60.0)
    result = assess_eos_candidate(anchor, teacher_only)
    assert result["accepted"] is False
    assert "free_eos_gain_too_small" in result["blockers"]


def test_eos_candidate_rejects_premature_stopping_and_repetition_regression():
    anchor = {
        "free_eos_pct": 30.0,
        "terminal_eos_top1_pct": 40.0,
        "repeated_trigram_pct": 2.0,
        "utf8_pct": 100.0,
        "mean_generated_tokens": 20.0,
    }
    candidate = {
        "free_eos_pct": 80.0,
        "terminal_eos_top1_pct": 70.0,
        "repeated_trigram_pct": 4.0,
        "utf8_pct": 100.0,
        "mean_generated_tokens": 2.0,
    }
    result = assess_eos_candidate(anchor, candidate)
    assert result["accepted"] is False
    assert "premature_stop_risk" in result["blockers"]
    assert "repetition_regressed" in result["blockers"]


def test_math_policy_prefers_transfer_over_id_memorization():
    report = validate_policy()
    assert abs(sum(E3_OBJECTIVE_MIX.values()) - 1.0) < 1e-9
    assert abs(sum(MATH_REPAIR_DATA_MIX.values()) - 1.0) < 1e-9
    assert E3_OBJECTIVE_MIX["rawmath_transfer"] > E3_OBJECTIVE_MIX["rawmath_id"]
    assert MATH_REPAIR_DATA_MIX["math_transfer"] > MATH_REPAIR_DATA_MIX["math_id"]
    assert "transfer/OOD" in report["principle"]


def test_hf_math_eos_manifest_keeps_risky_sources_fail_closed():
    manifest = json.loads(Path("hf_math_eos_sources_v4.json").read_text(encoding="utf8"))
    by_id = {row["dataset_id"]: row for row in manifest["sources"]}
    assert by_id["OpenAssistant/oasst2"]["decision"] == "approved_filtered_eos_sft"
    assert by_id["HuggingFaceTB/finemath"]["decision"].startswith("quarantine")
    assert by_id["microsoft/orca-math-word-problems-200k"]["decision"].startswith("quarantine")
    assert by_id["TIGER-Lab/MathInstruct"]["decision"].startswith("quarantine")
    assert by_id["nvidia/OpenMathInstruct-2"]["decision"] == "blocked_for_current_benchmark_purity"
    assert manifest["e3_external_data_policy"]["math_external_admitted_now"] == []


def test_e3_source_contract_is_transfer_heavy_and_eos_explicit():
    text = Path("train_weakness_repair50_e3.py").read_text(encoding="utf8")
    required = [
        "TransferTrainingProblemSampler",
        "transfer_math_weight",
        "math_eos_weight",
        "free_eos_guard",
        "terminal_eos_guard",
        "repetition_guard",
        "transfer_math_guard",
        "forced_eos\": False",
        "foreign_weights_used\": False",
    ]
    for needle in required:
        assert needle in text
