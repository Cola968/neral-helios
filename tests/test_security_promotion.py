from cyberbench_helios import run_benchmark
from promotion_gate50 import evaluate


def test_helios_cyberbench_v1_passes():
    report=run_benchmark()
    assert report["passed"]
    assert report["prompt_false_positive_rate"]==0.0
    assert report["prompt_false_negative_rate"]==0.0
    assert report["secret_redaction"]["recall"]==1.0
    assert report["static_code_checks"]["required_finding_recall"]==1.0


def _comparison(candidate_overrides=None):
    baseline={
        "utf8_valid_pct":100.0,
        "eos_pct":29.0,
        "repeated_trigram_pct":2.95,
        "rawmath_v1_exact_pct":0.0,
        "v3_pair_overall_pct":55.0,
        "v4_pair_overall_pct":55.0,
        "context_recall_overall_pct":50.0,
        "true_byte_bpb_macro_1024":3.0,
        "v4_math_pct":50.0,
        "v4_code_pct":50.0,
    }
    candidate={
        "utf8_valid_pct":100.0,
        "eos_pct":45.0,
        "repeated_trigram_pct":2.0,
        "rawmath_v1_exact_pct":40.0,
        "v3_pair_overall_pct":60.0,
        "v4_pair_overall_pct":62.0,
        "context_recall_overall_pct":60.0,
        "true_byte_bpb_macro_1024":2.8,
        "v4_math_pct":65.0,
        "v4_code_pct":60.0,
    }
    candidate.update(candidate_overrides or {})
    return {"stages":{"baseline20":baseline,"stage_c":candidate}}


def test_promotion_gate_requires_real_wins():
    result=evaluate(_comparison())
    assert result["ready_for_rc_review"]
    assert result["strict_wins"]==9
    assert result["release_ready"] is False


def test_promotion_gate_fails_closed_on_missing_metric():
    report=_comparison();del report["stages"]["stage_c"]["v4_code_pct"]
    result=evaluate(report)
    assert not result["ready_for_rc_review"]
    assert "v4_code_pct" in result["missing_metrics"]


def test_promotion_gate_blocks_severe_repetition_regression():
    result=evaluate(_comparison({"repeated_trigram_pct":12.0}))
    assert not result["ready_for_rc_review"]
    assert any(x["metric"]=="repeated_trigram_pct" for x in result["severe_regressions"])
