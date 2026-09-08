from m1_gate50 import assess, pareto_compare


def base():
    return {
        "nis": 65.25, "free_eos_pct": 11.0, "free_repeated_trigram_pct": 7.41, "utf8_pct": 100.0,
        "rawmath_exact_pct": 8.0, "rawmath_transfer_pct": 0.0, "rawmath_ood_pct": 0.0,
        "targeted_math_pct": 40.0, "targeted_code_pct": 45.0, "targeted_de_pct": 80.0,
        "targeted_reasoning_pct": 70.0, "v4_overall_pct": 77.27, "v4_math_pct": 83.33,
        "v4_code_pct": 100.0, "context_recall_pct": 100.0, "v5_single_pct": 70.0,
        "v5_multi_pct": 25.0, "v5_bpb_4095": 1.711,
    }


def good_candidate():
    after = dict(base())
    after.update(
        rawmath_exact_pct=22.0,
        rawmath_transfer_pct=10.0,
        rawmath_ood_pct=8.0,
        targeted_math_pct=45.0,
        nis=65.5,
        free_eos_pct=12.0,
        free_repeated_trigram_pct=7.0,
        v5_bpb_4095=1.70,
    )
    return after


def test_gate_passes_real_math_generalization_with_zero_regressions():
    result = assess(base(), good_candidate())
    assert result["pass"] is True
    assert result["failures"] == []
    assert result["pareto"]["pass"] is True


def test_gate_rejects_in_distribution_only_memorization():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=25.0, rawmath_transfer_pct=1.0, rawmath_ood_pct=0.0)
    result = assess(before, after)
    assert result["pass"] is False
    assert "rawmath_transfer_ood_real_gain" in result["failures"]


def test_gate_rejects_even_small_nis_regression():
    before = base(); after = good_candidate(); after["nis"] = before["nis"] - 0.01
    result = assess(before, after)
    assert result["pass"] is False
    assert "pareto_regression:nis" in result["failures"]


def test_gate_rejects_math_gain_that_destroys_code():
    before = base(); after = good_candidate(); after["targeted_code_pct"] = 44.9
    result = assess(before, after)
    assert result["pass"] is False
    assert "pareto_regression:targeted_code_pct" in result["failures"]


def test_gate_rejects_math_gain_that_destroys_german():
    before = base(); after = good_candidate(); after["targeted_de_pct"] = 79.9
    result = assess(before, after)
    assert result["pass"] is False
    assert "pareto_regression:targeted_de_pct" in result["failures"]


def test_gate_rejects_math_gain_that_destroys_reasoning():
    before = base(); after = good_candidate(); after["targeted_reasoning_pct"] = 69.9
    result = assess(before, after)
    assert result["pass"] is False
    assert "pareto_regression:targeted_reasoning_pct" in result["failures"]


def test_gate_rejects_worse_repetition_even_if_other_metrics_improve():
    before = base(); after = good_candidate(); after["free_repeated_trigram_pct"] = 7.42
    result = assess(before, after)
    assert result["pass"] is False
    assert "pareto_regression:free_repeated_trigram_pct" in result["failures"]


def test_missing_candidate_bpb_fails_when_source_has_evidence():
    before = base(); after = good_candidate(); after["v5_bpb_4095"] = None
    result = assess(before, after)
    assert result["pass"] is False
    assert "pareto_regression:v5_bpb_4095" in result["failures"]


def test_ceiling_metrics_may_remain_equal():
    before = base(); after = good_candidate()
    result = pareto_compare(before, after)
    assert result["checks"]["utf8_pct"] is True
    assert result["checks"]["v4_code_pct"] is True
    assert result["checks"]["context_recall_pct"] is True
