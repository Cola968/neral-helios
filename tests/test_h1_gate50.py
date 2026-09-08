from h1_gate50 import assess


def base():
    return {
        "nis": 65.25,
        "free_eos_pct": 11.0,
        "free_repeated_trigram_pct": 7.4,
        "utf8_pct": 100.0,
        "rawmath_exact_pct": 8.0,
        "rawmath_transfer_pct": 0.0,
        "rawmath_ood_pct": 0.0,
        "targeted_math_pct": 40.0,
        "targeted_code_pct": 45.0,
        "targeted_de_pct": 80.0,
        "targeted_reasoning_pct": 70.0,
        "v4_overall_pct": 77.27,
        "v4_math_pct": 83.33,
        "v4_code_pct": 100.0,
        "context_recall_pct": 100.0,
        "v5_single_pct": 70.0,
        "v5_multi_pct": 25.0,
        "v5_bpb_4095": 1.71,
    }


def test_accepts_no_regression_real_nis_gain():
    before = base(); after = dict(before)
    after["nis"] = 65.8
    after["rawmath_exact_pct"] = 10.0
    result = assess(before, after)
    assert result["pass"] is True
    assert result["promotion_review_eligible"] is True


def test_rejects_even_one_tracked_regression():
    before = base(); after = dict(before)
    after["nis"] = 70.0
    after["targeted_de_pct"] = 75.0
    result = assess(before, after)
    assert result["pass"] is False
    assert "targeted_de_pct" in result["pareto"]["regressions"]


def test_rejects_flat_nis_even_if_pareto_clean():
    before = base(); after = dict(before)
    result = assess(before, after)
    assert result["pass"] is False
    assert "nis_real_gain" in result["failures"]


def test_nis70_is_only_stretch_signal():
    before = base(); after = dict(before)
    after["nis"] = 70.0
    after["rawmath_exact_pct"] = 20.0
    result = assess(before, after)
    assert result["stretch"]["nis_ge_70"] is True
    assert result["pass"] is True
