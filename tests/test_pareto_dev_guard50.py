from pareto_dev_guard50 import assess_dev


def base():
    return {
        "id_exact_pct": 10.0,
        "transfer_exact_pct": 0.0,
        "id_nll": 1.00,
        "transfer_nll": 1.20,
        "sft_macro_nll": 1.00,
        "sft_math_nll": 1.05,
        "sft_code_nll": 0.90,
        "sft_reasoning_nll": 0.95,
        "sft_de_nll": 0.85,
        "lm_macro": 1.10,
        "free_eos_pct": 20.0,
        "free_repetition_pct": 6.0,
        "free_health_score": 14.0,
        "context_dev_pct": 80.0,
    }


def test_accepts_math_growth_with_everything_else_preserved():
    before = base(); after = dict(before)
    after.update(id_exact_pct=12.0, transfer_exact_pct=2.0, id_nll=.96, transfer_nll=1.10)
    result = assess_dev(before, after)
    assert result["selectable"] is True
    assert result["regressions"] == []
    assert result["math_growth"]["exact_growth"] is True


def test_rejects_code_regression_even_when_math_improves():
    before = base(); after = dict(before)
    after.update(id_exact_pct=20.0, transfer_exact_pct=10.0, sft_code_nll=.91)
    result = assess_dev(before, after)
    assert result["selectable"] is False
    assert "sft_code_nll" in result["regressions"]


def test_rejects_repetition_regression_even_when_eos_improves():
    before = base(); after = dict(before)
    after.update(id_nll=.99, transfer_nll=1.18, free_eos_pct=40.0, free_repetition_pct=6.1)
    result = assess_dev(before, after)
    assert result["selectable"] is False
    assert "free_repetition_pct" in result["regressions"]


def test_accepts_soft_math_growth_before_exact_match_moves():
    before = base(); after = dict(before)
    after.update(id_nll=.99, transfer_nll=1.19)
    result = assess_dev(before, after)
    assert result["selectable"] is True
    assert result["math_growth"]["soft_nll_growth"] is True


def test_equal_saturated_metrics_are_allowed():
    before = base(); before["context_dev_pct"] = 100.0
    after = dict(before); after.update(id_exact_pct=11.0, context_dev_pct=100.0)
    result = assess_dev(before, after)
    assert result["selectable"] is True
