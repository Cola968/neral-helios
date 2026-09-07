from generation_health_gate50 import assess_metrics, extract_metrics


def healthy():
    return {
        "utf8_valid_pct": 100.0,
        "eos_pct": 20.0,
        "repeated_trigram_pct": 4.0,
        "v3_chat_eos_pct": 20.0,
        "v3_chat_repeated_trigram_pct": 10.0,
        "v4_greedy_eos_pct": 12.5,
        "v4_greedy_repeated_trigram_pct": 12.0,
    }


def test_healthy_generation_passes():
    result = assess_metrics(healthy())
    assert result["proceed"]
    assert result["failures"] == []


def test_b2_like_zero_eos_and_repetition_fails():
    row = healthy() | {
        "eos_pct": 0.0,
        "repeated_trigram_pct": 13.4,
        "v3_chat_eos_pct": 0.0,
        "v3_chat_repeated_trigram_pct": 32.1,
        "v4_greedy_eos_pct": 0.0,
        "v4_greedy_repeated_trigram_pct": 28.7,
    }
    result = assess_metrics(row)
    assert not result["proceed"]
    assert "free_eos_below_5pct" in result["failures"]
    assert "free_repetition_above_10pct" in result["failures"]
    assert result["next_stage"] == "generation_repair_before_more_specialization"


def test_reference_regression_is_severe_even_if_absolute_thresholds_pass():
    reference = healthy() | {"eos_pct": 40.0, "repeated_trigram_pct": 2.0}
    candidate = healthy() | {"eos_pct": 20.0, "repeated_trigram_pct": 8.0}
    result = assess_metrics(candidate, reference)
    assert not result["proceed"]
    assert "free_eos_regressed_gt_10pp" in result["severe"]
    assert "free_repetition_regressed_gt_5pp" in result["severe"]


def test_missing_metric_fails_closed():
    row = healthy(); row.pop("v4_greedy_eos_pct")
    result = assess_metrics(row)
    assert not result["proceed"]
    assert result["reason"] == "missing_metrics"


def test_extract_metrics_uses_raw_fraction_reports():
    free = {"summary": {"valid_utf8_fraction": 1.0, "eos_fraction": .2, "mean_repeated_trigram_fraction": .04}}
    v3 = {"free_generation": {"chat_instruction": {"summary": {"eos_fraction": .1, "mean_repeated_trigram_fraction": .12}}}}
    v4 = {"generation_grid": {"summary": {"greedy": {"eos_fraction": .125, "mean_repeated_trigram_fraction": .2}}}}
    metrics = extract_metrics(free, v3, v4)
    assert metrics["utf8_valid_pct"] == 100.0
    assert metrics["eos_pct"] == 20.0
    assert metrics["v4_greedy_eos_pct"] == 12.5
