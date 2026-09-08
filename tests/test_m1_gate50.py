from m1_gate50 import assess


def base():
    return {
        "nis": 65.25, "free_eos_pct": 11.0, "free_repeated_trigram_pct": 7.41, "utf8_pct": 100.0,
        "rawmath_exact_pct": 8.0, "rawmath_transfer_pct": 0.0, "rawmath_ood_pct": 0.0,
        "targeted_math_pct": 40.0, "targeted_code_pct": 45.0, "targeted_de_pct": 80.0,
        "targeted_reasoning_pct": 70.0, "v4_overall_pct": 77.27, "v4_math_pct": 83.33,
        "v4_code_pct": 100.0, "context_recall_pct": 100.0, "v5_single_pct": 70.0,
        "v5_multi_pct": 25.0, "v5_bpb_4095": 1.711,
    }


def test_gate_passes_real_math_generalization_with_preservation():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=22.0, rawmath_transfer_pct=10.0, rawmath_ood_pct=8.0,
                 targeted_math_pct=45.0, nis=64.8, free_eos_pct=12.0,
                 free_repeated_trigram_pct=7.0, v5_bpb_4095=1.76)
    result = assess(before, after)
    assert result["pass"] is True
    assert result["failures"] == []


def test_gate_rejects_in_distribution_only_memorization():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=25.0, rawmath_transfer_pct=1.0, rawmath_ood_pct=0.0)
    result = assess(before, after)
    assert result["pass"] is False
    assert "rawmath_transfer_ood_real_gain" in result["failures"]


def test_gate_rejects_math_gain_that_destroys_code():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=25.0, rawmath_transfer_pct=10.0, rawmath_ood_pct=10.0,
                 targeted_code_pct=30.0)
    result = assess(before, after)
    assert result["pass"] is False
    assert "targeted_code_protected" in result["failures"]


def test_gate_rejects_math_gain_that_destroys_german():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=25.0, rawmath_transfer_pct=10.0, rawmath_ood_pct=10.0,
                 targeted_de_pct=70.0)
    result = assess(before, after)
    assert result["pass"] is False
    assert "targeted_de_protected" in result["failures"]


def test_gate_rejects_math_gain_that_destroys_reasoning():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=25.0, rawmath_transfer_pct=10.0, rawmath_ood_pct=10.0,
                 targeted_reasoning_pct=60.0)
    result = assess(before, after)
    assert result["pass"] is False
    assert "targeted_reasoning_protected" in result["failures"]


def test_missing_candidate_bpb_fails_when_source_has_evidence():
    before = base(); after = dict(before)
    after.update(rawmath_exact_pct=20.0, rawmath_transfer_pct=8.0, rawmath_ood_pct=8.0, v5_bpb_4095=None)
    result = assess(before, after)
    assert result["pass"] is False
    assert "long_context_bpb_protected" in result["failures"]
