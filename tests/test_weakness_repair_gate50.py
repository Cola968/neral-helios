from weakness_repair_gate50 import assess


def base():
    return {
        "nis":65.25,"free_eos_pct":11.0,"free_repeated_trigram_pct":7.41,"utf8_pct":100.0,
        "rawmath_exact_pct":8.0,"rawmath_transfer_pct":0.0,"rawmath_ood_pct":0.0,
        "targeted_math_pct":40.0,"targeted_code_pct":45.0,"targeted_de_pct":80.0,"targeted_reasoning_pct":70.0,
        "v4_overall_pct":77.27,"v4_math_pct":83.33,"v4_code_pct":100.0,"context_recall_pct":100.0,
        "v5_single_pct":70.0,"v5_multi_pct":25.0,"v5_bpb_4095":1.711,
    }


def test_gate_passes_real_repair_with_preservation():
    before=base();after=dict(before)
    after.update(free_eos_pct=24.0,free_repeated_trigram_pct=4.8,rawmath_exact_pct=12.0,
                 targeted_math_pct=50.0,targeted_code_pct=55.0,nis=65.0,v5_bpb_4095=1.75)
    result=assess(before,after)
    assert result["pass"] is True
    assert result["failures"]==[]


def test_gate_rejects_cosmetic_eos_gain_with_repetition_regression():
    before=base();after=dict(before)
    after.update(free_eos_pct=24.0,free_repeated_trigram_pct=8.0)
    result=assess(before,after)
    assert result["pass"] is False
    assert "repetition_improves_and_le_6" in result["failures"]
