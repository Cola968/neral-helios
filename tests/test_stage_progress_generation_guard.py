from stage_progress_gate50 import decide


def metrics(**overrides):
    row = {
        "utf8_valid_pct": 100.0,
        "eos_pct": 20.0,
        "repeated_trigram_pct": 4.0,
        "v3_chat_eos_pct": 20.0,
        "v3_chat_repeated_trigram_pct": 10.0,
        "v4_greedy_eos_pct": 12.5,
        "v4_greedy_repeated_trigram_pct": 12.0,
        "rawmath_v1_exact_pct": 0.0,
        "v3_pair_overall_pct": 50.0,
        "v4_pair_overall_pct": 60.0,
        "context_recall_overall_pct": 50.0,
        "true_byte_bpb_macro_1024": 2.0,
    }
    row.update(overrides)
    return row


def test_progress_gate_can_pass_healthy_broad_progress():
    before = metrics()
    after = metrics(eos_pct=25.0, rawmath_v1_exact_pct=1.0, v4_pair_overall_pct=62.0, true_byte_bpb_macro_1024=1.95)
    result = decide({"before": before, "after": after})
    assert result["proceed"]
    assert result["generation_health"]["proceed"]


def test_progress_gate_blocks_b2_like_zero_eos_even_with_other_wins():
    before = metrics(eos_pct=20.0, repeated_trigram_pct=3.0)
    after = metrics(
        eos_pct=0.0,
        repeated_trigram_pct=13.4,
        v3_chat_eos_pct=0.0,
        v3_chat_repeated_trigram_pct=32.0,
        v4_greedy_eos_pct=0.0,
        v4_greedy_repeated_trigram_pct=28.7,
        rawmath_v1_exact_pct=5.0,
        v3_pair_overall_pct=55.0,
        v4_pair_overall_pct=65.0,
        context_recall_overall_pct=80.0,
        true_byte_bpb_macro_1024=1.9,
    )
    result = decide({"before": before, "after": after})
    assert not result["proceed"]
    assert any(x.startswith("generation:") for x in result["severe"])
    assert not result["generation_health"]["proceed"]
