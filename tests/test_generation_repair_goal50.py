import generation_repair_goal50 as gate


def candidate():
    return {
        "v4_pair_overall_pct":64.0,"context_recall_overall_pct":75.0,"v4_code_pct":70.0,"v4_math_pct":50.0,
        "true_byte_bpb_macro_1024":2.40,"rawmath_v1_exact_pct":5.0,
    }


def gen():
    return {
        "utf8_valid_pct":100.0,"eos_pct":20.0,"repeated_trigram_pct":5.0,
        "v3_chat_eos_pct":15.0,"v3_chat_repeated_trigram_pct":10.0,
        "v4_greedy_eos_pct":20.0,"v4_greedy_repeated_trigram_pct":10.0,
    }


def patch_metrics(monkeypatch,after=None,generation=None,nis=50.0):
    before={"rawmath_v1_exact_pct":0.0,"v4_pair_overall_pct":63.0,"context_recall_overall_pct":80.0,"v4_code_pct":75.0,"v4_math_pct":33.3,"true_byte_bpb_macro_1024":2.407}
    values=iter([before,after or candidate()])
    monkeypatch.setattr(gate,"direct_metrics",lambda *args:next(values))
    monkeypatch.setattr(gate,"extract_metrics",lambda *args:generation or gen())
    monkeypatch.setattr(gate,"nis_v1",lambda metrics:{"score":nis,"status":"complete"})


def test_goal_passes_when_generation_and_capability_targets_are_met(monkeypatch):
    patch_metrics(monkeypatch)
    result=gate.assess({}, {}, {}, {}, {}, {}, {}, {})
    assert result["proceed"]
    assert result["failures"]==[]


def test_goal_rejects_zero_eos_even_when_capabilities_are_good(monkeypatch):
    bad=gen()|{"eos_pct":0.0}
    patch_metrics(monkeypatch,generation=bad)
    result=gate.assess({}, {}, {}, {}, {}, {}, {}, {})
    assert not result["proceed"]
    assert "free_eos_below_10pct" in result["failures"]


def test_goal_rejects_context_regression(monkeypatch):
    bad=candidate()|{"context_recall_overall_pct":60.0}
    patch_metrics(monkeypatch,after=bad)
    result=gate.assess({}, {}, {}, {}, {}, {}, {}, {})
    assert not result["proceed"]
    assert "context_below_70pct" in result["failures"]


def test_goal_rejects_nis_below_floor(monkeypatch):
    patch_metrics(monkeypatch,nis=47.9)
    result=gate.assess({}, {}, {}, {}, {}, {}, {}, {})
    assert not result["proceed"]
    assert "nis_below_48" in result["failures"]
