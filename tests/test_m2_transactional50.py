from m2_transactional50 import (
    acceptance, adapt_weights, exact_transfer_growth, normalize_weights, regressions,
    safety_buffers, transfer_progress,
)
from pareto_dev_guard50 import METRICS


def base_vector():
    out = {}
    for spec in METRICS:
        if spec.direction == "higher":
            out[spec.key] = 50.0
        else:
            out[spec.key] = 1.0
    out["id_exact_pct"] = 8.0
    out["transfer_exact_pct"] = 0.0
    out["id_nll"] = 1.0
    out["transfer_nll"] = 1.0
    out["free_utf8_pct"] = 100.0
    out["free_eos_pct"] = 40.0
    out["free_repetition_pct"] = 10.0
    return out


def safe_candidate(source):
    candidate = dict(source)
    candidate["id_nll"] = .99
    candidate["transfer_nll"] = .99
    candidate["sft_de_nll"] = .98
    candidate["free_eos_pct"] = source["free_eos_pct"] + 3.0
    candidate["free_repetition_pct"] = source["free_repetition_pct"] * .60
    candidate["free_health_score"] = source["free_health_score"] + 1.0
    return candidate


def test_accepts_strict_no_regression_math_progress_with_safety_margin():
    source = base_vector(); anchor = dict(source); candidate = safe_candidate(source)
    decision = acceptance(source, anchor, candidate, best_score=0.0, min_score_gain=.01)
    assert decision["accepted"]
    assert not decision["anchor_regressions"]
    assert decision["transfer_progress"]


def test_rejects_even_one_non_math_regression():
    source = base_vector(); candidate = safe_candidate(source)
    candidate["sft_code_nll"] = 1.01
    decision = acceptance(source, source, candidate, best_score=0.0, min_score_gain=.01)
    assert not decision["accepted"]
    assert "sft_code_nll" in decision["source_assessment"]["regressions"]


def test_anchor_must_be_monotonic_too():
    source = base_vector(); anchor = dict(source); anchor["free_eos_pct"] = 45.0
    candidate = safe_candidate(source); candidate["free_eos_pct"] = 44.0
    assert "free_eos_pct" in regressions(anchor, candidate)
    assert not acceptance(source, anchor, candidate, best_score=0.0, min_score_gain=.01)["accepted"]


def test_utf8_regression_is_now_visible_to_dev_pareto():
    source = base_vector(); candidate = safe_candidate(source)
    candidate["free_utf8_pct"] = 99.0
    decision = acceptance(source, source, candidate, best_score=0.0, min_score_gain=.01)
    assert not decision["accepted"]
    assert "free_utf8_pct" in decision["source_assessment"]["regressions"]


def test_transfer_stagnation_blocks_commit_and_adapts_weight():
    source = base_vector(); candidate = safe_candidate(source)
    candidate["transfer_nll"] = .999
    assert not transfer_progress(source, candidate)
    decision = acceptance(source, source, candidate, best_score=0.0, min_score_gain=.01)
    assert not decision["accepted"]
    assert "transfer_stagnation" in decision["anchor_regressions"]
    w = {"id_math": .18, "transfer_math": .22, "sft": .30, "lm": .30}
    changed = adapt_weights(w, ["transfer_stagnation"])
    assert changed["transfer_math"] > normalize_weights(w)["transfer_math"]


def test_old_m2_style_small_de_and_generation_gains_fail_safety_buffer():
    source = base_vector(); candidate = dict(source)
    candidate["id_nll"] = .98; candidate["transfer_nll"] = .98
    candidate["sft_de_nll"] = .995
    candidate["free_eos_pct"] = source["free_eos_pct"] + 1.0
    candidate["free_repetition_pct"] = source["free_repetition_pct"] * .70
    bad = safety_buffers(source, candidate)
    assert "german_safety_margin" in bad
    assert "repetition_safety_margin" in bad
    assert "eos_safety_margin" in bad


def test_adaptive_weights_stay_normalized_and_shift_protection():
    w = {"id_math": .18, "transfer_math": .22, "sft": .30, "lm": .30}
    changed = adapt_weights(w, ["sft_code_nll", "context_dev_pct"])
    assert abs(sum(changed.values()) - 1.0) < 1e-12
    assert changed["sft"] > normalize_weights(w)["sft"]
    assert changed["lm"] > normalize_weights(w)["lm"]


def test_exact_transfer_growth_is_explicit():
    source = base_vector(); candidate = dict(source)
    candidate["transfer_exact_pct"] = 3.0
    assert exact_transfer_growth(source, candidate)
