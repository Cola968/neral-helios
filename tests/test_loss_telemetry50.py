import json
import math

from loss_telemetry50 import ce_to_bits, ce_to_perplexity, eval_due, should_select, summarize_metrics, validation_derived


def test_ce_conversions_are_exact_for_simple_values():
    assert ce_to_perplexity(0.0) == 1.0
    assert math.isclose(ce_to_perplexity(math.log(4.0)), 4.0, rel_tol=1e-12)
    assert math.isclose(ce_to_bits(math.log(2.0)), 1.0, rel_tol=1e-12)


def test_validation_derived_preserves_domains():
    row = {"macro": 1.2, "domains": {"de": 1.1, "en": 1.3}}
    d = validation_derived(row)
    assert d["macro_ce"] == 1.2
    assert set(d["domain_perplexity"]) == {"de", "en"}
    assert d["domain_perplexity"]["de"] < d["domain_perplexity"]["en"]


def test_checkpoint_selection_respects_guard_and_small_real_improvement():
    assert should_select(1.1994, 1.2000, 0.0005, eligible=True)
    assert not should_select(1.1997, 1.2000, 0.0005, eligible=True)
    assert not should_select(1.1000, 1.2000, 0.0005, eligible=False)


def test_early_eval_schedule_is_denser_without_changing_training_steps():
    assert eval_due(100, 250, 100, 1000)
    assert eval_due(1000, 250, 100, 1000)
    assert not eval_due(1100, 250, 100, 1000)
    assert eval_due(1250, 250, 100, 1000)


def test_metric_trend_summary(tmp_path):
    p = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 100, "validation": {"macro": 1.30}},
        {"step": 200, "validation": {"macro": 1.25}},
        {"step": 300, "validation": {"macro": 1.20}},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    s = summarize_metrics(p, window=3)
    assert s["evaluations"] == 3
    assert s["best_ce"] == 1.20
    assert s["status"] == "improving"
