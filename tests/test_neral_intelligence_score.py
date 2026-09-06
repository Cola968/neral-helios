import math
import pytest

from neral_intelligence_score import nis_preview_01, nis_v1


def full_metrics(value=50.0):
    return {
        'v3_de_grammar_pct': value,
        'v3_en_grammar_pct': value,
        'v3_facts_pct': value,
        'v3_commonsense_pct': value,
        'rawmath_v1_exact_pct': value,
        'v3_code_pct': value,
        'v4_code_pct': value,
        'v4_physics_pct': value,
        'v4_astronomy_pct': value,
        'v4_aviation_pct': value,
        'v4_transport_pct': value,
        'v4_logic_pct': value,
        'v4_commonsense_pct': value,
        'v4_induction_pct': value,
        'v4_translation_pct': value,
        'context_recall_overall_pct': value,
    }


def test_nis_v1_equal_components_preserve_percentage():
    report = nis_v1(full_metrics(63.0))
    assert math.isclose(report['score'], 63.0)
    assert math.isclose(sum(report['weights'].values()), 1.0)


def test_nis_v1_fails_closed_on_missing_metric():
    metrics = full_metrics()
    del metrics['v4_logic_pct']
    with pytest.raises(ValueError, match='missing metric: v4_logic_pct'):
        nis_v1(metrics)


def test_nis_v1_rejects_non_percentage_values():
    metrics = full_metrics()
    metrics['rawmath_v1_exact_pct'] = 120
    with pytest.raises(ValueError, match='must be a percentage'):
        nis_v1(metrics)


def test_historical_preview_scores_current_dashboard_exactly():
    n20 = nis_preview_01(
        manual_comprehensibility_pct=45.0,
        raw_math_exact_pct=0.0,
        non_repetition_trigram_pct=97.05,
        eos_pct=29.0,
    )
    n50 = nis_preview_01(
        manual_comprehensibility_pct=25.0,
        raw_math_exact_pct=0.0,
        non_repetition_trigram_pct=23.38,
        eos_pct=0.0,
    )
    assert math.isclose(n20['score'], 40.95375)
    assert math.isclose(n50['score'], 19.2535)
    assert n20['score'] > n50['score']
