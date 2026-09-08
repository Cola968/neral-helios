import math

import pytest

from benchmark_nis2_chat50 import EXACT_GENERATION_CASES, PAIR_CASES, SUITE_VERSION, suite_sha256
from hf_source_policy50 import SOURCES, assert_training_allowed
from neral_intelligence_score_v2 import WEIGHTS, generation_health, score_from_components
from public_eval_registry50 import EVALS, validate_registry


def test_nis2_weights_and_balance_contract():
    assert math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-12)
    perfect = {k: 100.0 for k in WEIGHTS}
    report = score_from_components(perfect)
    assert report["version"] == "NIS-2.0"
    assert report["score"] == pytest.approx(100.0)
    assert report["coverage_pct"] == 100.0


def test_nis2_balance_penalizes_hidden_capability_hole():
    flat = {k: 60.0 for k in WEIGHTS}
    uneven = dict(flat)
    uneven["math_generalization"] = 5.0
    # Compensate arithmetic mean roughly elsewhere, but the balance term must still punish the hole.
    uneven["language_knowledge"] = 100.0
    uneven["science_domain"] = 100.0
    a = score_from_components(flat)
    b = score_from_components(uneven)
    assert b["balance_component"] < a["balance_component"]
    assert b["bottleneck"]["component"] == "math_generalization"


def test_generation_health_uses_explicit_targets():
    perfect = generation_health({"free_eos_pct": 60.0, "free_repeated_trigram_pct": 0.0, "utf8_pct": 100.0})
    weak = generation_health({"free_eos_pct": 10.0, "free_repeated_trigram_pct": 15.0, "utf8_pct": 99.0})
    assert perfect == pytest.approx(100.0)
    assert weak < perfect


def test_frozen_chat_suite_has_all_new_domains_and_stable_shape():
    assert SUITE_VERSION == "nis2-chat-core-v1"
    assert len(PAIR_CASES) == 40
    assert len(EXACT_GENERATION_CASES) == 8
    categories = {row[0] for row in PAIR_CASES}
    assert categories == {"instruction_following", "dialogue_state", "truthfulness", "robustness", "compositional_reasoning"}
    assert len(suite_sha256()) == 64
    assert len({(row[1], row[2], row[3]) for row in PAIR_CASES}) == len(PAIR_CASES)


def test_new_hf_public_evals_are_fail_closed():
    registry = validate_registry()
    for key in ("ifeval", "truthfulqa", "boolq", "arc"):
        dataset_id = EVALS[key].dataset_id
        assert key in registry
        assert SOURCES[dataset_id].role == "public-eval-holdout"
        assert SOURCES[dataset_id].allow_training is False
        with pytest.raises(ValueError):
            assert_training_allowed(dataset_id)
