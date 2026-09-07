from __future__ import annotations

import math

import pytest

from build_targeted_sft50_oasst2 import (
    GERMAN_HUMAN_QUOTA,
    HUMAN_SOURCES,
    MEASURED_CLEAN_DE_TRAIN,
    WEIGHTS,
)
from fetch_oasst2_sft50 import UPSTREAM_REVISION, UPSTREAM_SHA256, choose_pairs, reviewed_human
from hf_source_policy50 import SOURCES, assert_not_reserved_eval, assert_training_allowed, validate_registry


def _message(mid, role, lang, text, parent=None, *, rank=None, synthetic=False, model_name=None, review=True, deleted=False):
    return {
        "message_id": mid,
        "parent_id": parent,
        "role": role,
        "lang": lang,
        "text": text,
        "rank": rank,
        "synthetic": synthetic,
        "model_name": model_name,
        "review_result": review,
        "deleted": deleted,
    }


def test_hf_registry_is_fail_closed_for_public_eval_data():
    audit = validate_registry()
    assert "OpenAssistant/oasst2" in audit["training_allowed"]
    assert {"openai/gsm8k", "google-research-datasets/mbpp", "deepmind/code_contests"}.issubset(audit["public_eval_holdouts"])
    assert_training_allowed("OpenAssistant/oasst2")
    with pytest.raises(ValueError):
        assert_training_allowed("openai/gsm8k")
    with pytest.raises(ValueError):
        assert_training_allowed("unknown/example")
    with pytest.raises(ValueError):
        assert_not_reserved_eval(["OpenAssistant/oasst2", "google-research-datasets/mbpp"])


def test_oasst2_filter_keeps_only_reviewed_non_synthetic_human_pairs():
    rows = [
        _message("p-de", "prompter", "de", "Erkläre Gravitation kurz."),
        _message("a-de-synth", "assistant", "de", "Synthetische Antwort", "p-de", rank=0, synthetic=True, model_name="some-model"),
        _message("a-de-human", "assistant", "de", "Gravitation ist die Anziehung zwischen Massen.", "p-de", rank=1),
        _message("a-de-worse", "assistant", "de", "Eine zweite menschliche Antwort.", "p-de", rank=2),
        _message("p-en", "prompter", "en", "What is force?"),
        _message("a-en", "assistant", "en", "Force is an interaction that can change motion.", "p-en", rank=0),
        _message("p-bad", "prompter", "en", "Bad prompt", review=False),
        _message("a-bad", "assistant", "en", "Should not survive", "p-bad", rank=0),
    ]
    pairs = choose_pairs(rows, max_en=100, max_de=100)
    assert len(pairs["de"]) == 1
    assert len(pairs["en"]) == 1
    assert pairs["de"][0]["answer"] == "Gravitation ist die Anziehung zwischen Massen."
    assert pairs["de"][0]["upstream_rank"] == 1
    assert all("user_id" not in row for lang in pairs.values() for row in lang)
    assert not reviewed_human(rows[1])
    assert reviewed_human(rows[2])


def test_oasst2_mix_preserves_targeted_curriculum_and_measured_capacity_headroom():
    assert math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(WEIGHTS[s] for s in HUMAN_SOURCES), 0.30, abs_tol=1e-12)
    assert WEIGHTS["oasst2-human-de"] == 0.0375
    assert GERMAN_HUMAN_QUOTA == 750
    assert MEASURED_CLEAN_DE_TRAIN == 805
    assert GERMAN_HUMAN_QUOTA < MEASURED_CLEAN_DE_TRAIN
    assert WEIGHTS["neral-generated-math-de"] + WEIGHTS["neral-generated-math-en"] == 0.24
    assert WEIGHTS["neral-generated-code-de"] + WEIGHTS["neral-generated-code-en"] == 0.20


def test_oasst2_pin_is_explicit_and_sha256_shaped():
    assert len(UPSTREAM_REVISION) == 40
    assert len(UPSTREAM_SHA256) == 64
    assert SOURCES["OpenAssistant/oasst2"].license == "Apache-2.0"
