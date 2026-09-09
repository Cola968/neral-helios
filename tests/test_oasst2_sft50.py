from build_targeted_sft50_oasst2 import (
    GERMAN_HUMAN_QUOTA,
    HUMAN_SOURCES,
    MEASURED_CLEAN_DE_TRAIN,
    WEIGHTS,
)
from fetch_oasst2_sft50 import choose_pairs, reviewed_human


def _row(message_id, role, lang, text, *, parent_id=None, rank=None, synthetic=False, review_result=True, deleted=False, model_name=None):
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "role": role,
        "lang": lang,
        "text": text,
        "rank": rank,
        "synthetic": synthetic,
        "review_result": review_result,
        "deleted": deleted,
        "model_name": model_name,
    }


def test_reviewed_human_is_fail_closed():
    base = _row("a", "assistant", "de", "Antwort")
    assert reviewed_human(base) is True
    assert reviewed_human(base | {"synthetic": True}) is False
    assert reviewed_human(base | {"review_result": False}) is False
    assert reviewed_human(base | {"deleted": True}) is False
    assert reviewed_human(base | {"model_name": "some-model"}) is False
    assert reviewed_human(base | {"synthetic": None}) is False


def test_choose_pairs_keeps_best_reviewed_human_answer_only():
    rows = [
        _row("p-de", "prompter", "de", "Erkläre Gravitation einfach."),
        _row("a-de-r1", "assistant", "de", "Gute menschliche Antwort.", parent_id="p-de", rank=1),
        _row("a-de-r0", "assistant", "de", "Beste menschliche Antwort.", parent_id="p-de", rank=0),
        _row("a-de-synth", "assistant", "de", "Synthetisch.", parent_id="p-de", rank=-1, synthetic=True),
        _row("p-en", "prompter", "en", "Explain orbital speed."),
        _row("a-en", "assistant", "en", "A reviewed human answer.", parent_id="p-en", rank=0),
    ]
    pairs = choose_pairs(rows, max_en=10, max_de=10)
    assert len(pairs["de"]) == 1
    assert pairs["de"][0]["answer"] == "Beste menschliche Antwort."
    assert len(pairs["en"]) == 1
    assert pairs["en"][0]["answer"] == "A reviewed human answer."


def test_targeted_oasst2_mix_contract():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12
    assert abs(sum(WEIGHTS[source] for source in HUMAN_SOURCES) - 0.30) < 1e-12
    assert GERMAN_HUMAN_QUOTA == 750
    assert GERMAN_HUMAN_QUOTA < MEASURED_CLEAN_DE_TRAIN
    assert WEIGHTS["neral-generated-de-core"] >= 0.20
    assert WEIGHTS["neral-generated-math-de"] > 0
    assert WEIGHTS["neral-generated-code-de"] > 0
