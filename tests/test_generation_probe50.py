import pytest

from generation_probe50 import PROMPTS, diagnostics, parse_budgets, suite_sha256, summarize_rows


def test_probe_suite_is_stable_and_bilingual():
    assert len(suite_sha256()) == 64
    assert {p["language"] for p in PROMPTS} == {"de", "en"}
    assert len({p["id"] for p in PROMPTS}) == len(PROMPTS)


def test_budget_parser_is_strict():
    assert parse_budgets("64,128,256") == (64, 128, 256)
    with pytest.raises(ValueError):
        parse_budgets("128,64")
    with pytest.raises(ValueError):
        parse_budgets("64,64")
    with pytest.raises(ValueError):
        parse_budgets("0,64")


def test_diagnostics_detects_eos_and_repetition():
    # Byte-token IDs for ASCII are byte+4; EOS is 2.
    text = "alpha beta gamma alpha beta gamma"
    ids = [ord(c) + 4 for c in text] + [2]
    d = diagnostics(ids, text)
    assert d["eos"] is True
    assert d["valid_utf8"] is True
    assert d["repeated_trigram_fraction"] > 0
    assert d["top_trigram_count"] >= 2


def test_summary_reports_budget_health_and_prefix_consistency():
    rows = []
    for prompt in PROMPTS:
        for budget, ids in ((64, [10, 11]), (128, [10, 11, 12])):
            rows.append(prompt | {
                "budget": budget,
                "raw_token_ids": ids,
                "tokens_per_second": 100.0,
                "diagnostics": {
                    "valid_utf8": True,
                    "eos": False,
                    "generated_tokens": len(ids),
                    "repeated_trigram_fraction": 0.0,
                },
            })
    s = summarize_rows(rows, (64, 128))
    assert s["by_budget"]["64"]["responses"] == len(PROMPTS)
    assert s["greedy_prefix_consistency_fraction"] == 1.0
