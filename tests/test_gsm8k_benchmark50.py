from fractions import Fraction

from benchmark_gsm8k50 import canonical_number, extract_prediction, prompt_for


def test_canonical_number_handles_common_gsm8k_formats():
    assert canonical_number("1,250") == Fraction(1250, 1)
    assert canonical_number("-4.5") == Fraction(-9, 2)
    assert canonical_number("3/4") == Fraction(3, 4)
    assert canonical_number("$42.") == Fraction(42, 1)
    assert canonical_number("not-a-number") is None


def test_extraction_prefers_explicit_answer_marker():
    raw, value, mode = extract_prediction("12 + 30 = 42. Answer: 42")
    assert raw == "42"
    assert value == Fraction(42, 1)
    assert mode == "answer_marker"


def test_extraction_has_auditable_last_number_fallback():
    raw, value, mode = extract_prediction("The calculation gives 19.")
    assert raw == "19"
    assert value == Fraction(19, 1)
    assert mode == "last_number_fallback"
    assert extract_prediction("No numeric output here") == (None, None, "no_number")


def test_prompt_contract_is_fixed_and_tool_free():
    p = prompt_for("What is 2 + 2?")
    assert p.startswith("User: Solve this math problem carefully.")
    assert "Answer: <number>" in p
    assert p.endswith("\nAssistant:")
    assert "SOLVE" not in p
    assert "Helios" not in p
