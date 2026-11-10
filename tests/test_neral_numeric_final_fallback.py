"""Regression contracts for bounded, unambiguous numeric final-answer extraction.

No arbitrary last-number extraction. Equivalent formatting is not an LLM oracle.
"""
import unittest
from neral_numeric_final import extract_numeric_final
from scripts.neral_same_base_ab_score import _correct
from neral_think import extract_final


class NumericFinalFallbackTests(unittest.TestCase):
    def test_explicit_final_markers_and_small_decorations(self):
        for text in ("FINAL: 42", "FINAL_ANSWER: 42", "**FINAL_ANSWER: 42**",
                     "FINAL_ANSWER = 42", "Antwort: 42", "Result: 42"):
            with self.subTest(text=text):
                self.assertEqual(extract_numeric_final(text), "42")
                self.assertTrue(_correct(42, text, True))

    def test_last_line_explicit_answer_after_a_working(self):
        for text in (
            "Rechne: 6 * 7\n42",
            "Zuerst 6, dann 7.\nDas Ergebnis beträgt 42.",
            "Zwischenergebnis 6.\nDie Antwort lautet 42.",
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_numeric_final(text), "42")
                self.assertTrue(_correct(42, text, True))

    def test_equation_mode_still_does_not_fake_a_bare_integer(self):
        self.assertEqual(extract_final("**FINAL_ANSWER: x = 4**"), "x = 4")
        self.assertIsNone(extract_numeric_final("x = 3; x = 4"))

    def test_never_extract_an_ambiguous_or_incidental_number(self):
        for text in (
            "Der Weg hat 6 Schritte, 7 Minuten und 42 km.",
            "Ergebnis 41 oder 42",
            "FINAL_ANSWER: 41\nFINAL_ANSWER: 42",
            "FINAL: 41\n42",
            "FINAL: 42\nWeitere Erklaerung.",
            "Das Ergebnis beträgt 42 oder 43.",
            "Hier der Code:\n```python\nprint(42)\n```",
            "Zuerst 7, danach 6 und am Ende 4.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_numeric_final(text))

    def test_reject_incomplete_numeric_finals_and_wrong_expected_gold(self):
        self.assertFalse(_correct(42, "FINAL_ANSWER: 42", False))
        self.assertFalse(_correct(43, "Das Ergebnis beträgt 42.", True))
        self.assertFalse(_correct(42, "Die Antwort lautet 41.", True))


if __name__ == "__main__":
    unittest.main()
