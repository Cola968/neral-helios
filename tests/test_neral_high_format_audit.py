"""Offline audit of strict numerical final-answer parsing.

This test intentionally DOES NOT relax historic scorer contracts or re-score
past Hard36 runs. A genuinely ambiguous answer remains a format miss; unambiguous variants are accepted.
"""
import unittest
from neral_think import extract_final
from scripts.neral_same_base_ab_score import _correct


class FinalAnswerFormatAuditTests(unittest.TestCase):
    def test_supported_numeric_final_formats_are_equivalent(self):
        for response in ("FINAL: 42", "FINAL_ANSWER: 42",
                         "Schritt 1: 6*7\nFINAL: 42", "42"):
            with self.subTest(response=response):
                self.assertTrue(_correct(42, response, True), response)

    def test_explicit_result_in_terminal_prose_is_scored(self):
        response = "Das Ergebnis beträgt 42."
        self.assertTrue(_correct(42, response, True))
        self.assertEqual(extract_final(response), "42")

    def test_markdown_decorated_final_is_accepted_when_unique_and_terminal(self):
        response = "**FINAL_ANSWER: 42**"
        self.assertTrue(_correct(42, response, True))
        self.assertEqual(extract_final(response), "42")

    def test_incomplete_or_duplicate_final_lines_remain_invalid(self):
        for response, complete in (("FINAL: 42", False),
                                   ("FINAL: 41\nFINAL: 42", True),
                                   ("FINAL: 42\nextra", True)):
            with self.subTest(response=response):
                self.assertFalse(_correct(42, response, complete))


if __name__ == "__main__":
    unittest.main()
