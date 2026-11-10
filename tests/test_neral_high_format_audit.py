"""Offline audit of strict numerical final-answer parsing.

This test intentionally DOES NOT relax historic scorer contracts or re-score
past Hard36 runs. Right answer in a nonconforming format is still a format miss.
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

    def test_correct_number_in_prose_is_not_counted_as_correct(self):
        response = "Das Ergebnis beträgt 42."
        self.assertFalse(_correct(42, response, True))
        self.assertIsNone(extract_final(response))

    def test_markdown_decorated_final_is_rejected_by_strict_benchmark_contract(self):
        response = "**FINAL_ANSWER: 42**"
        self.assertFalse(_correct(42, response, True))
        self.assertIsNone(extract_final(response))

    def test_incomplete_or_duplicate_final_lines_remain_invalid(self):
        for response, complete in (("FINAL: 42", False),
                                   ("FINAL: 41\nFINAL: 42", True),
                                   ("FINAL: 42\nextra", True)):
            with self.subTest(response=response):
                self.assertFalse(_correct(42, response, complete))


if __name__ == "__main__":
    unittest.main()
