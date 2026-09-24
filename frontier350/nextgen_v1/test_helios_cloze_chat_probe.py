"""No GPU, checkpoint loading, data access, or model inference in these tests."""
import unittest

from helios_cloze_chat_probe import (
    PUBLIC_CLOZE_CASES, first_word, summarize, validate_public_cases,
)


class ClozeChatProbeTests(unittest.TestCase):
    def test_cases_are_locked_balanced_public_fixtures(self):
        validate_public_cases()
        self.assertEqual(len(PUBLIC_CLOZE_CASES), 8)
        self.assertEqual(sum(item[1] == "de" for item in PUBLIC_CLOZE_CASES), 4)
        self.assertEqual(sum(item[1] == "en" for item in PUBLIC_CLOZE_CASES), 4)

    def test_first_word_not_substring_anywhere(self):
        self.assertEqual(first_word('  "Berlin," is the city'), "berlin")
        self.assertEqual(first_word("  SONNE."), "sonne")
        self.assertEqual(first_word(" 4."), "4")
        self.assertIsNone(first_word("  ... "))
        self.assertNotEqual(first_word("It is Berlin"), "berlin")
        self.assertNotEqual(first_word("Assistant: Berlin"), "berlin")

    def test_aggregates_not_optimistic(self):
        rows = []
        for i in range(8):
            rows.append({"id": str(i),
                         "document_completion": {"first_word_match": i < 2,
                                                 "answer": "ok" if i != 7 else "",
                                                 "finish_reason": "eos" if i == 0 else "max_tokens"},
                         "chat_question": {"first_word_match": i == 0,
                                           "answer": "no", "finish_reason": "max_tokens"}})
        summary = summarize(rows)
        self.assertEqual(summary["document_completion"], {
            "first_word_matches_predeclared_answers": 2, "nonempty": 7, "native_eos": 1, "denominator": 8})
        self.assertEqual(summary["chat_question"], {
            "first_word_matches_predeclared_answers": 1, "nonempty": 8, "native_eos": 0, "denominator": 8})
        with self.assertRaises(ValueError):
            summarize(rows[:7])
        with self.assertRaises(ValueError):
            summarize(rows[:-1] + [rows[0]])


if __name__ == "__main__":
    unittest.main()
