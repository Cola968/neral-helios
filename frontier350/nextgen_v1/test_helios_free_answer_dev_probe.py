"""CPU-only code tests: no model checkpoint, training or evaluation claims."""
import unittest

import helios_free_answer_dev_probe as probe


class FreeAnswerDevelopmentContract(unittest.TestCase):
    def test_predeclared_cases_are_balanced(self):
        rows = probe.selected_cases()
        self.assertEqual(len(rows), 8)
        self.assertEqual({row[1] for row in rows},
                         {"arithmetic", "everyday", "astronomy", "aviation"})
        for domain in {row[1] for row in rows}:
            self.assertEqual({row[2] for row in rows if row[1] == domain}, {"de", "en"})

    def test_chat_instructions_match_forced_choice_dev_suite(self):
        self.assertEqual(probe.instruction("Wie viel ist 3 + 4?", "de"),
                         "Antworte mit genau einer Ziffer. Wie viel ist 3 + 4?")
        self.assertEqual(probe.instruction("What is 9 - 8?", "en"),
                         "Answer with exactly one digit. What is 9 - 8?")
        with self.assertRaises(ValueError):
            probe.instruction("Question", "fr")

    def test_strict_exact_never_counts_prefix_as_correct(self):
        self.assertTrue(probe.score(" 4 \n", "4")["strict_exact"])
        self.assertFalse(probe.score("4 apples", "4")["strict_exact"])
        self.assertTrue(probe.score("4 apples", "4")["starts_with_correct_digit"])
        self.assertFalse(probe.score("42", "4")["starts_with_correct_digit"])
        self.assertFalse(probe.score("", "4")["nonempty"])
        self.assertFalse(probe.score("wrong 4", "4")["strict_exact"])
        with self.assertRaises(ValueError):
            probe.score("4", "42")

    def test_aggregate_uses_only_full_answers_not_token_rankings(self):
        rows = []
        for index, case in enumerate(probe.selected_cases()):
            target = case[4]
            raw = probe.score(target if index == 0 else "0", target)
            guarded = probe.score(target + " words" if index == 0 else target, target)
            rows.append({"id": case[0],
                         "raw": {"score": raw},
                         "guarded": {"score": guarded, "finish_reason": "max_tokens"}})
        totals = probe.summarize(rows)
        self.assertEqual(totals["raw"]["strict_exact"], 1)
        self.assertEqual(totals["guarded"]["strict_exact"], 7)
        self.assertEqual(totals["guarded"]["starts_with_correct_digit"], 8)
        self.assertEqual(totals["guarded"]["finish_reasons"]["max_tokens"], 8)
        with self.assertRaises(ValueError):
            probe.summarize(rows[:-1])


if __name__ == "__main__":
    unittest.main()
