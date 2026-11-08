"""RED contracts for LOW/HIGH depth: isolated local reasoning, no model training."""
from __future__ import annotations

import unittest

from neral_think import think_reply


def chat(question):
    return [
        {"role": "system", "content": "Du bist NERAL. Sage ehrlich, was du weisst."},
        {"role": "user", "content": question},
    ]


class HighDepthTests(unittest.TestCase):
    def test_non_oracle_high_runs_a_real_second_reasoning_and_revision_pass(self):
        calls = []
        answers = iter((
            "Erster Entwurf: F = m*a, deshalb ist die Einheit Meter.",
            "Korrigierte Erklärung: Die Krafteinheit ist Newton (kg*m/s²).",
        ))

        def local_model(turns):
            calls.append(turns)
            return next(answers)

        out = think_reply(chat("Erkläre die Einheit der Kraft ausführlich."),
                          local_model, deep_review=True)
        self.assertEqual(out["mode"], "HIGH")
        self.assertEqual(out["calls_used"], 2)
        self.assertEqual(out["review_passes"], 1)
        self.assertEqual(out["review_status"], "completed_unverified")
        self.assertEqual(out["text"], "Korrigierte Erklärung: Die Krafteinheit ist Newton (kg*m/s²).")
        self.assertFalse(out["verified"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][-2]["role"], "assistant")
        self.assertIn("Erster Entwurf", calls[1][-2]["content"])
        self.assertIn("prüf", calls[1][-1]["content"].lower())
        self.assertEqual(calls[0][-1]["role"], "user")

    def test_known_math_stays_exact_and_does_not_waste_a_revision_pass(self):
        calls = []
        def model(turns):
            calls.append(turns)
            return "FINAL: 42"
        out = think_reply(chat("Berechne schrittweise: 6*7"), model, deep_review=True)
        self.assertEqual(out["mode"], "HIGH")
        self.assertEqual(out["calls_used"], 1)
        self.assertTrue(out["verified"])
        self.assertEqual(out["review_passes"], 0)
        self.assertEqual(len(calls), 1)

    def test_second_pass_failure_preserves_first_answer_without_false_verification(self):
        calls = []
        def model(turns):
            calls.append(turns)
            if len(calls) == 2:
                raise RuntimeError("C:/private/secret/checkpoint.safetensors")
            return "Erster, unbestätigter Entwurf."
        out = think_reply(chat("Erkläre die Sternentwicklung."), model, deep_review=True)
        self.assertEqual(out["calls_used"], 2)
        self.assertEqual(out["text"], "Erster, unbestätigter Entwurf.")
        self.assertEqual(out["review_status"], "review_failed")
        self.assertFalse(out["verified"])
        self.assertNotIn("secret", str(out))
        self.assertEqual(out["review_passes"], 0)

    def test_second_pass_is_skipped_when_budget_has_elapsed(self):
        ticks = iter((0.0, 31.0, 31.5))
        calls = []
        def model(turns):
            calls.append(turns)
            return "Die kurze erste Antwort."
        out = think_reply(chat("Erkläre die Phasen eines Sterns."), model,
                          deep_review=True, clock=lambda: next(ticks), max_seconds=30)
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_status"], "skipped_time_budget")
        self.assertEqual(len(calls), 1)

    def test_long_draft_is_not_silently_truncated_for_inaccurate_review(self):
        calls = []
        def model(turns):
            calls.append(turns)
            return "L" * 7000
        out = think_reply(chat("Schreibe eine sehr lange Erklärung."), model,
                          deep_review=True)
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_status"], "skipped_long_draft")
        self.assertEqual(len(out["text"]), 7000)
        self.assertFalse(out["verified"])

    def test_default_core_think_behavior_is_backwards_compatible_without_deep_review(self):
        calls = []
        def model(turns):
            calls.append(turns)
            return "Physik ist eine Naturwissenschaft."
        out = think_reply(chat("Was ist Physik?"), model)
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_passes"], 0)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
