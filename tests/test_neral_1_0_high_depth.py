"""HIGH quality boundary: no unverified second pass or prompt mutation."""
from __future__ import annotations
import unittest
from neral_think import think_reply


def chat(question):
    return [
        {"role": "system", "content": "Du bist NERAL. Sage ehrlich, was du weisst."},
        {"role": "user", "content": question},
    ]


class HighDepthTests(unittest.TestCase):
    def test_open_question_preserves_first_complete_reply(self):
        original = chat("Erkläre die Einheit der Kraft ausführlich.")
        calls = []
        def local_model(turns):
            calls.append(turns)
            if len(calls) > 1:
                raise AssertionError("Unverified second model review forbidden")
            return "Die Krafteinheit ist Newton (kg*m/s²)."
        out = think_reply(original, local_model, deep_review=True)
        self.assertEqual(out["mode"], "HIGH")
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_passes"], 0)
        self.assertEqual(out["review_status"], "disabled_unverified_review")
        self.assertEqual(out["text"], "Die Krafteinheit ist Newton (kg*m/s²).")
        self.assertFalse(out["verified"])
        self.assertEqual(calls, [original])

    def test_known_math_is_verified_without_a_revision_pass(self):
        calls = []
        def model(turns):
            calls.append(turns)
            return "FINAL: 42"
        out = think_reply(chat("Berechne schrittweise: 6*7"), model, deep_review=True)
        self.assertEqual(out["mode"], "HIGH")
        self.assertEqual(out["calls_used"], 1)
        self.assertTrue(out["verified"])
        self.assertEqual(out["review_passes"], 0)
        self.assertEqual(out["review_status"], "not_needed_oracle")
        self.assertEqual(len(calls), 1)

    def test_open_ended_question_never_executes_a_second_unchecked_call(self):
        calls = []
        def model(turns):
            calls.append(turns)
            if len(calls) == 2:
                raise RuntimeError("C:/private/secret/checkpoint.safetensors")
            return "Erster, unbestätigter Entwurf."
        out = think_reply(chat("Erkläre die Sternentwicklung."), model,
                          deep_review=True)
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["text"], "Erster, unbestätigter Entwurf.")
        self.assertEqual(out["review_status"], "disabled_unverified_review")
        self.assertFalse(out["verified"])
        self.assertNotIn("secret", str(out))

    def test_time_budget_is_not_used_for_a_second_unverified_pass(self):
        ticks = iter((0.0, 31.0))
        calls = []
        def model(turns):
            calls.append(turns)
            return "Die kurze erste Antwort."
        out = think_reply(chat("Erkläre die Phasen eines Sterns."), model,
                          deep_review=True, clock=lambda: next(ticks), max_seconds=30)
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_status"], "disabled_unverified_review")
        self.assertEqual(len(calls), 1)

    def test_long_first_draft_kept_without_review_or_truncation(self):
        calls = []
        def model(turns):
            calls.append(turns)
            return "L" * 7000
        out = think_reply(chat("Schreibe eine sehr lange Erklärung."), model,
                          deep_review=True)
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_status"], "disabled_unverified_review")
        self.assertEqual(len(out["text"]), 7000)
        self.assertFalse(out["verified"])

    def test_legacy_core_think_without_deep_review_remains_available(self):
        calls = []
        def model(turns):
            calls.append(turns)
            return "Physik ist eine Naturwissenschaft."
        out = think_reply(chat("Was ist Physik?"), model)
        self.assertEqual(out["mode"], "THINK")
        self.assertEqual(out["calls_used"], 1)
        self.assertEqual(out["review_passes"], 0)
        self.assertIn("THINK (lokaler Prüfmodus)", calls[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
