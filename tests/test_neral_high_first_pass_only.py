"""Step A RED: HIGH must preserve the base model's first complete reply.

No GPU access, model-weight changes, or benchmark-specific question routing.
"""
import unittest
from neral_think import think_reply


def conversation(question):
    return [
        {"role": "system", "content": "Du bist NERAL. Beantworte die Nutzerfrage."},
        {"role": "user", "content": question},
    ]


class HighFirstPassOnlyTests(unittest.TestCase):
    def test_open_question_does_not_replace_correct_first_reply(self):
        original = conversation("Welche Einheit hat die Kraft? Erkläre.")
        calls = []

        def model(messages):
            calls.append(messages)
            return ("Die SI-Einheit der Kraft ist Newton (N)." if len(calls) == 1
                    else "Falsch: Meter.")

        result = think_reply(original, model, deep_review=True)
        self.assertEqual(result["mode"], "HIGH")
        self.assertEqual(result["text"], "Die SI-Einheit der Kraft ist Newton (N).")
        self.assertEqual(result["calls_used"], 1)
        self.assertEqual(result["review_passes"], 0)
        self.assertEqual(result["review_status"], "disabled_unverified_review")
        self.assertFalse(result["verified"])
        self.assertEqual(calls, [original])

    def test_high_preserves_model_messages_without_an_extra_system_suffix(self):
        original = conversation("Erkläre kurz die Keplerschen Gesetze.")
        original_snapshot = [dict(row) for row in original]
        seen = []

        def model(messages):
            seen.append([dict(row) for row in messages])
            return "Drei Gesetze beschreiben die Bewegung der Planeten."

        result = think_reply(original, model, deep_review=True)
        self.assertEqual(seen, [original_snapshot])
        self.assertEqual(original, original_snapshot)
        self.assertEqual(result["calls_used"], 1)

    def test_code_and_json_do_not_get_rewritten_by_an_unverified_review(self):
        for question, original_response in (
            ("Schreibe Python-Code für eine Funktion.", "```python\nprint(42)\n```"),
            ("Antworte ausschließlich mit gültigem JSON.", '{"result":42}'),
        ):
            with self.subTest(question=question):
                calls = []
                def model(messages):
                    calls.append(messages)
                    if len(calls) > 1:
                        raise AssertionError("An unverified second pass must not run")
                    return original_response
                result = think_reply(conversation(question), model, deep_review=True)
                self.assertEqual(result["text"], original_response)
                self.assertEqual(result["calls_used"], 1)
                self.assertEqual(len(calls), 1)

    def test_exact_math_oracle_stays_available_without_unverified_review(self):
        calls = []
        def model(messages):
            calls.append(messages)
            return "FINAL: 42"
        result = think_reply(conversation("Berechne schrittweise: 6*7"), model,
                             deep_review=True)
        self.assertEqual(result["mode"], "HIGH")
        self.assertEqual(result["calls_used"], 1)
        self.assertTrue(result["verified"])
        self.assertEqual(result["final_answer"], "42")
        self.assertEqual(calls, [conversation("Berechne schrittweise: 6*7")])


if __name__ == "__main__":
    unittest.main()
