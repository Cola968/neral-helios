"""HIGH returns the first model reply without an unchecked format rewrite."""
import json
import unittest
from neral_think import think_reply


def turns(question):
    return [
        {"role": "system", "content": "Du bist NERAL."},
        {"role": "user", "content": question},
    ]


def first_pass(question, draft):
    calls = []
    def respond(messages):
        calls.append(messages)
        if len(calls) > 1:
            raise AssertionError("Do not execute a second unverified answer")
        return draft
    result = think_reply(turns(question), respond, deep_review=True)
    return result, calls


class HighFormatRetentionTests(unittest.TestCase):
    def test_requested_python_code_fence_stays_intact(self):
        draft = "```python\ndef answer():\n    return 42\n```"
        result, calls = first_pass("Schreibe Python-Code für eine Funktion.", draft)
        self.assertEqual(result["text"], draft)
        self.assertEqual(result["calls_used"], 1)
        self.assertEqual(result["review_passes"], 0)
        self.assertEqual(result["review_status"], "disabled_unverified_review")
        self.assertFalse(result["verified"])
        self.assertEqual(calls, [turns("Schreibe Python-Code für eine Funktion.")])

    def test_json_only_stays_valid_json_and_remains_unverified(self):
        draft = '{"answer":42,"ok":true}'
        result, calls = first_pass("Antworte ausschließlich mit gültigem JSON.", draft)
        self.assertEqual(result["text"], draft)
        self.assertEqual(json.loads(result["text"])["answer"], 42)
        self.assertEqual(result["calls_used"], 1)
        self.assertFalse(result["verified"])
        self.assertEqual(len(calls), 1)

    def test_requested_language_not_rewritten(self):
        draft = "```python\nprint(42)\n```"
        result, calls = first_pass("Schreibe Python-Code für diese Aufgabe.", draft)
        self.assertIn("```python", result["text"])
        self.assertEqual(len(calls), 1)

    def test_normal_question_does_not_force_code_or_json_output(self):
        result, calls = first_pass(
            "Erkläre mir JSON anhand eines Beispiels.",
            "JSON ist ein Datenformat und benötigt ein gültiges Schema.",
        )
        self.assertEqual(result["text"],
                         "JSON ist ein Datenformat und benötigt ein gültiges Schema.")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
