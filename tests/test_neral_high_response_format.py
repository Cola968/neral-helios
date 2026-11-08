"""HIGH must not quietly destroy an explicitly requested output format."""
import json
import unittest

from neral_think import think_reply


def turns(question):
    return [
        {"role": "system", "content": "Du bist NERAL."},
        {"role": "user", "content": question},
    ]


def simulated_review(question, draft, revised):
    calls = []
    answers = iter((draft, revised))
    def respond(messages):
        calls.append(messages)
        return next(answers)
    result = think_reply(turns(question), respond, deep_review=True)
    return result, calls


class HighFormatRetentionTests(unittest.TestCase):
    def test_code_review_must_not_delete_requested_python_code(self):
        draft = "```python\ndef answer():\n    return 42\n```"
        revised = "Eine Funktion wäre möglich. Hier ist eine allgemeine Erklärung."
        result, calls = simulated_review(
            "Schreibe Python-Code für eine Funktion.", draft, revised)
        self.assertEqual(result["text"], draft)
        self.assertEqual(result["calls_used"], 2)
        self.assertEqual(result["review_passes"], 1)
        self.assertEqual(result["review_status"], "review_rejected_format")
        self.assertFalse(result["verified"])
        self.assertEqual(len(calls), 2)

    def test_code_review_may_repair_code_if_output_format_survives(self):
        draft = "```python\ndef answer():\n    return missing\n```"
        revised = "```python\ndef answer():\n    return 42\n```"
        result, calls = simulated_review(
            "Schreibe Python-Code für eine Funktion.", draft, revised)
        self.assertEqual(result["text"], revised)
        self.assertEqual(result["review_status"], "completed_unverified")
        self.assertEqual(len(calls), 2)

    def test_code_review_cannot_change_declared_language(self):
        draft = "```python\nprint(42)\n```"
        revised = "```javascript\nconsole.log(42);\n```"
        result, _ = simulated_review(
            "Schreibe Python-Code für diese Aufgabe.", draft, revised)
        self.assertEqual(result["text"], draft)
        self.assertEqual(result["review_status"], "review_rejected_format")

    def test_json_only_review_cannot_replace_valid_json_with_prose(self):
        draft = '{"answer":42,"ok":true}'
        revised = "Die Antwort lautet zweiundvierzig."
        result, calls = simulated_review(
            "Antworte ausschließlich mit gültigem JSON.", draft, revised)
        self.assertEqual(result["text"], draft)
        self.assertEqual(result["review_status"], "review_rejected_format")
        self.assertEqual(result["calls_used"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(result["text"])["answer"], 42)

    def test_json_only_review_can_replace_valid_json_with_valid_json(self):
        draft = '{"answer":41}'
        revised = '{"answer":42}'
        result, _ = simulated_review(
            "Antworte ausschließlich mit gültigem JSON.", draft, revised)
        self.assertEqual(json.loads(result["text"])["answer"], 42)
        self.assertEqual(result["review_status"], "completed_unverified")
        self.assertFalse(result["verified"])

    def test_explanation_about_json_is_not_treated_as_json_only_request(self):
        draft = '{"ok":true}'
        revised = "JSON ist ein Datenformat und benötigt ein gültiges Schema."
        result, _ = simulated_review(
            "Erkläre mir JSON anhand eines Beispiels.", draft, revised)
        self.assertEqual(result["text"], revised)
        self.assertEqual(result["review_status"], "completed_unverified")

    def test_non_code_question_can_be_revised_to_plain_text(self):
        draft = "```python\nprint(42)\n```"
        revised = "Ein Python-Beispiel hilft, aber die Erklärung lautet anders."
        result, _ = simulated_review(
            "Erkläre mir die Programmiergeschichte.", draft, revised)
        self.assertEqual(result["text"], revised)
        self.assertEqual(result["review_status"], "completed_unverified")


if __name__ == "__main__":
    unittest.main()
