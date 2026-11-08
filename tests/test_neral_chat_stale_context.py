"""Regressionen für NERAL: Identität und frischer Gesprächsbezug (CPU-only)."""
import unittest
from flask import Flask
from neral_local_chat_api import register_neral_local_chat_api
from neral_think import think_reply


class LocalModel:
    model = "qwen3:8b"
    def __init__(self):
        self.calls = 0
    def respond_chat(self, messages):
        self.calls += 1
        return "Der Pratt & Whitney PW1000 ist ein Turbofan."
    def respond_think(self, messages):
        self.calls += 1
        return "Der Pratt & Whitney PW1000 ist ein Turbofan."
    respond = respond_chat


class ChatRegressionTests(unittest.TestCase):
    def test_runtime_identity_uses_real_active_model_without_repeating_previous_answer(self):
        backend = LocalModel()
        app = Flask(__name__)
        register_neral_local_chat_api(app, backend)
        client = app.test_client()
        history = [
            {"role": "user", "content": "Ist der PW1000 ein Turbofan?"},
            {"role": "assistant", "content": "Der Pratt & Whitney PW1000 ist ein Turbofan."},
        ]
        for path, question in [
            ("/api/neral/chat/generate", "Wer bist du?"),
            ("/api/neral/chat/think", "Nein, wie heißt du, welches Modell bist du?"),
        ]:
            response = client.post(path, json={"prompt": question, "history": history})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertIn("NERAL", response.json["text"])
            self.assertIn("qwen3:8b", response.json["text"])
            self.assertIn("Adapter", response.json["text"])
            self.assertNotIn("PW1000", response.json["text"])
            self.assertFalse(response.json["model_generated"])
        self.assertEqual(backend.calls, 0)

    def test_high_does_not_echo_previous_answer_on_new_unrelated_question(self):
        previous = "Der Pratt & Whitney PW1000 ist ein Getriebe-Turbofan und kein Turboprop."
        turns = [
            {"role":"system","content":"Du bist NERAL."},
            {"role":"user","content":"Erkläre den PW1000."},
            {"role":"assistant","content":previous},
            {"role":"user","content":"Was ist ein Schwarzes Loch?"},
        ]
        out = think_reply(turns, lambda _: previous, deep_review=True)
        self.assertNotIn("PW1000", out["text"])
        self.assertEqual(out["review_status"], "stale_context_rejected")
        self.assertFalse(out["verified"])

    def test_high_keeps_fresh_draft_when_self_review_regresses_to_previous_answer(self):
        previous = "Der Pratt & Whitney PW1000 ist ein Getriebe-Turbofan und kein Turboprop."
        turns = [
            {"role":"system","content":"Du bist NERAL."},
            {"role":"user","content":"Erkläre den PW1000."},
            {"role":"assistant","content":previous},
            {"role":"user","content":"Was ist ein Schwarzes Loch?"},
        ]
        responses = iter(("Ein Schwarzes Loch ist ein Bereich, aus dem kein Licht entkommt.", previous))
        out = think_reply(turns, lambda _: next(responses), deep_review=True)
        self.assertIn("Schwarzes Loch", out["text"])
        self.assertEqual(out["review_status"], "review_rejected_stale")


if __name__ == "__main__":
    unittest.main()
