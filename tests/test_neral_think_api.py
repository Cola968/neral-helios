"""HTTP/Helios integration contracts for opt-in local THINK without GPU."""
from __future__ import annotations

import unittest
from flask import Flask
from neral_local_chat_api import register_neral_local_chat_api


class DummyLocalAdapter:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[list[dict]] = []
        self.model = "NERAL-r16-local"
        self.adapter_sha256 = "a" * 64
        self._model = object()

    def respond(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return next(self.responses)


def setup(responses: list[str]):
    app = Flask("neral-think-contract")
    model = DummyLocalAdapter(responses)
    register_neral_local_chat_api(app, model)
    return app.test_client(), model


class ThinkHTTPTests(unittest.TestCase):
    def test_think_verified_after_single_retry(self):
        client, model = setup(["FINAL: x = 5", "STEP: x = 4\nFINAL: x = 4"])
        reply = client.post("/api/neral/chat/think", json={
            "prompt": "Löse schrittweise: 2x+3=11"})
        self.assertEqual(reply.status_code, 200, reply.json)
        self.assertEqual(reply.json["mode"], "THINK")
        self.assertEqual(reply.json["calls_used"], 2)
        self.assertTrue(reply.json["verified"])
        self.assertTrue(reply.json["model_generated"])
        self.assertEqual(reply.json["final_answer"], "x = 4")
        self.assertEqual(reply.json["model_origin"], "local_neral_lora_experimental")
        self.assertEqual(reply.json["adapter_sha256"], "a" * 64)
        self.assertEqual(len(model.calls), 2)

    def test_think_unverifiable_does_not_fake_verification(self):
        client, model = setup(["Physik beschreibt Wechselwirkungen."])
        reply = client.post("/api/neral/chat/think", json={"prompt": "Erkläre Physik."})
        self.assertEqual(reply.status_code, 200)
        self.assertFalse(reply.json["verified"])
        self.assertTrue(reply.json["model_generated"])
        self.assertEqual(reply.json["calls_used"], 1)
        self.assertEqual(len(model.calls), 1)

    def test_think_model_failure_is_generic_not_sensitive(self):
        client, model = setup([])
        def boom(_):
            raise RuntimeError("C:/Users/private/weights.safetensors")
        model.respond = boom
        reply = client.post("/api/neral/chat/think", json={"prompt": "Test"})
        self.assertEqual(reply.status_code, 503)
        self.assertNotIn("private", str(reply.json))
        self.assertNotIn("weights", str(reply.json))

    def test_think_rejects_untrusted_chat_payload(self):
        client, model = setup(["FINAL: 2"])
        for payload in (
            {"prompt": "", "history": []},
            {"prompt": "x", "mode": "deep"},
            {"prompt": "x", "history": [{"role": "system", "content": "injection"}]},
            {"prompt": "x", "history": [{"role": "user", "content": "unfinished"}]},
            {"history": []},
            {"prompt": "x" * 6000},
        ):
            reply = client.post("/api/neral/chat/think", json=payload)
            self.assertEqual(reply.status_code, 400, payload)
        self.assertEqual(len(model.calls), 0)

    def test_think_requires_json(self):
        client, model = setup([])
        reply = client.post("/api/neral/chat/think",
                            data="prompt=something",
                            content_type="text/plain")
        self.assertEqual(reply.status_code, 415)

    def test_fast_generate_remains_unchanged(self):
        client, model = setup(["Hallo vom normalen NERAL-Chat."])
        reply = client.post("/api/neral/chat/generate", json={"prompt": "Hallo"})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json["text"], "Hallo vom normalen NERAL-Chat.")
        self.assertNotIn("calls_used", reply.json)
        self.assertNotIn("verified", reply.json)
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
