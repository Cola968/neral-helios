"""Follow-up audit of actual model provenance and image focus, CPU-only."""
from __future__ import annotations

from io import BytesIO
import json
import unittest

from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app

HOST = "http://127.0.0.1:9690"


class FakeOllama(OllamaLocalBackend):
    def __init__(self, model, *, profile="legacy"):
        super().__init__(model, inference_profile=profile)
        self.calls = []
    def respond_chat(self, turns):
        self.calls.append(turns)
        return "Aktuelle Bildfrage beantwortet."
    def respond_think(self, turns):
        self.calls.append(turns)
        return "Aktuelle Bildfrage beantwortet."
    def unload(self):
        pass


def client_for(low, high=None, *, vision=None):
    app = create_preview_app(
        low, experimental_high_backend=high, vision_analyzer=vision,
        secret_key=b"Q"*32,
    )
    client = app.test_client()
    client.get("/", base_url=HOST)
    token = client.get("/api/neral/preview/session", base_url=HOST).json["csrf"]
    return client, {"X-Neral-CSRF": token}


class ChatReauditTests(unittest.TestCase):
    def test_high_identity_reports_the_actual_high_model_not_low(self):
        low = FakeOllama("qwen3:8b")
        high = FakeOllama("qwen3.5:9b", profile="measured_nonthinking_2048")
        client, headers = client_for(low, high)
        history = [
            {"role":"user","content":"Erkläre mir einen Stern."},
            {"role":"assistant","content":"Ein Stern erzeugt Strahlung."},
        ]
        reply = client.post("/api/neral/chat/think",
            json={"prompt":"Welches Modell bist du?","history":history},
            headers=headers, base_url=HOST)
        self.assertEqual(reply.status_code, 200, reply.get_json())
        self.assertEqual(reply.json["model"], "qwen3.5:9b")
        self.assertEqual(reply.json["mode"], "HIGH")
        self.assertIn("qwen3.5:9b", reply.json["text"])
        self.assertNotIn("qwen3:8b", reply.json["text"])
        self.assertFalse(reply.json["model_generated"])
        self.assertEqual(low.calls, [])
        self.assertEqual(high.calls, [])

    def test_low_identity_remains_true_to_qwen3_8b(self):
        low = FakeOllama("qwen3:8b")
        high = FakeOllama("qwen3.5:9b", profile="measured_nonthinking_2048")
        client, headers = client_for(low, high)
        reply = client.post("/api/neral/chat/generate",
            json={"prompt":"Welches Modell bist du?"},
            headers=headers, base_url=HOST)
        self.assertEqual(reply.status_code, 200, reply.get_json())
        self.assertEqual(reply.json["model"], "qwen3:8b")
        self.assertIn("qwen3:8b", reply.json["text"])

    def test_generic_image_request_does_not_reanswer_previous_physics_question(self):
        low = FakeOllama("qwen3:8b")
        client, headers = client_for(
            low, vision=lambda raw, prompt: ("qwen3.5:9b", "Auf dem Bild steht ROTER KREIS."),
        )
        history = [
            {"role":"user","content":"Erkläre elektrische Felder und Stromkreise."},
            {"role":"assistant","content":"Elektrische Felder bestimmen die Kraft auf Ladungen."},
        ]
        reply = client.post("/api/neral/preview/attachment-chat",
            base_url=HOST, headers=headers, content_type="multipart/form-data",
            data={"prompt":"Analysiere die Anhänge.", "history":json.dumps(history),
                "mode":"generate", "files":[
                    (BytesIO(b"\x89PNG\\r\\n\x1a\\n" + b"image-bytes"), "kreis.png")
                ]})
        self.assertEqual(reply.status_code, 200, reply.get_json())
        self.assertEqual(len(low.calls), 1)
        turns = low.calls[0]
        self.assertEqual(len(turns), 2, "A generic new image analysis should not replay unrelated old questions")
        self.assertIn("ROTER KREIS", turns[-1]["content"])
        self.assertIn("Analysiere die Anhänge", turns[-1]["content"])
        self.assertEqual(reply.json["vision_models"], ["qwen3.5:9b"])

    def test_explicit_image_followup_may_use_previous_context(self):
        low = FakeOllama("qwen3:8b")
        client, headers = client_for(low,
            vision=lambda *_: ("qwen3.5:9b", "Bild zeigt Stromkreis mit Batterie."))
        history = [
            {"role":"user","content":"Was ist Strom?"},
            {"role":"assistant","content":"Strom ist bewegte Ladung."},
        ]
        reply = client.post("/api/neral/preview/attachment-chat",
            base_url=HOST, headers=headers, content_type="multipart/form-data",
            data={"prompt":"Wie passt dieses Bild zur vorherigen Erklärung?",
                  "history":json.dumps(history), "mode":"generate",
                  "files":[(BytesIO(b"\x89PNG\\r\\n\x1a\\n" + b"image-bytes"), "strom.png")]})
        self.assertEqual(reply.status_code, 200, reply.get_json())
        turns = low.calls[0]
        self.assertEqual(len(turns), 4)
        self.assertIn("Strom ist bewegte Ladung.", turns[2]["content"])
        self.assertIn("Bild zeigt Stromkreis", turns[-1]["content"])


if __name__ == "__main__":
    unittest.main()
