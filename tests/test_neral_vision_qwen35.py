"""Image uploads must use installed Qwen 3.5 vision without extra downloads.

CPU-only tests: model inventory and model HTTP are mocked, but actual
Flask attachment routing and validation are exercised.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import unittest
from unittest.mock import patch

from neral_preview_attachments import (
    VisionUnavailable, _installed_vision_model, _local_vision,
)
from neral_preview_server import create_preview_app
from scripts.neral_1_0_local_live_smoke import (
    _red_png_test_image, verify_live_vision_attachment,
)

HOST = "http://127.0.0.1:9690"
ROUTE = "/api/neral/preview/attachment-chat"


class FakeModel:
    model = "qwen3:8b"
    def __init__(self):
        self.calls = []
    def respond_chat(self, turns):
        self.calls.append(turns)
        return "Auf dem Bild ist eine rote Fläche zu sehen."
    respond = respond_chat


class InstalledVisionRegressionTests(unittest.TestCase):
    def test_already_installed_qwen35_9b_is_selected_instead_of_new_download(self):
        inventory = {"models": [
            {"name":"qwen3:8b"}, {"name":"qwen3.5:9b"}
        ]}
        with patch("neral_preview_attachments._local_open", return_value=inventory) as send:
            chosen = _installed_vision_model()
        self.assertEqual(chosen, "qwen3.5:9b")
        self.assertEqual(send.call_count, 1)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/tags")
        self.assertIsNone(request.data)

    def test_only_unrecognized_text_model_still_requires_real_vision(self):
        inventory = {"models": [{"name":"qwen3:8b"}, {"name":"any-nonvision:latest"}]}
        with patch("neral_preview_attachments._local_open", return_value=inventory):
            with self.assertRaises(VisionUnavailable):
                _installed_vision_model()

    def test_installed_qwen35_receives_real_image_bytes_and_question_locally(self):
        png = _red_png_test_image()
        inventory = {"models": [{"name":"qwen3:8b"}, {"name":"qwen3.5:9b"}]}
        answer = {"done":True,"done_reason":"stop", "message":{
            "role":"assistant", "content":"Das Bild zeigt eine rote Fläche."}}
        with patch("neral_preview_attachments._local_open", side_effect=[inventory, answer]) as send:
            model, text = _local_vision(png, "Welche Farbe siehst du?")
        self.assertEqual(model, "qwen3.5:9b")
        self.assertIn("rote Fläche", text)
        first, second = (call.args[0] for call in send.call_args_list)
        self.assertEqual(first.full_url, "http://127.0.0.1:11434/api/tags")
        self.assertEqual(second.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(second.get_method(), "POST")
        payload = json.loads(second.data)
        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertGreaterEqual(payload["options"]["num_predict"], 900)
        self.assertIn("Welche Farbe siehst du?", payload["messages"][0]["content"])
        self.assertEqual(base64.b64decode(payload["messages"][0]["images"][0]), png)

    def test_real_flask_image_route_accepts_qwen35_analysis_and_uses_it(self):
        backend = FakeModel()
        app = create_preview_app(
            backend, secret_key=b"Q"*32,
            vision_analyzer=lambda png, prompt: ("qwen3.5:9b", "Das Bild ist rot."),
        )
        client = app.test_client()
        client.get("/", base_url=HOST)
        csrf = client.get("/api/neral/preview/session", base_url=HOST).json["csrf"]
        result = client.post(
            ROUTE, base_url=HOST, content_type="multipart/form-data",
            headers={"X-Neral-CSRF": csrf},
            data={"prompt": "Welche Farbe ist das?", "mode":"generate",
                  "history": "[]",
                  "files": [(BytesIO(_red_png_test_image()), "rot.png")]},
        )
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertEqual(result.json["vision_models"], ["qwen3.5:9b"])
        self.assertIn("Das Bild ist rot.", backend.calls[0][-1]["content"])
        self.assertEqual(result.json["attachments"][0]["kind"], "image")
        self.assertIs(result.json["file_bytes_persisted"], False)
        self.assertFalse(result.json["adapter_loaded"])

    def test_optional_release_vision_gate_recognizes_qwen35_as_real_vision(self):
        backend = FakeModel()
        app = create_preview_app(
            backend, secret_key=b"T"*32,
            vision_analyzer=lambda *_: ("qwen3.5:9b", "Das Bild zeigt Rot."),
        )
        client = app.test_client()
        client.get("/",base_url=HOST)
        csrf = client.get("/api/neral/preview/session",base_url=HOST).json["csrf"]
        self.assertTrue(verify_live_vision_attachment(
            client,csrf,{"qwen3.5:9b":"sha256:fixture", "qwen3:8b":"sha256:fixture"},
        ))


if __name__ == "__main__":
    unittest.main()
