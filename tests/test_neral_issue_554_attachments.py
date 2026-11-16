"""Issue 554: HIGH with files must use measured Ollama HIGH responder.

All tests use fake inference and local Flask. No GPU, file persistence or network.
"""
from io import BytesIO
import unittest
from unittest.mock import Mock

from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
from neral_preview_server import create_preview_app

HOST = "http://127.0.0.1:9690"
URL = "/api/neral/preview/attachment-chat"


class StubOllama(OllamaLocalBackend):
    def __init__(self, model, *, inference_profile="legacy"):
        super().__init__(model, inference_profile=inference_profile)
        self.high_messages = []
        self.think_calls = 0
        self.raise_error = False

    def unload(self):
        return None

    def respond_high_final(self, messages):
        self.high_messages.append(messages)
        if self.raise_error:
            raise OllamaCompletionError("token_limit")
        return "Ich nutze die Dateiinformation: RAUTE731."

    def respond_think(self, messages):
        self.think_calls += 1
        return "Alte allgemeine Antwort ohne Anhang."

    def respond_chat(self, messages):
        return "LOW"


class HighAttachments554(unittest.TestCase):
    def setUp(self):
        self.low = StubOllama("qwen3:8b")
        self.high = OllamaLocalBackend("qwen3.5:9b",
                inference_profile="measured_nonthinking_2048")
        self.high_messages = []
        self.raise_error = False
        def high_final(messages, *, cancel_event=None):
            self.high_messages.append(messages)
            if self.raise_error:
                raise OllamaCompletionError("token_limit")
            return "Ich nutze die Dateiinformation: RAUTE731."
        self.high.respond_high_final = Mock(side_effect=high_final)
        self.high.respond_think = Mock(return_value="Alte allgemeine Antwort ohne Anhang.")
        self.high.unload = Mock()
        self.app = create_preview_app(
            self.low, experimental_high_backend=self.high,
            secret_key=b"T" * 32,
            vision_analyzer=lambda raw,q: ("qwen3-vl:2b","Bild zeigt RAUTE731."),
        )
        self.client = self.app.test_client()
        self.client.get("/",base_url=HOST)
        self.csrf = self.client.get("/api/neral/preview/session",base_url=HOST).json["csrf"]

    def post(self,name="image.png",data=b"\x89PNG\r\n\x1a\nRAUTE731"):
        return self.client.post(URL,
            data={"prompt":"Was zeigt der Screenshot? Begründe es anhand der sichtbaren Details.",
                  "history":"[]","mode":"think",
                  "files":[(BytesIO(data),name)]},
            base_url=HOST,content_type="multipart/form-data",
            headers={"X-Neral-CSRF":self.csrf})

    def test_image_high_uses_same_finalizer_as_text_high_and_sees_image_context(self):
        response=self.post()
        self.assertEqual(response.status_code,200,response.get_json())
        self.assertEqual(response.json["text"],"Ich nutze die Dateiinformation: RAUTE731.")
        self.assertEqual(response.json["mode"],"HIGH")
        self.assertFalse(response.json["verified"])
        self.assertFalse(response.json["attachment_context_verified"])
        self.assertEqual(len(self.high_messages),1)
        self.assertIn("Bild zeigt RAUTE731",self.high_messages[0][-1]["content"])
        self.high.respond_think.assert_not_called()

    def test_stop_during_image_extraction_returns_safe_error_and_clears_busy_lock(self):
        request_id="a"*32
        # Use a separate app because the injected vision callable is fixed at setup.
        low=StubOllama("qwen3:8b")
        app=None
        def cancel_during_vision(raw,question):
            self.assertTrue(app._neral_chat_cancel_registry.cancel(request_id))
            return "qwen3-vl:2b","Bild zeigt RAUTE731."
        app=create_preview_app(low,secret_key=b"X"*32,vision_analyzer=cancel_during_vision)
        client=app.test_client()
        client.get("/",base_url=HOST)
        token=client.get("/api/neral/preview/session",base_url=HOST).json["csrf"]
        response=client.post(URL,
            data={"prompt":"Was siehst du?","history":"[]","mode":"generate",
                  "files":[(BytesIO(b"\\x89PNG\\r\\n\\x1a\\nRAUTE731"),"bild.png")]},
            headers={"X-Neral-CSRF":token,"X-Neral-Request-Id":request_id},
            content_type="multipart/form-data",base_url=HOST)
        self.assertEqual(response.status_code,409,response.get_json())
        self.assertEqual(response.json["error_code"],"cancelled")
        self.assertFalse(app._neral_chat_cancel_registry.cancel(request_id))

    def test_high_file_truncation_returns_specific_safe_error(self):
        self.raise_error=True
        response=self.post("notes.txt",b"RAUTE731 Textnotiz")
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.json["error_code"],"token_limit")
        self.assertIn("Ausgabelimit",response.json["error"])
        self.high.respond_think.assert_not_called()


if __name__ == "__main__":
    unittest.main()
