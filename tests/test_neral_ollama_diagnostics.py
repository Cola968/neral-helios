"""CPU-only regression for Qwen3.5 measured Ollama request/response diagnostics.

Never log, persist or expose user prompts, content, thinking or raw Ollama errors.
"""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from neral_agent_autonomy import (
    OllamaLocalBackend, OllamaCompletionError, AgentModelError,
)


class _LocalResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return json.dumps(self.body).encode("utf-8")


class MeasuredOllamaDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.backend = OllamaLocalBackend(
            "qwen3.5:9b", timeout=90,
            inference_profile="measured_nonthinking_2048")
        self.messages = [
            {"role":"system", "content":"safe local test"},
            {"role":"user", "content":"private prompt never in logs"},
        ]

    def test_thinking_only_empty_answer_captures_lengths_without_exposing_text(self):
        private_thought = "SECRET_THINKING_NOT_FOR_LOGS"
        class Local:
            def open(self, request, timeout):
                return _LocalResponse({
                    "done": True, "done_reason": "stop",
                    "message": {"content": "", "thinking": private_thought},
                })
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            with self.assertRaises(OllamaCompletionError) as raised:
                self.backend.respond_chat(self.messages)
        self.assertEqual(raised.exception.reason, "empty_answer")
        diag = self.backend.last_inference_diagnostic
        self.assertEqual(diag["ollama_http_status"], 200)
        self.assertEqual(diag["ollama_error_type"], "thinking_only")
        self.assertEqual(diag["content_length"], 0)
        self.assertEqual(diag["thinking_length"], len(private_thought))
        self.assertEqual(diag["think"], False)
        self.assertEqual(diag["endpoint"], "/api/chat")
        self.assertEqual(diag["option_keys"], sorted([
            "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "repeat_penalty", "num_predict", "num_ctx"]))
        self.assertNotIn(private_thought, str(diag))
        self.assertNotIn("private prompt", str(diag))
        self.assertEqual(len(self.backend.inference_diagnostic_events), 1)

    def test_http_400_invalid_option_exposes_status_and_sanitized_type(self):
        private_error = 'unsupported parameter: presence_penalty; SECRET_PROMPT_TEXT'
        class Local:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(
                    request.full_url, 400, "Bad Request", {},
                    io.BytesIO(json.dumps({"error": private_error}).encode("utf-8")))
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            with self.assertRaises(AgentModelError):
                self.backend.respond_chat(self.messages)
        diag = self.backend.last_inference_diagnostic
        self.assertEqual(diag["ollama_http_status"], 400)
        self.assertEqual(diag["ollama_error_type"], "invalid_option")
        self.assertNotIn("SECRET_PROMPT_TEXT", str(diag))
        self.assertNotIn("presence_penalty; SECRET", str(diag))

    def test_no_proxy_error_is_classified_without_error_body(self):
        class Local:
            def open(self, request, timeout):
                raise urllib.error.URLError("SECRET_HOST")
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            with self.assertRaises(AgentModelError):
                self.backend.respond_chat(self.messages)
        diag = self.backend.last_inference_diagnostic
        self.assertEqual(diag["ollama_error_type"], "transport_error")
        self.assertIsNone(diag["ollama_http_status"])
        self.assertNotIn("SECRET_HOST", str(diag))

    def test_direct_qwen_and_tir_http_route_send_identical_model_options(self):
        from scripts.neral_same_base_live import BASE_URL, _high_client
        from unittest.mock import patch as mock_patch
        sent = []
        class Local:
            def open(self, request, timeout):
                payload = json.loads(request.data)
                sent.append({"endpoint":request.full_url, "payload":payload})
                return _LocalResponse({
                    "done": True, "done_reason": "stop",
                    "message": {"content": "It is a simple counting problem.", "thinking": ""},
                })
        with mock_patch.dict("os.environ", {"NERAL_EXPERIMENTAL_TIR":"1"}):
            with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
                self.backend.respond_chat(self.messages)
                client,headers = _high_client(self.backend)
                response = client.post(
                    "/api/neral/chat/think",
                    json={"prompt":"Explain why a local clock advances.",
                          "experimental_tir":True},
                    base_url=BASE_URL,headers=headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertGreaterEqual(len(sent),2)
        keys = ("model", "think", "options", "keep_alive", "stream")
        for item in sent:
            self.assertEqual(item["endpoint"], "http://127.0.0.1:11434/api/chat")
            self.assertEqual({k:item["payload"][k] for k in keys},
                             {k:sent[0]["payload"][k] for k in keys})
            self.assertIs(item["payload"]["think"], False)


if __name__ == "__main__":
    unittest.main()
