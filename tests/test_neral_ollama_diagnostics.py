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

    def test_high_only_short_final_recovery_after_length(self):
        sent = []
        outputs = iter([
            {"done": True, "done_reason": "length",
             "message": {"content": "Private partial reasoning: 6+7 equals 13."}},
            {"done": True, "done_reason": "stop",
             "message": {"content": "FINAL_ANSWER: 13"}},
        ])
        class Local:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                return _LocalResponse(next(outputs))
        turns = [
            {"role": "system", "content": "NERAL"},
            {"role": "user", "content": "Calculate 6+7. End with FINAL_ANSWER: <integer>."},
        ]
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            answer = self.backend.respond_high_final(turns)
        self.assertEqual(answer, "FINAL_ANSWER: 13")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["options"]["num_predict"], 2048)
        self.assertEqual(sent[1]["options"]["num_predict"], 64)
        self.assertEqual(sent[1]["options"]["temperature"], 0)
        self.assertEqual(sent[1]["think"], False)
        self.assertEqual(sent[1]["messages"][-2]["role"], "assistant")
        self.assertEqual(sent[1]["messages"][-1]["role"], "user")
        self.assertTrue(self.backend.last_inference_diagnostic["final_recovery_succeeded"])
        self.assertNotIn("Private partial reasoning", str(self.backend.inference_diagnostic_events))

    def test_raw_qwen_and_unrequested_final_never_trigger_rescue(self):
        sent = []
        class Local:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                return _LocalResponse({
                    "done": True, "done_reason": "length",
                    "message": {"content": "The reasoning is not complete."}})
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            with self.assertRaises(OllamaCompletionError) as raised:
                self.backend.respond_chat(self.messages)
        self.assertEqual(raised.exception.reason, "token_limit")
        self.assertEqual(len(sent), 1)
        sent.clear()
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            with self.assertRaises(OllamaCompletionError):
                self.backend.respond_high_final(self.messages)
        self.assertEqual(len(sent), 1)

    def test_short_rescue_rejects_noncanonical_output(self):
        sent = []
        outputs = iter([
            {"done": True, "done_reason": "length",
             "message": {"content": "The solution is not clear."}},
            {"done": True, "done_reason": "stop",
             "message": {"content": "FINAL_ANSWER: 14 and maybe something else"}},
        ])
        class Local:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                return _LocalResponse(next(outputs))
        messages = [{"role": "system", "content": "NERAL"},
                    {"role": "user", "content": "Give FINAL_ANSWER: <integer>."}]
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            with self.assertRaises(OllamaCompletionError) as raised:
                self.backend.respond_high_final(messages)
        self.assertEqual(raised.exception.reason, "token_limit")
        self.assertEqual(len(sent), 2)

    def test_base_and_high_initial_wire_payloads_equal_for_non_oracle(self):
        from scripts.neral_same_base_live import BASE_URL, _high_client
        from neral_identity import build_identity_prompt
        captured = []
        class Local:
            def open(self, request, timeout):
                captured.append(json.loads(request.data))
                return _LocalResponse({
                    "done": True, "done_reason": "stop",
                    "message": {"content": "FINAL_ANSWER: 19"}})
        prompt = "How many distinct arrangements? Give FINAL_ANSWER: <integer>."
        system = build_identity_prompt("web","local_foundation_not_neral_adapter")
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            self.backend.respond_chat([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}])
            client, headers = _high_client(self.backend)
            response = client.post("/api/neral/chat/think",
                json={"prompt":prompt}, base_url=BASE_URL, headers=headers)
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0], captured[1])
        self.assertNotIn("seed", captured[0]["options"])

    def test_pair_seed_is_opt_in_and_identical_on_raw_and_high_first_turn(self):
        from scripts.neral_same_base_live import BASE_URL, _high_client
        from neral_identity import build_identity_prompt
        sent = []
        class Local:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                return _LocalResponse({
                    "done": True, "done_reason": "stop",
                    "message": {"content": "FINAL_ANSWER: 19"}})
        prompt = "Count combinations exactly. Give FINAL_ANSWER: <integer>."
        system = build_identity_prompt("web", "local_foundation_not_neral_adapter")
        self.backend.set_benchmark_seed(142)
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            self.backend.respond_chat([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}])
            client, headers = _high_client(self.backend)
            result = client.post("/api/neral/chat/think", json={"prompt":prompt},
                                 headers=headers, base_url=BASE_URL)
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(sent[0]["options"]["seed"], 142)
        self.assertEqual(sent[0]["options"]["temperature"], 0.7)
        self.backend.set_benchmark_seed(None)
        sent.clear()
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            self.backend.respond_chat(self.messages)
        self.assertNotIn("seed", sent[0]["options"])

    def test_repair_turn_gets_1536_tokens_without_changing_first_extract_or_raw(self):
        """RED: only the bounded second schema attempt may use a larger budget."""
        seen = []
        class Local:
            def open(self, request, timeout):
                seen.append(json.loads(request.data))
                return _LocalResponse({
                    "done": True, "done_reason": "stop",
                    "message": {"content": '{"kind":"none"}', "thinking": ""},
                })
        turns = [
            {"role": "system", "content": "You are a strict JSON extractor."},
            {"role": "user", "content": (
                "How many Boolean assignments to A,B have A XOR B true? "
                "Return one integer.")},
        ]
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            self.backend.respond_formal_extract(turns)
            self.backend.respond_formal_extract(turns, repair_turn=True)
            self.backend.respond_chat(self.messages)
        self.assertEqual([x["options"]["num_predict"] for x in seen],
                         [384, 1536, 2048])
        self.assertTrue(all(x["think"] is False for x in seen))
        self.assertEqual(seen[0]["format"], seen[1]["format"])
        self.assertNotIn("format", seen[2])
        self.assertEqual(seen[1]["options"]["temperature"], 0)
        with self.assertRaises((ValueError, AgentModelError)):
            self.backend.respond_formal_extract(turns, repair_turn="true")

    def test_benchmark_seed_is_type_and_range_guarded(self):
        for invalid in (True, -1, 2**31, "42", 1.5):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                self.backend.set_benchmark_seed(invalid)
        legacy = OllamaLocalBackend("qwen3:8b")
        with self.assertRaises(ValueError):
            legacy.set_benchmark_seed(42)

    def test_direct_qwen_and_tir_http_route_send_identical_model_options(self):
        """Turn-0 is intentionally new; RAW fallback still has identical wire options."""
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
        # Direct raw -> 384-token extractor -> exact raw Qwen fallback.
        self.assertEqual(len(sent),3)
        keys = ("model", "think", "options", "keep_alive", "stream")
        raw, extract, fallback = (item["payload"] for item in sent)
        self.assertEqual({k:raw[k] for k in keys},
                         {k:fallback[k] for k in keys})
        self.assertEqual(raw["options"]["num_predict"], 2048)
        self.assertNotIn("format",raw)
        self.assertNotIn("format",fallback)
        self.assertIsInstance(extract["format"], dict)
        self.assertIn("anyOf", extract["format"])
        self.assertEqual(extract["options"]["num_predict"],384)
        self.assertEqual(extract["options"]["temperature"],0)
        self.assertFalse(extract["think"])
        self.assertEqual(response.json["verification_status"],
                         "base_fallback_unverified")
        self.assertEqual(response.json["first_turn_telemetry"]["decision"],
                         "extraction_fallback")
        for item in sent:
            self.assertEqual(item["endpoint"], "http://127.0.0.1:11434/api/chat")
            self.assertFalse(item["payload"]["think"])


if __name__ == "__main__":
    unittest.main()
