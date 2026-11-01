"""Offline contracts for the strictly opt-in real Qwen/Flask probe.

No live Ollama, GPU, model downloads or user data in this test module.
"""
import unittest

from neral_qwen_live_probe import verify_installed_model, assess, CASES, classify_assistant_error, sanitized_protocol_event


class LiveProbeContracts(unittest.TestCase):
    def test_only_exact_local_qwen_tag_is_accepted(self):
        self.assertTrue(verify_installed_model({"models": [{"name": "qwen3.5:9b"}]}))
        self.assertFalse(verify_installed_model({"models": [{"name": "qwen3.5:4b"}]}))
        self.assertFalse(verify_installed_model({"models": []}))
        for malformed in ({}, None, {"models": "qwen3.5:9b"}, {"models": [None]}):
            with self.subTest(item=malformed), self.assertRaises(ValueError):
                verify_installed_model(malformed)

    def test_assessment_keeps_only_safe_metadata_no_raw_model_reply(self):
        secret_output = "Paris INTERNAL_SYNTHETIC_TEXT_NOT_FOR_LOGS"
        payload = {
            "mode": "neral-assistant", "source": "external-model",
            "base_model": "qwen3.5:9b", "model_generated": True,
            "text": secret_output, "tools_used": [],
            "incomplete": False, "finish_reason": "stop",
            "generated_tokens": 7, "tokens_per_second": 10.0
        }
        summary = assess(200, payload, CASES[0], 2.5)
        self.assertTrue(summary["valid_provider"])
        self.assertTrue(summary["answer_match"])
        self.assertNotIn(secret_output, repr(summary))
        self.assertNotIn("INTERNAL_SYNTHETIC_TEXT", repr(summary))
        self.assertEqual(summary["text_bytes"], len(secret_output.encode("utf8")))

    def test_truncated_or_fake_origin_never_counts_as_valid(self):
        base = {"mode": "neral-assistant", "source": "external-model",
                "base_model": "qwen3.5:9b", "model_generated": True,
                "text": "Paris", "tools_used": [], "incomplete": False,
                "finish_reason": "stop"}
        for updates in ({"incomplete": True}, {"source": "tool:solve"},
                        {"base_model": "arctus350"}, {"model_generated": False},
                        {"tools_used": ["shell"]}, {"tools_used": ["calculator", "calculator"]}):
            with self.subTest(updates=updates):
                result = assess(200, {**base, **updates}, CASES[0], 1.1)
                self.assertFalse(result["valid_provider"])

    def test_error_is_reported_without_private_body(self):
        result = assess(503, {"error": "private token or personal filename"}, CASES[0], 0.3)
        self.assertFalse(result["valid_provider"])
        self.assertEqual(result["http_status"], 503)
        self.assertNotIn("private token", repr(result))

    def test_cases_are_bounded_synthetic_and_deterministic(self):
        self.assertEqual([c["id"] for c in CASES], ["fact_de", "math_de", "code_de"])
        self.assertTrue(all(0 < c["max_new_tokens"] <= 128 for c in CASES))
        self.assertTrue(all(c["quality"] == "standard" for c in CASES))
        self.assertEqual(len({c["id"] for c in CASES}), len(CASES))


    def test_known_provider_exception_is_classified_without_echoing_arbitrary_text(self):
        self.assertEqual(classify_assistant_error("Ungültige Codeprüfparameter."), "invalid_code_check_arguments")
        self.assertEqual(classify_assistant_error("Codeprüfung überschreitet das sichere Ergebnislimit."), "code_check_result_limit")
        self.assertEqual(classify_assistant_error("Werkzeugaufrufe überschreiten das Limit."), "tool_budget")
        self.assertEqual(classify_assistant_error("SECRET prompt with private username"), "other_assistant_error")
        self.assertNotIn("SECRET", repr(classify_assistant_error("SECRET private data")))

    def test_raw_model_tool_arguments_are_never_logged(self):
        data = {"done_reason": "stop",
                "message": {"role": "assistant", "content": "SECRET_GENERATION_CONTENT",
                            "tool_calls": [{"function": {
                                "name": "tool-name-with-SECRET",
                                "arguments": {"code": "SECRET_USER_SOURCE_CODE"}}}]}}
        sanitized = sanitized_protocol_event(data)
        self.assertEqual(sanitized["tool_count"], 1)
        self.assertEqual(sanitized["tool_names"], ["other"])
        self.assertEqual(sanitized["argument_types"], ["dict"])
        self.assertNotIn("SECRET", repr(sanitized))
        self.assertNotIn("code", repr(sanitized))


if __name__ == "__main__":
    unittest.main()
