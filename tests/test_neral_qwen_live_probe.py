"""Offline contracts for the strictly opt-in real Qwen/Flask probe.

No live Ollama, GPU, model downloads or user data in this test module.
"""
import unittest

from neral_qwen_live_probe import verify_installed_model, assess, CASES


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


if __name__ == "__main__":
    unittest.main()
