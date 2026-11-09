"""Fresh hard36 TIR development-case and scalar accounting contracts."""
import unittest
from collections import Counter

from scripts.neral_tir_hard_live import generate_hard_cases, validated_tool_counts


class HardPilotContracts(unittest.TestCase):
    def test_six_harder_families_and_balanced_domains(self):
        cases = generate_hard_cases(20261108)
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({row["id"] for row in cases}), 36)
        self.assertEqual(dict(Counter(c["domain"] for c in cases)),
                         {"math": 12, "reasoning": 12, "code_trace": 12})
        self.assertEqual(len({(c["domain"], c["family"]) for c in cases}), 6)
        self.assertEqual(len({c["prompt"] for c in cases}), 36)
        self.assertTrue(all(type(c["answer"]) is int and c["kind"] == "integer"
                            for c in cases))
        self.assertTrue(any("UTC" in c["prompt"] for c in cases))
        self.assertTrue(any("off-by-one" in c["prompt"] for c in cases))

    def test_fresh_instances_not_same_prompt_for_other_seed(self):
        a = generate_hard_cases(20261108)
        b = generate_hard_cases(20261109)
        self.assertNotEqual([r["prompt"] for r in a],
                            [r["prompt"] for r in b])

    def test_counters_reject_faked_or_inconsistent_success(self):
        self.assertEqual(validated_tool_counts({"tool_calls_triggered": 2,
                                                "tool_calls_succeeded": 1}),
                         (2, 1))
        for item in ({"tool_calls_triggered": 1, "tool_calls_succeeded": 2},
                     {"tool_calls_triggered": True, "tool_calls_succeeded": 1},
                     {"tool_calls_triggered": -1, "tool_calls_succeeded": 0},
                     {"tool_calls_triggered": 3, "tool_calls_succeeded": 0}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                validated_tool_counts(item)

    def test_transient_503_retried_serially_with_backoff(self):
        from scripts.neral_tir_hard_live import run_one_with_backoff
        class Response:
            def __init__(self, status, body):
                self.status_code, self.body = status, body
            def get_json(self, silent=True):
                return self.body
        class Client:
            def __init__(self):
                self.calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                if self.calls < 3:
                    return Response(503, {"error_code": "timeout"})
                return Response(200, {
                    "model": "qwen3.5:9b", "inference_profile": "measured_nonthinking_2048",
                    "model_origin": "local_foundation_not_neral_adapter",
                    "adapter_loaded": False, "mode": "HIGH", "text": "FINAL_ANSWER: 17",
                    "tool_used": None, "model_generated": True})
        client = Client()
        pauses = []
        result = run_one_with_backoff(
            "neral_high", "Answer exactly 17", None, client, {},
            sleeper=pauses.append)
        self.assertEqual(client.calls, 3)
        self.assertEqual(result[0], "FINAL_ANSWER: 17")
        self.assertFalse(result[2])
        self.assertEqual(len(pauses), 2)
        self.assertGreater(pauses[1], pauses[0])

    def test_terminal_token_limit_is_not_silently_retried(self):
        from scripts.neral_tir_hard_live import run_one_with_backoff
        class Response:
            status_code = 503
            def get_json(self, silent=True):
                return {"error_code": "token_limit"}
        class Client:
            calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                return Response()
        client = Client()
        result = run_one_with_backoff(
            "neral_high", "difficult task", None, client, {},
            sleeper=lambda *_: None)
        self.assertEqual(client.calls, 1)
        self.assertEqual(result[-1], "token_limit")
        self.assertTrue(result[2])


if __name__ == "__main__":
    unittest.main()
