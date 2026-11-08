"""Regression contracts for the read-only NERAL FAST/THINK diagnostic."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.neral_fast_think_ab import pick_cases, compare_modes

DIGEST = 'sha256:' + 'a' * 64


class FakeClient:
    def __init__(self, *, change_digest=False):
        self.calls = []
        self.change_digest = change_digest
        self.inventories = 0

    def inventory(self):
        self.inventories += 1
        return {'qwen3:8b': ('sha256:' + 'b' * 64) if
                self.change_digest and self.inventories > 1 else DIGEST}

    def chat(self, model, prompt, *, think, max_tokens):
        self.calls.append((model, prompt, think, max_tokens))
        return {'content': 'FINAL_ANSWER: 12345678', 'reason': 'stop',
                'eval_count': 10, 'duration_ns': 1000000000}


class RecommendedSamplingTests(unittest.TestCase):
    def test_qwen3_uses_non_greedy_documented_sampling(self):
        from scripts.neral_fast_think_ab import RecommendedOllamaLocal

        class Stub(RecommendedOllamaLocal):
            def _request(self, path, payload=None):
                self.last_request = payload
                return {"message": {"content": "FINAL_ANSWER: 7"},
                        "done_reason": "stop", "eval_count": 8,
                        "total_duration": 3000000}

        backend = Stub()
        for think, temperature, top_p in (
            (False, 0.7, 0.8), (True, 0.6, 0.95),
        ):
            backend.chat("qwen3:8b", "Berechne 3+4.", think=think,
                         max_tokens=2048)
            options = backend.last_request["options"]
            self.assertEqual(backend.last_request["think"], think)
            self.assertEqual(options["temperature"], temperature)
            self.assertEqual(options["top_p"], top_p)
            self.assertEqual(options["top_k"], 20)
            self.assertEqual(options["num_predict"], 2048)
            self.assertEqual(options["seed"], 20261108)

    def test_non_qwen3_8b_is_not_mislabeled_recommended(self):
        from scripts.neral_fast_think_ab import RecommendedOllamaLocal
        with self.assertRaises(ValueError):
            RecommendedOllamaLocal().chat("qwen3.5:9b", "test",
                                         think=True, max_tokens=2048)


class ABTests(unittest.TestCase):
    def test_balanced_reproducible_cases(self):
        a = pick_cases(12)
        b = pick_cases(12)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertEqual({d: sum(c['domain'] == d for c in a)
                          for d in ('math', 'reasoning', 'code_trace')},
                         {'math': 4, 'reasoning': 4, 'code_trace': 4})
        self.assertEqual(len({c['family'] for c in a}), 12)

    def test_runs_both_modes_same_budget_and_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'result'
            client = FakeClient()
            report = compare_modes(client, 'qwen3:8b', folder, cases_count=6,
                                   max_tokens=1024)
            self.assertEqual(len(client.calls), 12)
            for i in range(6):
                first, second = client.calls[2 * i:2 * i + 2]
                self.assertEqual(first[0:2], second[0:2])
                self.assertEqual(first[3], second[3])
                self.assertEqual({first[2], second[2]}, {True, False})
            self.assertTrue(folder.joinpath('private-responses.jsonl').exists())
            self.assertTrue(folder.joinpath('scalar-summary.json').exists())
            self.assertEqual(report['n_per_mode'], 6)
            self.assertFalse(report['model_superiority_proven'])
            self.assertFalse(report['tools_compared'])

    def test_model_drift_blocks_success_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'result'
            with self.assertRaises(ValueError):
                compare_modes(FakeClient(change_digest=True), 'qwen3:8b',
                              folder, cases_count=6, max_tokens=1024)
            self.assertFalse(folder.joinpath('scalar-summary.json').exists())

    def test_rejects_uninstalled_model_and_invalid_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                compare_modes(FakeClient(), 'qwen3:9b', Path(tmp) / 'missing')
            with self.assertRaises(ValueError):
                pick_cases(7)


if __name__ == '__main__':
    unittest.main()
