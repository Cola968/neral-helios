"""Synthetic evaluation integrity checks; NOT real model generations."""
import json
from pathlib import Path
import tempfile
import unittest

from phoenix_gate import compare, load


def fixture(checkpoint: str) -> dict:
    return {"schema": 1, "checkpoint_sha256": checkpoint * 64,
            "protocol_sha256": "a" * 64,
            "cases": [{"id": f"case-{i}", "prompt_sha256": f"{i:064x}",
                       "answer": "An engineering fixture answer.", "finish_reason": "eos",
                       "raw_model": True, "tool_calls": 0,
                       "tokens_generated": 5, "max_tokens": 64}
                      for i in range(8)]}


class UniquePopulationContracts(unittest.TestCase):
    def test_load_rejects_same_prompt_under_different_ids(self):
        payload = fixture("b")
        payload["cases"][1]["prompt_sha256"] = payload["cases"][0]["prompt_sha256"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "local_test_only.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeated evaluation prompt hash"):
                load(path)

    def test_direct_compare_rejects_repeated_candidate_prompt(self):
        baseline, candidate = fixture("b"), fixture("c")
        candidate["cases"][1]["prompt_sha256"] = candidate["cases"][0]["prompt_sha256"]
        with self.assertRaises(ValueError):
            compare(baseline, candidate)

    def test_direct_compare_rejects_identical_duplicates_in_both(self):
        baseline, candidate = fixture("b"), fixture("c")
        for report in (baseline, candidate):
            report["cases"][1]["prompt_sha256"] = report["cases"][0]["prompt_sha256"]
        with self.assertRaisesRegex(ValueError, "repeats prompt hashes"):
            compare(baseline, candidate)

    def test_unique_population_remains_valid_but_never_released(self):
        report = compare(fixture("b"), fixture("c"))
        self.assertTrue(report["health_nonregression"])
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["answer_correctness_evaluated"])


if __name__ == "__main__":
    unittest.main()
