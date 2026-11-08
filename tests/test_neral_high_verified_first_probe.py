"""Contracts for a paired real-HTTP LOW/HIGH *system* comparison."""
from __future__ import annotations

import unittest
from neral_high_verified_first import direct_high_reply
from scripts.neral_high_verified_first_probe import (
    SEED, generate_fresh_system_cases, score_pairs, ScoreError,
)


class FreshVerifierSystemCases(unittest.TestCase):
    def test_24_reproducible_new_parameters_independent_reference_values(self):
        a = generate_fresh_system_cases(SEED)
        b = generate_fresh_system_cases(SEED)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertEqual(len({row["id"] for row in a}), 24)
        self.assertEqual({d: sum(row["domain"] == d for row in a)
                          for d in ("math", "logic", "python_trace")},
                         {"math": 8, "logic": 8, "python_trace": 8})
        for row in a:
            exact = direct_high_reply([
                {"role": "system", "content": "NERAL local Qwen3."},
                {"role": "user", "content": row["prompt"]},
            ])
            self.assertIsNotNone(exact, row["id"])
            self.assertFalse(exact["model_generated"])
            self.assertTrue(exact["verified"])
            self.assertEqual(exact["calls_used"], 0)
            self.assertEqual(int(exact["final_answer"]), row["answer"])

    def test_positive_paired_system_gain_does_not_claim_weights_changed(self):
        cases = generate_fresh_system_cases(SEED)
        pairs = []
        for index, case in enumerate(cases):
            low_answer = case["answer"] if index % 3 == 0 else case["answer"] + 77
            pairs.append({
                "id": case["id"],
                "low": {"text": str(low_answer), "complete": True, "seconds": 2.0},
                "high": {"text": str(case["answer"]), "complete": True,
                         "seconds": .01, "verified": True,
                         "model_generated": False, "tool_used": "neral_math_exact"},
            })
        report = score_pairs(cases, pairs, model_digest="sha256:" + "a"*64)
        self.assertEqual(report["low_correct"], 8)
        self.assertEqual(report["high_correct"], 24)
        self.assertGreater(report["system_gain_pp"], 60)
        self.assertTrue(report["development_gate_passed"])
        self.assertFalse(report["neural_improvement_proven"])
        self.assertFalse(report["release_validated"])
        self.assertNotIn("prompt", str(report))
        self.assertNotIn("answer", str(report))

    def test_missing_pair_and_fake_high_verifier_are_rejected(self):
        cases = generate_fresh_system_cases(SEED)
        pairs = [{
            "id": row["id"],
            "low": {"text": str(row["answer"]), "complete": True, "seconds": 1.0},
            "high": {"text": str(row["answer"]), "complete": True, "seconds": .01,
                     "verified": True, "model_generated": False,
                     "tool_used": "neral_math_exact"},
        } for row in cases]
        with self.assertRaises(ScoreError):
            score_pairs(cases, pairs[:-1], model_digest="sha256:" + "a"*64)
        pairs[0]["high"]["model_generated"] = True
        with self.assertRaises(ScoreError):
            score_pairs(cases, pairs, model_digest="sha256:" + "a"*64)


if __name__ == "__main__":
    unittest.main()
