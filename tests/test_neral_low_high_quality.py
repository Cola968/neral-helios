"""RED: HIGH must beat LOW on scored paired tasks, not just use more calls."""
from __future__ import annotations
import unittest
from scripts.neral_low_high_quality import score_paired, make_prompt, QualityGateError
from scripts.neral_v7_foundation_tournament import generate_cases


DIGEST = "sha256:" + "a" * 64


def result(case, low, high, *, tool=None):
    def part(mode, text, tool_used=None):
        return {
            "text": text, "complete": True, "mode": mode, "model": "qwen3:8b",
            "model_origin": "local_foundation_not_neral_adapter",
            "adapter_loaded": False, "seconds": 1.0,
            "tool_used": tool_used,
        }
    return {"id": case["id"], "low": part("LOW", low),
            "high": part("HIGH", high, tool)}


def benchmark(improved=8, regressed=0, tool=False):
    cases = generate_cases()
    rows = []
    for i, case in enumerate(cases):
        good = "FINAL_ANSWER: " + str(case["answer"])
        bad = "FINAL_ANSWER: " + str(case["answer"] + 101)
        low, high = good, good
        if i < improved:
            low = bad
        if improved <= i < improved + regressed:
            high = bad
        rows.append(result(case, low, high,
                           tool="neral_math_exact" if tool and i < improved else None))
    return cases, rows


class HighMustBeBetter(unittest.TestCase):
    def test_measured_gain_can_pass_development_gate_without_release_claim(self):
        cases, pairs = benchmark(improved=8)
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertTrue(report["development_gate_passed"])
        self.assertEqual(report["low_correct"], 64)
        self.assertEqual(report["high_correct"], 72)
        self.assertEqual(report["paired"]["wins"], 8)
        self.assertEqual(report["paired"]["losses"], 0)
        self.assertGreater(report["difference_pp"], 5)
        self.assertFalse(report["general_reasoning_superiority_proven"])
        self.assertFalse(report["release_validated"])
        self.assertNotIn("prompt", str(report))
        self.assertNotIn("text", report)

    def test_equal_results_are_not_a_success(self):
        cases, pairs = benchmark(improved=0)
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertFalse(report["development_gate_passed"])
        self.assertEqual(report["difference_pp"], 0)

    def test_regressions_fail_even_when_high_has_some_wins(self):
        cases, pairs = benchmark(improved=8, regressed=8)
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertFalse(report["development_gate_passed"])
        self.assertEqual(report["paired"]["losses"], 8)

    def test_tool_assisted_gains_are_disclosed_not_called_neural_gain(self):
        cases, pairs = benchmark(improved=8, tool=True)
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertEqual(report["high_tool_assisted"], 8)
        self.assertFalse(report["neural_improvement_proven"])

    def test_not_enough_cases_is_not_proof(self):
        cases, pairs = benchmark(improved=8)
        report = score_paired(cases[:12], pairs[:12], model_digest=DIGEST)
        self.assertFalse(report["development_gate_passed"])
        self.assertEqual(report["case_count"], 12)

    def test_missing_or_duplicate_cases_fail_closed(self):
        cases, pairs = benchmark()
        with self.assertRaises(QualityGateError):
            score_paired(cases, pairs[:-1], model_digest=DIGEST)
        with self.assertRaises(QualityGateError):
            score_paired(cases, [pairs[0], *pairs[1:-1], pairs[0]], model_digest=DIGEST)

    def test_fake_backend_origin_or_wrong_mode_is_rejected(self):
        cases, pairs = benchmark()
        pairs[0]["high"]["adapter_loaded"] = True
        with self.assertRaises(QualityGateError):
            score_paired(cases, pairs, model_digest=DIGEST)
        cases, pairs = benchmark()
        pairs[0]["low"]["mode"] = "HIGH"
        with self.assertRaises(QualityGateError):
            score_paired(cases, pairs, model_digest=DIGEST)

    def test_identical_question_prompt_for_both_modes(self):
        case = generate_cases()[0]
        prompt = make_prompt(case)
        self.assertIn(case["prompt"], prompt)
        self.assertIn("FINAL_ANSWER", prompt)
        self.assertEqual(prompt, make_prompt(case))


if __name__ == "__main__":
    unittest.main()
