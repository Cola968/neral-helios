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

    def test_high_final_and_final_answer_are_scored_equally(self):
        cases, pairs = benchmark(improved=0)
        for case, row in zip(cases, pairs):
            row["high"]["text"] = "FINAL: " + str(case["answer"])
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertEqual(report["high_correct"], 72)
        self.assertEqual(report["low_correct"], 72)
        self.assertEqual(report["difference_pp"], 0)

    def test_complete_long_reasoning_with_one_final_marker_scores_correct(self):
        # A 2048-token HIGH reply can exceed 4096 characters and still finish.
        cases, pairs = benchmark(improved=0)
        for case, row in zip(cases, pairs):
            row["high"]["text"] = (
                "Schrittweise Prüfung: " + ("Aussage korrekt geordnet. " * 230)
                + "\nFINAL: " + str(case["answer"])
            )
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertEqual(report["high_correct"], 72)
        self.assertEqual(report["low_correct"], 72)

    def test_long_answer_with_duplicated_final_is_rejected(self):
        cases, pairs = benchmark(improved=0)
        for case, row in zip(cases, pairs):
            row["high"]["text"] = (
                "FINAL: " + str(case["answer"]) + "\n"
                + ("Vollständige Herleitung. " * 210)
                + "\nFINAL: " + str(case["answer"])
            )
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertEqual(report["high_correct"], 0)

    def test_multiple_final_markers_are_not_accepted_as_correct(self):
        cases, pairs = benchmark(improved=0)
        for case, row in zip(cases, pairs):
            row["high"]["text"] = ("FINAL: " + str(case["answer"]) +
                                    "\\nFINAL_ANSWER: " + str(case["answer"]))
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertEqual(report["high_correct"], 0)
        self.assertEqual(report["low_correct"], 72)
        self.assertFalse(report["development_gate_passed"])

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

    def test_actual_quantitative_tool_tag_is_disclosed_not_rejected(self):
        # neral_agent_quantitative returns "neral_quant_exact" for exact
        # physical conversions; the LOW/HIGH scorer must keep provenance.
        cases, pairs = benchmark(improved=1)
        pairs[0]["high"]["tool_used"] = "neral_quant_exact"
        report = score_paired(cases, pairs, model_digest=DIGEST)
        self.assertEqual(report["high_tool_assisted"], 1)
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

    def test_local_http_probe_uses_same_prompt_and_keeps_only_aggregate(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.neral_low_high_quality import run_local_probe

        calls = []
        class FakeInventory:
            def inventory(self):
                return {"qwen3:8b": DIGEST}
        class FakeBackend:
            model = "qwen3:8b"
            def __init__(self, *_args, **_kwargs):
                pass
            def respond_chat(self, messages):
                calls.append(("LOW", messages[-1]["content"]))
                return "FINAL_ANSWER: 0"
            def respond_think(self, messages):
                calls.append(("HIGH", messages[-1]["content"]))
                return "FINAL_ANSWER: 0"

        with tempfile.TemporaryDirectory() as td, \
             patch("scripts.neral_low_high_quality.OllamaLocal", FakeInventory), \
             patch("neral_agent_autonomy.OllamaLocalBackend", FakeBackend):
            output = Path(td) / "aggregate.json"
            report = run_local_probe(count=12, output=output)
            self.assertEqual(report["case_count"], 12)
            self.assertFalse(report["development_gate_passed"])
            self.assertTrue(output.is_file())
            content = output.read_text(encoding="utf8")
            self.assertNotIn("FINAL_ANSWER", content)
            self.assertNotIn("Gib das Ergebnis", content)
            self.assertNotIn("prompt", content)
            self.assertEqual(json.loads(content)["case_count"], 12)
        self.assertEqual(calls[0][0], "LOW")
        self.assertEqual(calls[1][0], "HIGH")
        self.assertEqual(calls[0][1], calls[1][1])
        # Alternating order over cases: case 0 LOW→HIGH, case 1 HIGH→LOW.
        # HIGH no longer inserts an extra unverified second model call.
        self.assertEqual(calls[2][0], "HIGH")
        self.assertEqual(calls[3][0], "LOW")
        self.assertEqual(calls[2][1], calls[3][1])

    def test_identical_question_prompt_for_both_modes(self):
        case = generate_cases()[0]
        prompt = make_prompt(case)
        self.assertIn(case["prompt"], prompt)
        self.assertIn("FINAL_ANSWER", prompt)
        self.assertEqual(prompt, make_prompt(case))


if __name__ == "__main__":
    unittest.main()
