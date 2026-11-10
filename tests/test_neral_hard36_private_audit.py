"""CPU-only contracts for read-only private reasoning failure audit."""
import unittest
from scripts.neral_hard36_private_audit import _numeric_candidate, summarize_previous_private
from scripts.neral_same_base_ab_score import ARMS

class PrivateReasoningAuditTests(unittest.TestCase):
    def test_diagnostic_parser_can_find_nonterminal_final_marker(self):
        self.assertEqual(_numeric_candidate("Work here\nFINAL_ANSWER: 7\nThanks"), 7)
        self.assertEqual(_numeric_candidate("Reasoning\n7"), 7)
        self.assertIsNone(_numeric_candidate("FINAL_ANSWER: 7\nFINAL_ANSWER: 8"))

    def test_only_aggregate_counts_emitted(self):
        cases = [{"id": f"case-{i}", "domain": "reasoning" if i < 12 else
                 "math" if i < 24 else "code_trace", "answer": 7,
                 "prompt": "secret prompt"} for i in range(36)]
        obs = []
        for case in cases:
            raw = {"text": "FINAL_ANSWER: 7\nThanks", "complete": True, "http_error": False}
            if case["id"] == "case-0":
                raw = {"text": "", "complete": False, "http_error": True}
            obs.append({"id":case["id"], "arms":{
                a:dict(raw if a=="raw_qwen35" else
                       {"text":"FINAL_ANSWER: 7", "complete":True,"http_error":False})
                for a in ARMS}})
        result = summarize_previous_private({"scorer_input":{
            "cases":cases, "observations":obs}})
        raw = result["arms"]["raw_qwen35"]["reasoning"]
        self.assertEqual(raw["failed_inference"], 1)
        self.assertEqual(raw["format_only_failure"], 11)
        self.assertEqual(result["arms"]["raw_qwen35"]["by_domain"]["math"]["outcomes"],
                         {"format_only_failure": 12})
        serialized = str(result)
        self.assertNotIn("secret prompt",serialized)
        self.assertNotIn("Thanks",serialized)
        self.assertFalse(result["raw_answers_uploaded"])

    def test_math_losses_partition_failure_format_wrong_and_ambiguous(self):
        cases = [{"id": str(i), "domain": ("math" if i < 12 else
                  "reasoning" if i < 24 else "code_trace"), "answer": 7}
                 for i in range(36)]
        variants = [
            {"text": "", "complete": False, "http_error": True},
            {"text": "FINAL_ANSWER: 7\nextra", "complete": True, "http_error": False},
            {"text": "8", "complete": True, "http_error": False},
            {"text": "7.0", "complete": True, "http_error": False},
        ]
        observations = []
        for i, case in enumerate(cases):
            arms = {a: {"text": "7", "complete": True, "http_error": False} for a in ARMS}
            if i < 4:
                arms["neral_high_router"] = variants[i]
            observations.append({"id": case["id"], "arms": arms})
        result = summarize_previous_private({"scorer_input": {
            "cases": cases, "observations": observations}})
        self.assertEqual(result["tir_losses_vs_raw_by_domain"]["math"], {
            "failed_inference": 1, "format_only_failure": 1,
            "numeric_answer_wrong": 1, "no_unambiguous_numeric_final": 1})
        self.assertEqual(result["tir_losses_vs_raw_by_domain"]["reasoning"], {})
        self.assertFalse(result["independent_holdout_verified"])
        self.assertEqual(result["source"], "supplied_private_hard36")

if __name__=="__main__":
    unittest.main()

