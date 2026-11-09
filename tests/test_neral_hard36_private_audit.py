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
        serialized = str(result)
        self.assertNotIn("secret prompt",serialized)
        self.assertNotIn("Thanks",serialized)
        self.assertFalse(result["raw_answers_uploaded"])

if __name__=="__main__":
    unittest.main()
