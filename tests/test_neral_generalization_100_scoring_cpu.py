"""CPU-only evidence that the paired gate cannot forge a performance win."""
from __future__ import annotations
import unittest
from unittest.mock import patch
from scripts.neral_generalization_100_cases import build
from scripts.neral_generalization_100_run import (
    BASE_DIGEST, BASE_TAG, TUNED_TAG, TUNED_DIGEST,
    paired_score, parse_answer, guarded_case, require_exact_models,
)

class TransferHundredScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=build()

    def test_results_exactly_100_and_frozen_25_each(self):
        controls=[{"id":c["id"],"category":c["category"],
                   "arm":"raw_control","correct":i%3==0,
                   "integer_output":True,"status":"completed","seconds":1.5}
                  for i,c in enumerate(self.cases)]
        guarded=[{"id":c["id"],"category":c["category"],
                  "correct":i%3==0 or i==1,"integer_output":True,
                  "status":"completed","seconds":2.5,
                  "selection_policy":"raw_qwen_due_to_unverified_tir",
                  "verified":False}
                 for i,c in enumerate(self.cases)]
        report=paired_score(self.cases,controls,guarded)
        self.assertEqual(report["total"],100)
        self.assertEqual(report["gained"],1)
        self.assertEqual(report["lost"],0)
        self.assertEqual(report["net"],1)
        self.assertEqual(report["exact_mcnemar_two_sided_p"],1)
        self.assertEqual(report["base_correct"],34)
        self.assertEqual(report["guarded_correct"],35)
        self.assertEqual(set(report["by_category"]),{
            "mathematics","logic","programming","mixed"})
        self.assertEqual(len(report["cases"]),100)

    def test_edited_ids_rejected(self):
        a=[{"id":c["id"],"category":c["category"],"correct":True,
            "integer_output":True,"status":"completed","seconds":1}
           for c in self.cases]
        a[1]={**a[1],"id":a[0]["id"]}
        with self.assertRaises(ValueError):
            paired_score(self.cases,a,a)

    def test_exact_integer_outputs_not_scored_from_explanations(self):
        for text in ("Result is 42.","There are 42 assignments.","42 apples",
                     "FINAL_ANSWER: 42\nmore words","1 and 2",""):
            self.assertIsNone(parse_answer(text))
        self.assertEqual(parse_answer("  -42\n"),-42)
        self.assertEqual(parse_answer("FINAL_ANSWER: 42"),42)

    def test_two_digest_must_match_immutable_control_and_trained(self):
        with patch("scripts.neral_generalization_100_run.request",return_value={
            "models":[{"name":BASE_TAG,"digest":BASE_DIGEST},
                      {"name":TUNED_TAG,"digest":TUNED_DIGEST}]}):
            self.assertEqual(len(require_exact_models()),2)
        with patch("scripts.neral_generalization_100_run.request",return_value={
            "models":[{"name":BASE_TAG,"digest":BASE_DIGEST},
                      {"name":TUNED_TAG,"digest":"0"*64}]}):
            with self.assertRaises(RuntimeError):
                require_exact_models()

    def test_guard_rejects_unverified_claim_and_uses_real_raw_adapter(self):
        item=self.cases[0]
        forged={"text":"100","mode":"HIGH","verified":False,
           "model_generated":True,"calls_used":1,"seconds":0.01,
           "tool_calls_succeeded":0,"verification_status":"model_extracted_schema_unverified"}
        with patch("scripts.neral_generalization_100_run.run_experimental_tir",
                   return_value=forged),patch(
                   "scripts.neral_generalization_100_run.raw_chat",
                   return_value=str(item["expected_integer"])):
            result=guarded_case(item,24)
        self.assertTrue(result["correct"])
        self.assertEqual(result["selection_policy"],"raw_qwen_due_to_unverified_tir")
        self.assertFalse(result["verified"])

if __name__=="__main__":
    unittest.main()
