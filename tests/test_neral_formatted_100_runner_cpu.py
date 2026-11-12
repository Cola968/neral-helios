"""Regression gate contracts: every original case, unchanged hash, no scoring drift."""
import unittest
from scripts.neral_formatted_100_gpu import (
    CHECKPOINT_SHA256, summarize_repeat, verify_original_100,
)
from scripts.neral_blind_100_cases import build,serialize
from scripts.neral_blind_100_gpu import score_pairs

class FormattedRepeatScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=build()

    def test_exact_original_100_and_hash_pinning(self):
        self.assertEqual(CHECKPOINT_SHA256,"74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258")
        self.assertEqual(len(self.cases),100)
        self.assertEqual(verify_original_100(self.cases),CHECKPOINT_SHA256)
        cases=build()
        cases[0]={**cases[0],"question":cases[0]["question"]+" "}
        with self.assertRaises(ValueError):
            verify_original_100(cases)

    def test_format_and_token_limit_counted_per_category(self):
        rows=[]
        for i,c in enumerate(self.cases):
            row={"id":c["id"],"category":c["category"],
                 "base_correct":False,"dual_correct":i%2==0,
                 "base_completion_status":"completed",
                 "dual_completion_status":"completed",
                 "base_integer_format":False,"dual_integer_format":True,
                 "base_done_reason":"stop","dual_done_reason":"stop",
                 "base_seconds":2.0,"dual_seconds":10.0,
                 "base_load_ms":800,"dual_model_load_ms":[3500],
                 "both_resident":False}
            if i==0:
                row["base_completion_status"]="token_limit"
                row["base_done_reason"]="length"
                row["dual_completion_status"]="token_limit"
                row["dual_done_reason"]="length"
                row["dual_correct"]=False
                row["dual_integer_format"]=False
            rows.append(row)
        result=summarize_repeat(self.cases,rows)
        self.assertEqual(result["status"],"REPEAT_SAME_100_CHANGED_ANSWER_CONTRACT")
        self.assertEqual(result["primary_blind_test"],False)
        self.assertEqual(result["scoring_contract"],"only_terminal_FINAL_ANSWER_integer")
        self.assertEqual(result["token_limit"]["base"],1)
        self.assertEqual(result["token_limit"]["dual"],1)
        self.assertEqual(result["format_invalid"]["base"],99)
        self.assertEqual(result["format_invalid"]["dual"],0)
        self.assertEqual(result["by_category"]["mathematics"]["total"],25)
        self.assertEqual(result["gained"],49)
        self.assertEqual(result["lost"],0)
        self.assertEqual(result["base_correct"],0)
        self.assertEqual(result["dual_correct"],49)
        self.assertFalse(result["promotion_gate_eligible"])
        self.assertEqual(result["observed_both_models_resident_count"],0)
        with self.assertRaises(ValueError):
            summarize_repeat(self.cases,rows[:99])

if __name__=="__main__":
    unittest.main()
