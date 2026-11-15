"""Pure CPU admission tests on synthetic scalar evidence, never sealed prompts."""
import unittest
from scripts.neral_pr542_release_admission_cpu import assess_release_evidence

RUN="43709097701"
SOURCE="737413f03b347676108d507ff3fe7693975dd269"
SHA="74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
def strict_receipt():
    return {"github_actions_run_id":RUN,"source_commit":SOURCE,
            "existing_sealed_sha256":SHA,"case_count":100,
            "baseline_correct":2,"candidate_correct":12,
            "net_gain":10,"regressions":0,"release_gate_passed":True}
def corrected_receipt():
    cats={}
    for c in ("mathematics","logic","programming","mixed"):
        cats[c]={"qwen":10,"neral":13,"regressions":0,"count":25}
    return {"status":"offline_cpu_regrade_complete","source_run":RUN,
            "source_commit":SOURCE,"sealed_sha256":SHA,"case_count":100,
            "categories":cats,"qwen_correct":40,"neral_correct":52,
            "net_gain":12,"regression_count":0,"regression_ids":[],
            "qwen_ambiguous_ids":[],"neral_ambiguous_ids":[],
            "inference_calls":0,"gpu_calls":0,"raw_outputs_uploaded":False}

class Admission(unittest.TestCase):
    def test_original_green_gate_is_blocked_without_raw_review(self):
        result=assess_release_evidence(strict_receipt(),None)
        self.assertEqual(result["admission"],"blocked")
        self.assertIn("raw_evidence_missing",result["reasons"])
        self.assertFalse(result["automatic_merge_allowed"])

    def test_uncertain_nonterminal_answer_blocks_zero_regression_claim(self):
        secondary=corrected_receipt()
        secondary["qwen_ambiguous_ids"]=["blind100-logic-03"]
        result=assess_release_evidence(strict_receipt(),secondary)
        self.assertEqual(result["admission"],"blocked")
        self.assertIn("unresolved_semantic_ambiguity",result["reasons"])

    def test_corrected_regression_blocks_even_when_old_gate_green(self):
        secondary=corrected_receipt()
        secondary["categories"]["logic"]["regressions"]=1
        secondary["regression_count"]=1
        secondary["regression_ids"]=["blind100-logic-04"]
        result=assess_release_evidence(strict_receipt(),secondary)
        self.assertEqual(result["admission"],"blocked")
        self.assertIn("corrected_regressions",result["reasons"])

    def test_verified_secondary_gate_only_releases_to_manual_review(self):
        result=assess_release_evidence(strict_receipt(),corrected_receipt())
        self.assertEqual(result["admission"],"manual_review_only")
        self.assertFalse(result["automatic_merge_allowed"])

    def test_different_source_hash_rejected(self):
        secondary=corrected_receipt()
        secondary["sealed_sha256"]="0"*64
        result=assess_release_evidence(strict_receipt(),secondary)
        self.assertEqual(result["admission"],"blocked")
        self.assertIn("provenance_mismatch",result["reasons"])

    def test_category_counts_and_total_must_balance(self):
        secondary=corrected_receipt()
        secondary["categories"]["mixed"]["count"]=24
        result=assess_release_evidence(strict_receipt(),secondary)
        self.assertEqual(result["admission"],"blocked")
        self.assertIn("invalid_category_accounting",result["reasons"])

if __name__=="__main__":
    unittest.main()
