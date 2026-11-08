"""Fail-closed contracts for NERAL's evidence-driven, non-promoting training loop.

Synthetic model answers prove the decision machinery, NOT model intelligence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from scripts.neral_v8_unseen_suite import all_cases, pilot_cases
from scripts.neral_v8_unseen_eval import report_scores
from scripts.neral_v8_experiment_policy import evaluate_experiment, EvidenceError


def _responses(cases):
    return {
        name: [
            {"id":c["id"],"text":"FINAL_ANSWER: "+c["answer"],
             "complete":True,"stop_reason":"eos","new_tokens":16}
            for c in cases
        ]
        for name in ("qwen","v5","v7","v8")
    }


def _report(sample="pilot", *, transfer_gain=0, wrong_v8=None, length_censored=False):
    cases=pilot_cases() if sample=="pilot" else all_cases()
    answers=_responses(cases)
    new=[i for i,c in enumerate(cases) if c["part"]=="transfer"]
    for i in new[:transfer_gain]:
        for old in ("qwen","v5","v7"):
            answers[old][i]["text"]="FINAL_ANSWER: -123456"
    if wrong_v8 is not None:
        answers["v8"][wrong_v8]["text"]="FINAL_ANSWER: -123456"
    if length_censored:
        i=0
        answers["v8"][i].update(
            text="FINAL_ANSWER: "+cases[i]["answer"],
            new_tokens=512,complete=False,stop_reason="length",
        )
    return report_scores(cases,answers,sample),cases


class ExperimentEvidenceContracts(unittest.TestCase):

    def test_no_transfer_gain_is_rejected_even_when_loss_could_be_lower(self):
        report,_=_report("full")
        result=evaluate_experiment(report)
        self.assertEqual(result["decision"],"REJECT")
        self.assertIn("NO_TRANSFER_GAIN",result["failure_codes"])
        self.assertFalse(result["release_allowed"])
        self.assertFalse(result["adapter_promoted"])
        self.assertTrue(result["baseline_preserved"])
        self.assertTrue(result["no_training_authorized"])
        self.assertTrue(result["holdout_cases_must_not_enter_training"])

    def test_a_single_new_error_blocks_candidate_against_all_comparators(self):
        report,cases=_report("full",transfer_gain=20,wrong_v8=0)
        result=evaluate_experiment(report)
        self.assertEqual(result["decision"],"REJECT")
        self.assertIn("RETENTION_REGRESSION",result["failure_codes"])
        self.assertIn("REGRESSION_VS_QWEN",result["failure_codes"])
        self.assertIn("REGRESSION_VS_V5",result["failure_codes"])
        self.assertIn("REGRESSION_VS_V7",result["failure_codes"])
        self.assertNotIn(cases[0]["id"],json.dumps(result))
        self.assertNotIn(cases[0]["prompt"],json.dumps(result))

    def test_32_task_pilot_cannot_approve_even_perfect_candidate(self):
        report,_=_report("pilot",transfer_gain=16)
        result=evaluate_experiment(report)
        self.assertEqual(result["decision"],"REJECT")
        self.assertIn("PILOT_ONLY",result["failure_codes"])
        self.assertFalse(result["release_allowed"])

    def test_full_clean_gain_only_advances_to_independent_validation(self):
        report,_=_report("full",transfer_gain=20)
        self.assertTrue(report["gate"]["eligible_for_review"])
        result=evaluate_experiment(report)
        self.assertEqual(result["decision"],"INDEPENDENT_REVIEW_REQUIRED")
        self.assertEqual(result["failure_codes"],[])
        self.assertTrue(result["requires_new_blind_holdout"])
        self.assertFalse(result["independent_ood_proven"])
        self.assertFalse(result["release_allowed"])
        self.assertFalse(result["adapter_promoted"])
        self.assertEqual(result["next_experiment"],"BLIND_HOLDOUT_WITH_FROZEN_WEIGHTS")

    def test_length_censored_answers_are_not_counted_as_success(self):
        report,_=_report("full",transfer_gain=20,length_censored=True)
        result=evaluate_experiment(report)
        self.assertEqual(result["decision"],"REJECT")
        self.assertIn("EOS_OR_DECODING_REGRESSION",result["failure_codes"])
        self.assertIn("RETENTION_REGRESSION",result["failure_codes"])
        self.assertEqual(result["next_experiment"],"FIX_COMPLETION_BUDGET_WITHOUT_TRAINING")

    def test_tampered_or_untrusted_evidence_fails_closed(self):
        report,_=_report("full",transfer_gain=20)
        for key,value in (
            ("model_only_no_tools",False),
            ("matched_prompts",False),
            ("greedy_deterministic",False),
            ("same_eos_policy",False),
            ("independent_ood_proven",True),
            ("weights_promoted",True),
            ("release_validated",True),
            ("max_new_tokens",256),
            ("case_count",31),
        ):
            with self.subTest(key=key):
                changed=deepcopy(report)
                changed[key]=value
                with self.assertRaises(EvidenceError):
                    evaluate_experiment(changed)
        for altered in (
            {"raw_private_answer":"sensitive"},
            {"models":{"v8":{"correct":999}}},
            {"paired":{"v8_minus_v5":{"regressions":-1}}},
        ):
            changed=deepcopy(report)
            if "models" in altered:
                changed["models"]["v8"]["correct"]=999
            elif "paired" in altered:
                changed["paired"]["v8_minus_v5"]["regressions"]=-1
            else:
                changed.update(altered)
            with self.assertRaises(EvidenceError):
                evaluate_experiment(changed)

    def test_policy_is_embedded_in_real_v8_scalar_reports_without_raw_prompts(self):
        report,cases=_report("full",transfer_gain=20)
        self.assertEqual(
            report["experiment_policy"]["decision"],
            "INDEPENDENT_REVIEW_REQUIRED",
        )
        text=json.dumps(report)
        self.assertNotIn(cases[0]["prompt"],text)
        self.assertNotIn(cases[0]["id"],text)
        self.assertFalse(report["experiment_policy"]["release_allowed"])
        self.assertFalse(report["weights_promoted"])


if __name__=="__main__":
    unittest.main()
