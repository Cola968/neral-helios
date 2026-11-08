"""Private-output regression triage: tests never rely on real user/model answers."""
from __future__ import annotations

import json
import unittest

from scripts.neral_v8_unseen_suite import pilot_cases
from scripts.neral_v8_private_failure_triage import triage_response_records
from scripts.neral_v8_unseen_eval import ARMS, report_scores


def _responses(cases):
    return {
        name:[
            {"id":c["id"],"text":"FINAL_ANSWER: "+c["answer"],
             "complete":True,"stop_reason":"eos","new_tokens":12}
            for c in cases
        ]
        for name in ARMS
    }


class PrivateFailureTriageTests(unittest.TestCase):
    def test_exact_agreement_and_private_scalar_only_report(self):
        cases=pilot_cases()
        answers=_responses(cases)
        answers["v8"][0]["text"]="FINAL_ANSWER: 999999"
        answers["v8"][1]["text"]="Unparseable final reply"
        answers["qwen"][0].update(
            text="Generating reasoning",complete=False,
            stop_reason="length",new_tokens=512)
        report=triage_response_records(cases,answers,"pilot")
        self.assertEqual(report["status"],"NERAL_V8_PRIVATE_FAILURE_TRIAGE_READONLY")
        self.assertEqual(report["models"]["v8"]["wrong_final"],1)
        self.assertEqual(report["models"]["v8"]["unreadable_complete"],1)
        self.assertEqual(report["models"]["qwen"]["length_censored"],1)
        self.assertEqual(report["disagreements"]["v8_new_regressions_vs_v5"],2)
        self.assertEqual(report["disagreements"]["all_models_wrong"],0)
        self.assertEqual(report["disagreements"]["solved_by_at_least_one"],32)
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["adapter_promoted"])
        printed=json.dumps(report)
        for c in cases:
            self.assertNotIn(c["id"],printed)
            self.assertNotIn(c["prompt"],printed)
        for rs in answers.values():
            for row in rs:
                self.assertNotIn("Generating reasoning",printed)
                self.assertNotIn("Unparseable final reply",printed)

    def test_score_is_unchanged_and_length_errors_not_regraded(self):
        cases=pilot_cases()
        answers=_responses(cases)
        answers["v8"][0].update(
            text="FINAL_ANSWER: "+cases[0]["answer"],
            complete=False,stop_reason="length",new_tokens=512)
        report=triage_response_records(cases,answers,"pilot")
        original=report_scores(cases,answers,"pilot")
        self.assertEqual(report["models"]["v8"]["correct"],original["models"]["v8"]["correct"])
        self.assertEqual(report["models"]["v8"]["length_censored"],1)
        self.assertEqual(report["models"]["v8"]["parseable_but_unfinished"],1)
        self.assertEqual(report["disagreements"]["v8_new_regressions_vs_v5"],1)
        self.assertFalse(report["can_claim_superiority"])

    def test_long_complete_answers_count_as_unreadable_without_rewriting_scoring(self):
        cases=pilot_cases()
        answers=_responses(cases)
        answers["v8"][0].update(text="Reasoning "+("a"*4200)+"\nFINAL_ANSWER: "+
                                cases[0]["answer"])
        result=triage_response_records(cases,answers,"pilot")
        self.assertEqual(result["models"]["v8"]["scoring_char_limit_exceeded"],1)
        self.assertEqual(result["models"]["v8"]["unreadable_complete"],1)

    def test_rejects_missing_or_forged_rows(self):
        cases=pilot_cases()
        answers=_responses(cases)
        answers["v8"][0]["id"]="PRIVATE-OTHER-ID"
        with self.assertRaises(ValueError):
            triage_response_records(cases,answers,"pilot")


if __name__=="__main__":
    unittest.main()
