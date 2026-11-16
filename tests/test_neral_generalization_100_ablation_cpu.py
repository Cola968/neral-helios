"""Ablation: tested routing and paired-score contracts, CPU-only."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from scripts.neral_generalization_100_cases import build
from scripts.neral_generalization_100_ablation import (
    OLD_CONTROL_BITS, OLD_GUARDED_BITS, OLD_VERIFIED_BITS,
    baseline_by_id, paired_comparison, raw_qlora_case, dual_routed_case,
    DIAGNOSTIC_IDS, classify_regressions, expected_counterfactual,
)
from scripts.neral_generalization_100_run import BASE_TAG, TUNED_TAG, SYSTEM


class RealAblationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build()

    def test_pinned_original_results_and_counterfactual(self):
        self.assertEqual(len(OLD_CONTROL_BITS), 100)
        self.assertEqual(len(OLD_GUARDED_BITS), 100)
        self.assertEqual(len(OLD_VERIFIED_BITS), 100)
        self.assertEqual(OLD_CONTROL_BITS.count("1"), 29)
        self.assertEqual(OLD_GUARDED_BITS.count("1"), 40)
        self.assertEqual(OLD_VERIFIED_BITS.count("1"), 15)
        original=baseline_by_id(self.cases)
        self.assertEqual(sum(original.values()), 29)
        self.assertEqual(len(DIAGNOSTIC_IDS), 5)
        self.assertEqual(len(classify_regressions(self.cases)), 5)
        replay=expected_counterfactual(self.cases)
        self.assertEqual(replay["counterfactual_correct"],43)
        self.assertEqual(replay["counterfactual_gained"],14)
        self.assertEqual(replay["counterfactual_lost"],0)
        self.assertTrue(replay["assumes_identical_raw_replays"])

    def test_qlora_raw_does_not_invoke_any_tir_or_solver(self):
        sample=self.cases[4]
        with patch("scripts.neral_generalization_100_ablation.raw_chat",
                   return_value=str(sample["expected_integer"])) as query:
            with patch("scripts.neral_generalization_100_ablation.run_experimental_tir",
                       side_effect=AssertionError("no TIR")):
                result=raw_qlora_case(sample, 1234)
        self.assertTrue(result["correct"])
        self.assertEqual(result["output_integer"],sample["expected_integer"])
        self.assertEqual(query.call_count,1)
        model,messages,seed=query.call_args.args
        self.assertEqual(model,TUNED_TAG)
        self.assertEqual(messages,[{"role":"system","content":SYSTEM},
                                   {"role":"user","content":sample["question"]}])
        self.assertEqual(seed,1234)

    def test_dual_uses_base_model_for_controller_internal_raw_fallback(self):
        sample=self.cases[0]
        called=[]
        def backend(tag,messages,seed):
            called.append((tag,[dict(x) for x in messages],seed))
            if tag!=BASE_TAG:raise AssertionError("raw callback must be untrained")
            return str(sample["expected_integer"])
        def fake_tir(messages,respond,*,extract_constraints):
            got=respond(messages)
            self.assertEqual(got,str(sample["expected_integer"]))
            return {"text":got,"mode":"HIGH","model_generated":True,
                    "verified":False,"verification_status":"base_fallback_unverified",
                    "verification_scope":"none","calls_used":1,
                    "tool_calls_succeeded":0,"tool_calls_triggered":0,
                    "seconds":0.01}
        with patch("scripts.neral_generalization_100_ablation.raw_chat",side_effect=backend), \
             patch("scripts.neral_generalization_100_ablation.run_experimental_tir",side_effect=fake_tir):
            result=dual_routed_case(sample,10)
        self.assertTrue(result["correct"])
        self.assertEqual(result["selection_policy"],"raw_qwen_already_selected")
        self.assertEqual([x[0] for x in called],[BASE_TAG])

    def test_dual_tool_and_schema_routing_qlora_but_unverified_answer_base(self):
        sample=self.cases[0]
        called=[];extracted=[]
        def backend(tag,messages,seed):
            called.append((tag,messages,seed))
            return str(sample["expected_integer"])
        def schema(tag,turns,*,seed,repair):
            extracted.append((tag,turns,repair))
            return "{}"
        def fake_tir(messages,respond,*,extract_constraints):
            modified=[dict(x) for x in messages]
            modified[0]["content"] += " TOOL PLANNING"
            respond(modified)
            extract_constraints([{"role":"system","content":"SCHEMA"},
                                 {"role":"user","content":"QUESTION"}])
            return {"text":"999","mode":"HIGH","model_generated":True,
                    "verified":False,"verification_status":"model_extracted_schema_unverified",
                    "verification_scope":"model_extracted_finite_domain",
                    "calls_used":2,"tool_calls_succeeded":1,
                    "tool_calls_triggered":1,"seconds":0.01}
        with patch("scripts.neral_generalization_100_ablation.raw_chat",side_effect=backend), \
             patch("scripts.neral_generalization_100_ablation.query_structured",side_effect=schema), \
             patch("scripts.neral_generalization_100_ablation.run_experimental_tir",side_effect=fake_tir):
            result=dual_routed_case(sample,100)
        self.assertEqual([x[0] for x in called],[TUNED_TAG,BASE_TAG])
        self.assertEqual([x[0] for x in extracted],[TUNED_TAG])
        self.assertEqual(result["selection_policy"],"raw_qwen_due_to_unverified_tir")
        self.assertEqual(result["fallback_model"],BASE_TAG)
        self.assertTrue(result["correct"])

    def test_verified_tool_answer_is_not_replaced_with_base(self):
        sample=self.cases[51]
        report={"text":str(sample["expected_integer"]),"mode":"HIGH",
           "model_generated":False,"verified":True,
           "verification_status":"verified",
           "verification_scope":"exact_user_python_execution",
           "review_method":"exact_user_ast_match",
           "tool_used":"neral_python_sandbox",
           "tool_calls_triggered":1,"tool_calls_succeeded":1,
           "calls_used":0,"seconds":0.02}
        with patch("scripts.neral_generalization_100_ablation.raw_chat",
                   side_effect=AssertionError("raw forbidden")) as query, \
             patch("scripts.neral_generalization_100_ablation.run_experimental_tir",
                   return_value=report):
            result=dual_routed_case(sample,55)
        self.assertTrue(result["correct"])
        self.assertEqual(result["selection_policy"],"independently_verified_tir")
        self.assertFalse(result["fallback_model"])
        query.assert_not_called()

    def test_paired_100_derived_from_actual_original_baseline(self):
        prior=baseline_by_id(self.cases)
        replays=[{"id":c["id"],"category":c["category"],
                  "correct":bool(prior[c["id"]]),"status":"completed",
                  "output_integer":100,"seconds":0.4,
                  "selection_policy":"raw_qwen_already_selected"}
                  for c in self.cases]
        stats=paired_comparison(self.cases,replays,"test")
        self.assertEqual(stats["candidate_correct"],29)
        self.assertEqual(stats["gained"],0)
        self.assertEqual(stats["lost"],0)
        self.assertEqual(stats["by_category"]["mathematics"]["control"],11)
        self.assertEqual(stats["by_category"]["logic"]["control"],0)
        self.assertEqual(stats["by_category"]["programming"]["control"],5)
        self.assertEqual(stats["by_category"]["mixed"]["control"],13)

if __name__=="__main__":
    unittest.main()
