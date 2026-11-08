"""TDD contracts for NERAL v8 post-training, not-in-SFT logic evaluation.

This fixture uses synthetic model replies. Passing tests means the scorer and
data are valid, NOT that any LoRA weights actually improved.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v8_unseen_suite import (
    CASES_PER_FAMILY, FAMILIES, PROMPT_SEED, all_cases, pilot_cases,
    fresh_oracle, audit_fresh_suite, bytes_for_cases
)
from scripts.neral_v8_unseen_eval import (
    ARMS, STATUS, report_scores, validate_private_rows,
    evaluate_candidate_gate, verify_v8_receipt,
)


def _answers(cases):
    return [
        {"id":row["id"],"text":"FINAL_ANSWER: "+row["answer"],
         "complete":True,"stop_reason":"eos","new_tokens":24}
        for row in cases
    ]


class V8UnseenContracts(unittest.TestCase):
    def test_128_cases_16_families_64_64_split_and_deterministic(self):
        rows=all_cases()
        self.assertEqual(len(rows),128)
        self.assertEqual(len(set(x["id"] for x in rows)),128)
        self.assertEqual(len(set(x["prompt"] for x in rows)),128)
        self.assertEqual(Counter(r["part"] for r in rows),
                         {"retention":64,"transfer":64})
        self.assertEqual(len(FAMILIES["retention"]),8)
        self.assertEqual(len(FAMILIES["transfer"]),8)
        self.assertEqual(set(Counter(x["family"] for x in rows).values()),{8})
        self.assertEqual(bytes_for_cases(rows),bytes_for_cases(all_cases()))
        self.assertIsInstance(PROMPT_SEED,int)
        self.assertTrue(audit_fresh_suite(rows)["exact_prompt_disjoint_from_sft"])

    def test_32_case_pilot_is_exactly_balanced_across_families(self):
        pilot=pilot_cases()
        self.assertEqual(len(pilot),32)
        self.assertEqual(Counter(r["part"] for r in pilot),
                         {"retention":16,"transfer":16})
        self.assertEqual(set(Counter(r["family"] for r in pilot).values()),{2})
        self.assertEqual(pilot,pilot_cases())
        self.assertEqual([r["id"] for r in pilot],
                         [r["id"] for r in all_cases() if int(r["ordinal"])<2])

    def test_all_oracles_independent_of_stored_gold(self):
        rows=all_cases()
        for row in rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(int(row["answer"]),fresh_oracle(row))
                self.assertTrue(row["prompt"])
                self.assertLess(len(row["prompt"]),2200)
                self.assertTrue(row["part"] in ("retention","transfer"))
        sabotaged=deepcopy(rows[0])
        sabotaged["answer"]=str(int(sabotaged["answer"])+1)
        self.assertNotEqual(fresh_oracle(sabotaged),int(sabotaged["answer"]))
        self.assertRaises(ValueError,audit_fresh_suite,
                          [sabotaged]+rows[1:])

    def test_reject_reordered_and_injected_response_rows(self):
        cases=pilot_cases()
        correct=_answers(cases)
        self.assertTrue(validate_private_rows(cases,correct))
        reverse=list(correct)
        reverse[0],reverse[1]=reverse[1],reverse[0]
        with self.assertRaises(ValueError):
            validate_private_rows(cases,reverse)
        edits=({"complete":1},{"new_tokens":True},{"new_tokens":513},
               {"stop_reason":"length","new_tokens":3},
               {"stop_reason":"length","complete":True},
               {"complete":False,"stop_reason":"eos"},
               {"text":"FINAL_ANSWER: 2\u0000"},
               {"payload":"from remote website"})
        for edit in edits:
            bad=[dict(r) for r in correct]
            bad[0].update(edit)
            with self.subTest(edit=edit),self.assertRaises(ValueError):
                validate_private_rows(cases,bad)

    def test_precise_scoring_and_paired_regression_without_claim(self):
        cases=pilot_cases()
        answers={arm:_answers(cases) for arm in ARMS}
        # Simulate a candidate gain over both Qwen and predecessor, but
        # with exactly one new regression: the safety gate must reject.
        answers["qwen"][0]["text"]="FINAL_ANSWER: 99999"
        answers["v5"][0]["text"]="FINAL_ANSWER: 99999"
        answers["v7"][0]["text"]="FINAL_ANSWER: 99999"
        answers["v8"][1]["text"]="FINAL_ANSWER: 99999"
        answers["v5"][2].update(text="FINAL_ANSWER: "+cases[2]["answer"],
                                 new_tokens=512,complete=False,
                                 stop_reason="length")
        result=report_scores(cases,answers,"pilot")
        self.assertEqual(result["status"],STATUS)
        self.assertEqual(result["models"]["v8"]["correct"],31)
        self.assertEqual(result["models"]["v5"]["length_censored"],1)
        self.assertEqual(result["paired"]["v8_minus_v5"]["improvements"],2)
        self.assertEqual(result["paired"]["v8_minus_v5"]["regressions"],1)
        self.assertFalse(result["gate"]["eligible_for_review"])
        self.assertFalse(result["weights_promoted"])
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["independent_ood_proven"])
        payload=json.dumps(result)
        self.assertNotIn(cases[0]["prompt"],payload)
        self.assertNotIn(cases[0]["id"],payload)

    def test_claim_requires_64_transfer_and_64_retention_not_pilot(self):
        full=all_cases()
        right={arm:_answers(full) for arm in ARMS}
        pilot=report_scores(pilot_cases(),{a:_answers(pilot_cases())
                                          for a in ARMS},"pilot")
        self.assertFalse(pilot["gate"]["eligible_for_review"])
        self.assertIn("pilot",pilot["gate"]["failed_checks"])
        report=report_scores(full,right,"full")
        self.assertEqual(report["models"]["v8"]["correct"],128)
        self.assertFalse(report["gate"]["eligible_for_review"])
        self.assertFalse(report["independent_ood_proven"])
        self.assertFalse(report["weights_promoted"])
        # Synthetic external labels cannot manufacture a release approval.
        self.assertTrue("transfer_gain" in report["gate"]["failed_checks"])

    def test_gate_requires_significant_transfer_and_retention_safety(self):
        ret=[{"id":f"ret-{i}","family":f"f{i//8}",
              "baseline_correct":(i%5!=0),
              "candidate_correct":(i%5!=0)} for i in range(64)]
        transfer=[{"id":f"tra-{i}","family":f"t{i//8}",
                   "baseline_correct":False,
                   "candidate_correct":i<12} for i in range(64)]
        result=evaluate_candidate_gate(ret,transfer,pilot=False)
        self.assertTrue(result["eligible_for_review"])
        self.assertFalse(result["adapter_promoted"])
        self.assertFalse(result["release_validated"])
        transfer[0]["candidate_correct"]=False
        transfer[1]["candidate_correct"]=False
        transfer[2]["candidate_correct"]=False
        transfer[3]["candidate_correct"]=False
        result=evaluate_candidate_gate(ret,transfer,pilot=False)
        self.assertFalse(result["eligible_for_review"])

    def test_v8_receipt_must_match_actual_local_weight_bytes(self):
        from scripts.neral_v8_selfverify_data import build_selfverify_experiment
        from scripts.neral_v6_fresh_dev_eval import V5_SHA
        from scripts.neral_v8_selfverify_train import MODEL_STATUS
        from hashlib import sha256
        _,_,meta=build_selfverify_experiment()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            adapter=root/"adapter"
            adapter.mkdir()
            weights=adapter/"adapter_model.safetensors"
            weights.write_bytes(b"fake selfverify candidate test only")
            (adapter/"adapter_config.json").write_text(
                '{"peft_type":"LORA","r":16,"task_type":"CAUSAL_LM",'
                '"target_modules":["q_proj","k_proj","v_proj","o_proj",'
                '"gate_proj","up_proj","down_proj"]}',encoding="utf8")
            proof={
                "status":MODEL_STATUS,"steps_completed":240,
                "train_rows":960,"validation_rows":128,
                "source_counts":meta["source_counts"],
                "checkpoint_mode":"best_validation_loss",
                "checkpoint_interval":40,"checkpoint_selected_step":200,
                "train_sha256":meta["train_sha256"],
                "validation_sha256":meta["validation_sha256"],
                "candidate_adapter_sha256":sha256(weights.read_bytes()).hexdigest(),
                "predecessor_adapter_sha256":V5_SHA,
                "release_validated":False,
                "adapter_promoted":False,
                "independent_transfer_proven":False,
            }
            receipt=root/"scalar-private.json"
            receipt.write_text(json.dumps(proof),encoding="utf8")
            observed=verify_v8_receipt(receipt,adapter)
            self.assertEqual(observed["candidate_adapter_sha256"],
                             proof["candidate_adapter_sha256"])
            proof["steps_completed"]=239
            receipt.write_text(json.dumps(proof),encoding="utf8")
            with self.assertRaises(ValueError):
                verify_v8_receipt(receipt,adapter)

if __name__=="__main__":
    unittest.main()
