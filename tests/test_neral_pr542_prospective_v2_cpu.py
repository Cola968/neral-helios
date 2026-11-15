"""CPU-only TDD for prospective PR542 v2; synthetic records, no GPU, no old test answers."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_pr542_prospective_v2 import (
    SOURCE_SHA,CATEGORIES,FORMAT_SUFFIX,append_durable,classify_proof,dry_run,
    extract,messages_for,prompt_for,records,row,summarize,sha256_file
)

def synthetic_case(cat,index):
    return {"id":f"synthetic-{cat}-{index:02d}","category":cat,
            "family":"synthetic","question":"A synthetic arithmetic question?",
            "question_sha256":"0"*64,"gold_integer":index+13}

class ProspectiveHarnessCPU(unittest.TestCase):
    def test_equal_input_contract_and_fixed_suffix(self):
        case=synthetic_case("logic",1)
        msgs=messages_for(case)
        self.assertEqual(msgs[1]["content"],prompt_for(case))
        self.assertTrue(msgs[1]["content"].endswith(FORMAT_SUFFIX))
        self.assertEqual(msgs[0]["role"],"system")

    def test_same_extractor_for_both_arms_prose_and_integer(self):
        for sample in ("Final answer: 42","The answer is 42.",
                       "We calculated 6 and 7; the result is 42.",
                       "NERAL_FINAL_TEXT: FINAL_ANSWER: 42",
                       "FINAL_ANSWER: 42"):
            with self.subTest(sample=sample):
                self.assertEqual(extract(sample)[0],42)
        self.assertIsNone(extract("I considered 5 and 10 and 11")[0])

    def test_cpu_dummy_roundtrip_and_readonly_reread(self):
        with tempfile.TemporaryDirectory() as root:
            location=Path(root)/"persisted"
            dry_run(location)
            data=records(location/"raw-200.jsonl",4)
            self.assertEqual(len(data),4)
            self.assertTrue(all(p["correct"] for p in data))
            receipt=json.loads((location/"dummy-receipt.json").read_text())
            self.assertEqual(receipt["sha256"],sha256_file(location/"raw-200.jsonl"))
            with self.assertRaises(FileExistsError):
                dry_run(location)

    def test_incomplete_pair_is_rejected_before_reporting(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/"raw-200.jsonl"
            c=synthetic_case("logic",1)
            with target.open("x",encoding="utf8"):pass
            append_durable(target,row(c,"qwen","The answer is 14.",0.1,
                                      decision="raw_qwen",proof={"z3":False,"sandbox":False}))
            with self.assertRaises(RuntimeError):
                records(target,2)

    def test_z3_requires_complete_source_proof_not_consensus(self):
        positive={"selection_policy":"independently_source_verified_nl",
                  "proof_backend":"z3_unsat","verified":True,
                  "safe_to_promote":True,"selected_extractor":"qlora"}
        self.assertTrue(classify_proof(positive,"logic_z3")["z3"])
        for k,value in (("proof_backend","consensus"),
                        ("safe_to_promote",False),
                        ("verified",False)):
            altered={**positive,k:value}
            self.assertFalse(classify_proof(altered,"logic_z3")["z3"])

    def test_sandbox_requires_source_and_exact_ast_proof(self):
        positive={"selection_policy":"independently_verified_tir",
                  "verified":True,"model_generated":False,
                  "tool_used":"neral_python_sandbox","tool_calls_succeeded":1,
                  "verification_status":"verified",
                  "verification_scope":"exact_user_python_execution",
                  "review_method":"exact_user_ast_match"}
        self.assertTrue(classify_proof(positive,"tir_sandbox_pr537")["sandbox"])
        self.assertFalse(classify_proof({**positive,"model_generated":True},
                                        "tir_sandbox_pr537")["sandbox"])

    def test_100_pairs_report_correct_net_and_regression_ids(self):
        allrows=[]
        for cat in CATEGORIES:
            for idx in range(1,26):
                c=synthetic_case(cat,idx)
                b=idx<=10
                n=idx<=12
                allrows.append(row(c,"qwen",f"FINAL_ANSWER: {c['gold_integer'] if b else 0}",
                                  .1,decision="raw_qwen",proof={"z3":False,"sandbox":False}))
                allrows.append(row(c,"neral",f"The answer is {c['gold_integer'] if n else 0}.",
                                  .2,decision="raw_qwen_due_to_unverified_tir",
                                  proof={"z3":False,"sandbox":False}))
        summary=summarize(allrows,sealed_hash="f"*64)
        self.assertEqual(summary["baseline_correct"],40)
        self.assertEqual(summary["candidate_correct"],48)
        self.assertEqual(summary["net_gain"],8)
        self.assertEqual(summary["regressions"],0)
        self.assertTrue(summary["release_gate_passed"])
        changed=next(x for x in allrows
                     if x["arm"]=="neral" and x["id"]=="synthetic-logic-02")
        changed["reply"]="FINAL_ANSWER: -999"
        changed["extracted_integer"]=-999
        changed["correct"]=False
        summary=summarize(allrows,sealed_hash="f"*64)
        self.assertEqual(summary["regressions"],1)
        self.assertIn("synthetic-logic-02",summary["regression_ids"])
        self.assertFalse(summary["release_gate_passed"])

    def test_no_partial_reports(self):
        with self.assertRaises(RuntimeError):
            summarize([],sealed_hash="f"*64)

if __name__=="__main__":
    unittest.main()
