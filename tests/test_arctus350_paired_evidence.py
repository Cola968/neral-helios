"""Research gate: compare REAL user-supplied native checkpoint observations, never invent scores.

TDD RED first. Fixtures here are synthetic shape examples only.
"""
from __future__ import annotations
import copy
import hashlib
import json
import unittest
from frontier350.arctus350_paired_evidence import compare_native_runs, read_run_json, EvidenceError

TOKENIZER_SHA = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"


def fixture(sha="a"*64):
    cases=[
        {"id":f"heldout-exact-{i}","category":"instruction_exact",
         "correct":True,"eos":True} for i in range(4)
    ]+[
        {"id":f"heldout-math-{i}","category":"math",
         "correct":i<3,"eos":True} for i in range(4)
    ]
    spec=[{"id":c["id"],"category":c["category"]} for c in cases]
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    return {
        "schema":"arctus350-native-case-observations-v1",
        "model_family":"neral-arctus350",
        "evaluation_source":"native-checkpoint-readonly",
        "benchmark_id":"arctus350-internal-dev-synthetic-v0",
        "benchmark_sha256":fingerprint,
        "prompt_template_sha256":"d"*64,
        "scoring_version":"exact-instruction-v1",
        "tokenizer_sha256":TOKENIZER_SHA,
        "checkpoint_sha256":sha,
        "sampling":{"temperature":0,"top_k":0,"max_new_tokens":64},
        "cases":cases,
    }


class PairedNativeEvidenceTests(unittest.TestCase):
    def test_candidate_improvement_is_counted_but_not_called_a_model_promotion(self):
        baseline=fixture("a"*64)
        candidate=fixture("b"*64)
        candidate["cases"][-1]["correct"]=True
        report=compare_native_runs(baseline,candidate)
        self.assertEqual(report["case_count"],8)
        self.assertEqual(report["baseline_checkpoint_sha256"],"a"*64)
        self.assertEqual(report["candidate_checkpoint_sha256"],"b"*64)
        self.assertEqual(report["overall"]["baseline_correct"],7)
        self.assertEqual(report["overall"]["candidate_correct"],8)
        self.assertEqual(report["overall"]["gained"],1)
        self.assertEqual(report["overall"]["lost"],0)
        self.assertEqual(report["categories"]["math"]["baseline_correct"],3)
        self.assertEqual(report["categories"]["math"]["candidate_correct"],4)
        self.assertEqual(report["decision"],"INSUFFICIENT_STATISTICAL_EVIDENCE")
        self.assertEqual(report["statistics"]["status"],"INSUFFICIENT_EVIDENCE")
        self.assertFalse(report["provenance_verified"])
        self.assertEqual(report["release_authorization"],"none")
        self.assertNotIn("passed",str(report).lower())

    def test_regression_in_one_category_blocks_candidate_even_with_higher_total(self):
        baseline=fixture("a"*64)
        candidate=fixture("b"*64)
        # Math: +3 new correct, exact-instructions: -1 -> aggregate +2.
        for row in baseline["cases"]:
            if row["category"]=="math":row["correct"]=False
        for row in candidate["cases"]:
            if row["category"]=="math":row["correct"]=row["id"]!="heldout-math-3"
            if row["id"]=="heldout-exact-0":row["correct"]=False
        report=compare_native_runs(baseline,candidate)
        self.assertEqual(report["overall"]["gained"],3)
        self.assertEqual(report["overall"]["lost"],1)
        self.assertEqual(report["decision"],"REJECT_CATEGORY_REGRESSION")

    def test_missing_gain_not_promoted(self):
        baseline=fixture("a"*64)
        candidate=fixture("b"*64)
        self.assertEqual(compare_native_runs(baseline,candidate)["decision"],"NO_MEASURED_GAIN")

    def test_mismatched_cases_and_settings_are_not_comparable(self):
        for field,value in [
            ("tokenizer_sha256","9"*64),
            ("benchmark_sha256","f"*64),
            ("prompt_template_sha256","7"*64),
            ("scoring_version","different"),
            ("benchmark_id","other"),
            ("evaluation_source","tool:calculator"),
            ("model_family","qwen"),
            ("sampling",{"temperature":0.7,"top_k":0,"max_new_tokens":64}),
        ]:
            with self.subTest(field=field):
                reference=fixture("a"*64)
                candidate=fixture("b"*64)
                candidate[field]=value
                with self.assertRaises(EvidenceError):
                    compare_native_runs(reference,candidate)
        for modification in ("missing","extra","category","duplicate"):
            with self.subTest(modification=modification):
                reference=fixture("a"*64)
                candidate=fixture("b"*64)
                if modification=="missing":candidate["cases"].pop()
                if modification=="extra":candidate["cases"].append({"id":"leaked-test-1","category":"math","correct":True,"eos":True})
                if modification=="category":candidate["cases"][0]["category"]="math"
                if modification=="duplicate":candidate["cases"][1]["id"]=candidate["cases"][0]["id"]
                with self.assertRaises(EvidenceError):
                    compare_native_runs(reference,candidate)

    def test_bad_score_types_raw_generation_and_checkpoint_reuse_fail_closed(self):
        for field,bad in [("correct",1),("correct","true"),("correct",None),
                          ("eos","true"),("eos",0)]:
            with self.subTest(field=field,bad=bad):
                baseline=fixture("a"*64)
                candidate=fixture("b"*64)
                candidate["cases"][0][field]=bad
                with self.assertRaises(EvidenceError):
                    compare_native_runs(baseline,candidate)
        for key,value in (("raw_prompt","private model text"),("model_reply","secret user text")):
            candidate=fixture("b"*64)
            candidate["cases"][0][key]=value
            with self.assertRaises(EvidenceError):
                compare_native_runs(fixture("a"*64),candidate)
        with self.assertRaises(EvidenceError):
            compare_native_runs(fixture("a"*64),fixture("a"*64))

    def test_wrong_fingerprint_and_original_tokenizer_drift_rejected(self):
        corrupted=fixture()
        corrupted["cases"][0]["id"]="new-id-with-relabel"
        with self.assertRaises(EvidenceError):
            compare_native_runs(corrupted,fixture("b"*64))
        baseline=fixture()
        candidate=fixture("b"*64)
        baseline["tokenizer_sha256"]=candidate["tokenizer_sha256"]="a"*64
        with self.assertRaises(EvidenceError):
            compare_native_runs(baseline,candidate)

    def test_duplicate_json_keys_and_nonfinite_json_constants_rejected(self):
        with self.assertRaises(EvidenceError):
            read_run_json('{"schema":"first","schema":"second"}')
        for raw in ('{"v":NaN}','{"v":Infinity}','[1,2,3]'):
            with self.subTest(raw=raw),self.assertRaises(EvidenceError):
                read_run_json(raw)

    def test_input_order_compatible_and_no_raw_answer_leak_in_report(self):
        base=fixture("a"*64)
        candidate=fixture("b"*64)
        candidate["cases"]=list(reversed(candidate["cases"]))
        report=compare_native_runs(base,candidate)
        self.assertEqual(report["decision"],"NO_MEASURED_GAIN")
        self.assertNotIn("heldout-exact-0",json.dumps(report))
        self.assertNotIn("cases",report)
        self.assertEqual(base,fixture("a"*64))


if __name__=="__main__":
    unittest.main()
