"""CPU-only paired 12-task TIR benchmark contracts, independent of GPU."""
from __future__ import annotations
import copy
import hashlib
import unittest
from scripts.neral_high12_manual import select_cases
from scripts.neral_low_high_quality import make_prompt
from scripts.neral_tir12_manual import score_tir_pairs, Tir12GateError, AUTHORIZATION

DIGEST="sha256:"+"a"*64

def fixture(*,raw_correct=7,tir_correct=7):
    cases=select_cases(271828)
    rows=[]
    for i,case in enumerate(cases):
        arms={}
        for arm,good in (("raw_qwen35",i<raw_correct),("neral_high_router",i<tir_correct)):
            arms[arm]={"text":"FINAL_ANSWER: "+str(case["answer"] if good else case["answer"]+1000),
                "complete":True,"failed":False,"failure_reason":None,"seconds":1.0,
                "model":"qwen3.5:9b","model_digest":DIGEST,"profile":"measured_nonthinking_2048",
                "tool_used":None,"tool_calls_triggered":0,"tool_calls_succeeded":0}
        rows.append({"id":case["id"],"prompt_sha256":hashlib.sha256(
            make_prompt(case).encode("utf8")).hexdigest(),"arms":arms})
    return cases,rows

class PairedTir12Contracts(unittest.TestCase):
    def test_fixed_authorization_and_balanced_families(self):
        self.assertEqual(AUTHORIZATION,"NERAL_TIR12_GPU_MANUAL_ONLY")
        cases,rows=fixture()
        self.assertEqual(len(cases),12)
        self.assertEqual(sorted([sum(c["domain"]==x for c in cases)
                     for x in ("math","reasoning","code_trace")]),[4,4,4])

    def test_equal_scores_are_development_only_and_never_release_approval(self):
        cases,rows=fixture()
        r=score_tir_pairs(cases,rows,model_digest=DIGEST)
        self.assertTrue(r["development_gate_passed"])
        self.assertEqual(r["raw_correct"],7)
        self.assertEqual(r["tir_correct"],7)
        self.assertEqual(r["paired_wins"],0)
        self.assertFalse(r["release_approval"])
        self.assertFalse(r["independent_holdout_verified"])
        self.assertFalse(r["same_inference_budget"])
        self.assertEqual(r["raw_first_turn_token_budget"],2048)
        self.assertEqual(r["tir_turn0_token_budget"],384)
        self.assertEqual(r["tir_fallback_token_budget"],2048)
        self.assertEqual(r["tir_turn0_temperature"],0)

    def test_tool_metrics_are_scalar_and_observation_is_never_unverified_stdout(self):
        cases,rows=fixture(raw_correct=5,tir_correct=6)
        item=rows[0]["arms"]["neral_high_router"]
        item.update(tool_used="neral_python_sandbox",tool_calls_triggered=1,tool_calls_succeeded=1)
        result=score_tir_pairs(cases,rows,model_digest=DIGEST)
        self.assertEqual(result["tool_calls_succeeded"],1)
        self.assertEqual(result["paired_wins"],1)
        self.assertNotIn(cases[0]["prompt"],str(result))
        self.assertNotIn("FINAL_ANSWER",str(result))

    def test_technical_failure_still_fails_gate_and_truncation_has_own_counter(self):
        cases,rows=fixture()
        rows[0]["arms"]["neral_high_router"].update(
            text="",complete=False,failed=True,failure_reason="timeout")
        r=score_tir_pairs(cases,rows,model_digest=DIGEST)
        self.assertFalse(r["development_gate_passed"])
        self.assertEqual(r["technical_failures"]["neral_high_router"],1)
        cases,rows=fixture()
        for arm in ("raw_qwen35","neral_high_router"):
            rows[0]["arms"][arm].update(
                text="",complete=False,failed=True,failure_reason="token_limit")
        r=score_tir_pairs(cases,rows,model_digest=DIGEST)
        self.assertEqual(r["error_counts"]["neral_high_router"]["length_truncated"],1)
        self.assertEqual(r["technical_failures"]["neral_high_router"],0)

    def test_bad_provenance_and_fake_tool_counts_rejected(self):
        cases,rows=fixture()
        for change in [
            {"tool_calls_triggered":0,"tool_calls_succeeded":1},
            {"tool_calls_triggered":2,"tool_calls_succeeded":1},
            {"tool_used":"neral_python_sandbox","tool_calls_triggered":0,"tool_calls_succeeded":0},
            {"model_digest":"sha256:"+"b"*64},
        ]:
            broken=copy.deepcopy(rows)
            broken[0]["arms"]["neral_high_router"].update(change)
            with self.assertRaises(Tir12GateError):
                score_tir_pairs(cases,broken,model_digest=DIGEST)
        broken=copy.deepcopy(rows)
        broken[0]["arms"]["raw_qwen35"]["tool_calls_triggered"]=1
        with self.assertRaises(Tir12GateError):
            score_tir_pairs(cases,broken,model_digest=DIGEST)

if __name__=="__main__":unittest.main()
