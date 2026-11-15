"""Unseen prospective100 harness's contract tests, CPU only, fixed QA seed.

No real prospective seed/corpus is created or read by these tests.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_1_1_prospective_blind100 import (
    AUTHORIZE,BASE_TAG,TUNED_TAG,CATS,CONTROL_MAX_TOKENS,
    PILOT_SEED,base_packet,score_final,tally,frozen_cases,
    write_new,sha,json_bytes,recover_unstarted_future_corpus
)

class Prospective100HarnessCPU(unittest.TestCase):
    def test_100_distinct_25_each_families_and_casepack_sha(self):
        rows,blob=frozen_cases(PILOT_SEED)
        self.assertEqual(len(rows),100)
        self.assertEqual(len({x["question_sha256"] for x in rows}),100)
        self.assertEqual(sum(x["category"]==CATS[0] for x in rows),25)
        for cat in CATS:
            items=[x for x in rows if x["category"]==cat]
            self.assertEqual(len(items),25)
            self.assertEqual(len({x["family"] for x in items}),5)
            for row in items:
                self.assertEqual(row["question_sha256"],sha(row["question"].encode("utf8")))
                self.assertEqual(type(row["gold_integer"]),int)
        self.assertEqual(sha(blob),hashlib.sha256(blob).hexdigest())

    def test_no_partial_or_empty_output_is_scored(self):
        for done in ("length","transport_error","incomplete",None):
            result=score_final("FINAL_ANSWER: 72",done,72)
            self.assertFalse(result["strict_correct"])
            self.assertFalse(result["offline_correct"])
        for output in ("", " ", "No answer, perhaps 72", "FINAL_ANSWER: 72\nFINAL_ANSWER: 11"):
            result=score_final(output,"stop",72)
            self.assertFalse(result["strict_correct"])

    def test_strict_terminal_and_offline_crosscheck_symmetric(self):
        self.assertTrue(score_final("FIRST STEP\nFINAL_ANSWER: 47","stop",47)["strict_correct"])
        self.assertFalse(score_final("FINAL_ANSWER: 48","stop",47)["strict_correct"])
        # Standalone clear explanatory prose can be recovered OFFLINE but
        # cannot change the frozen strict primary gate.
        alternative=score_final("The result is 47.","stop",47)
        self.assertFalse(alternative["strict_correct"])

    def test_deterministic_non_thinking_direct_control_settings(self):
        msg=[{"role":"system","content":"Shared system prompt"},
             {"role":"user","content":"Calculate an integer"}]
        packet=base_packet(msg)
        self.assertEqual(packet["model"],BASE_TAG)
        self.assertEqual(packet["messages"],msg)
        self.assertEqual(packet["options"]["num_predict"],2048)
        self.assertEqual(packet["options"]["temperature"],0)
        self.assertEqual(packet["options"]["seed"],0)
        self.assertIs(packet["think"],False)

    def test_all_correct_no_regression_and_85_gate_on_fakes(self):
        cases,_=frozen_cases(PILOT_SEED)
        rows=[]
        for c in cases:
            rows.append({"id":c["id"],"category":c["category"],
                "base_strict_correct":True,"neral_strict_correct":True,
                "base_offline_correct":True,"neral_offline_correct":True,
                "base_seconds":1.0,"neral_seconds":2.0,
                "strategy":"DIRECT","independent_verified":False})
        result=tally(cases,rows)
        self.assertEqual(result["strict"],{"base":100,"neral":100,"gained":0,"lost":0})
        self.assertTrue(result["pass_gate"])

    def test_single_regression_blocks_even_if_total_above_85(self):
        cases,_=frozen_cases(PILOT_SEED)
        rows=[]
        for i,c in enumerate(cases):
            rows.append({"id":c["id"],"category":c["category"],
                "base_strict_correct":True,
                "neral_strict_correct":i!=1,
                "base_offline_correct":True,
                "neral_offline_correct":i!=1,
                "base_seconds":1.0,"neral_seconds":2.0,
                "strategy":"PYTHON_SANDBOX",
                "independent_verified":i!=1})
        result=tally(cases,rows)
        self.assertEqual(result["strict"]["neral"],99)
        self.assertEqual(result["strict"]["lost"],1)
        self.assertFalse(result["pass_gate"])

    def test_100_cannot_be_scored_partially_or_with_duplicates(self):
        cases,_=frozen_cases(PILOT_SEED)
        row={"id":cases[0]["id"],"category":cases[0]["category"]}
        with self.assertRaises(ValueError):
            tally(cases,[row]*100)
        with self.assertRaises(ValueError):
            tally(cases,[])

    def test_exact_original_seed_and_casepack_recovered_without_new_draw(self):
        cases,blob=frozen_cases(PILOT_SEED)
        pilot={"cases":[{"case_id":"cpu-"+str(k)} for k in range(4)],
               "losses":0,"base_correct":4,"bounded_correct":4}
        receipt=json_bytes(pilot)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_new(root/"prospective100-sealed-cases.jsonl",blob)
            write_new(root/"prospective100-seed-secret.txt",
                (PILOT_SEED+"\n").encode("utf8"))
            write_new(root/"pilot-safety-raw.json",receipt)
            # QA test seed is deliberately NOT 64 hex chars and must fail.
            with self.assertRaisesRegex(RuntimeError,"SHA mismatch"):
                recover_unstarted_future_corpus(root,sha(blob),sha(receipt))

    def test_recovered_sha_tamper_fails_closed_without_seed_redraw(self):
        # A fixed SHA-derived QA seed is NOT the actual confidential GPU seed.
        fake_seed=hashlib.sha256(PILOT_SEED.encode()).hexdigest()
        cases,blob=frozen_cases(fake_seed)
        pilot=json_bytes({"cases":[{"case_id":"cpu-"+str(k)} for k in range(4)],
                          "losses":0,"base_correct":4,"bounded_correct":4})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_new(root/"prospective100-sealed-cases.jsonl",blob)
            write_new(root/"prospective100-seed-secret.txt",(fake_seed+"\n").encode())
            write_new(root/"pilot-safety-raw.json",pilot)
            from unittest.mock import patch
            with patch("scripts.neral_1_1_prospective_blind100.secrets.token_hex",
                       side_effect=AssertionError("Do not draw a fresh seed")):
                recovered=recover_unstarted_future_corpus(root,sha(blob),sha(pilot))
            self.assertEqual(recovered[0],fake_seed)
            self.assertEqual(recovered[2],blob)
            with self.assertRaisesRegex(RuntimeError,"SHA mismatch"):
                recover_unstarted_future_corpus(root,sha(blob),sha(b"tampered"))
            (root/"prospective100-sealed-cases.jsonl").write_bytes(blob+b"changed")
            with self.assertRaisesRegex(RuntimeError,"SHA mismatch"):
                recover_unstarted_future_corpus(root,sha(blob),sha(pilot))

    def test_artifacts_created_once_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"sealed.json"
            write_new(file,b"secret-seal")
            self.assertEqual(file.read_bytes(),b"secret-seal")
            with self.assertRaises(FileExistsError):
                write_new(file,b"tamper")

if __name__=="__main__":unittest.main()
