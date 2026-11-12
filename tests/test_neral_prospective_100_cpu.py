"""Prospective original questions: independent golds and exact corpus exclusions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_prospective_100_cases import (
    build,serialize,exclusion_receipt,shuffle_seed,
)
from scripts.neral_generalization_100_cases import build as old_dev
from scripts.neral_blind_100_cases import build as used_blind
from scripts.neral_qlora_dataset import generate, compact,load_holdout

ROOT=Path(__file__).resolve().parents[1]
HOLDOUT=ROOT/"datasets/neral_qwen35_holdout_v1.jsonl"

class New100Integrity(unittest.TestCase):
    def test_four_balanced_categories_and_five_family_coverage(self):
        sample=build("testing-only-seed-01")
        self.assertEqual(sample,build("testing-only-seed-01"))
        self.assertEqual(len(sample),100)
        self.assertEqual(len({x["question_sha256"] for x in sample}),100)
        for cat in ("mathematics","logic","programming","mixed"):
            current=[x for x in sample if x["category"]==cat]
            self.assertEqual(len(current),25)
            self.assertEqual(len({x["family"] for x in current}),5)
            for row in current:
                self.assertEqual(type(row["gold_integer"]),int)
                self.assertLess(abs(row["gold_integer"]),10**9)
                self.assertEqual(hashlib.sha256(row["question"].encode("utf8")).hexdigest(),row["question_sha256"])
        self.assertEqual(sample,[json.loads(z) for z in serialize(sample).decode().splitlines()])
        self.assertNotEqual(sample,build("testing-only-seed-02"))
        self.assertEqual(len(shuffle_seed("ab12"*10)),64)

    def test_exact_training_sha_and_prior_24_old100_and_used_blind_disjoint(self):
        excluded,_=load_holdout(HOLDOUT)
        generated=generate(pairs=360,exclusion_questions=excluded)
        payload="".join(compact(x)+"\n" for x in generated).encode()
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/"train.jsonl"
            original.write_bytes(payload)
            cases=build("test-fixed-not-for-gpu")
            manifest=exclusion_receipt(cases,original,HOLDOUT)
            self.assertEqual(manifest["count"],100)
            self.assertEqual(manifest["categories"],{
                "mathematics":25,"logic":25,"programming":25,"mixed":25})
            self.assertEqual(manifest["prospective100_sha256"],hashlib.sha256(serialize(cases)).hexdigest())
            for previous in (old_dev()[0],used_blind()[0]):
                corrupted=[dict(x) for x in cases]
                corrupted[0]={**corrupted[0],"question":previous["question"],
                    "question_sha256":hashlib.sha256(previous["question"].encode()).hexdigest()}
                with self.assertRaisesRegex(ValueError,"overlap"):
                    exclusion_receipt(corrupted,original,HOLDOUT)
            old_row=generated[0]
            leaked=old_row["messages"][1]["content"]
            broken=[dict(x) for x in cases]
            broken[0]={**broken[0],"question":leaked,
                       "question_sha256":hashlib.sha256(leaked.encode()).hexdigest()}
            with self.assertRaisesRegex(ValueError,"overlap"):
                exclusion_receipt(broken,original,HOLDOUT)
    def test_seed_is_never_hardcoded_as_original_model_test_evidence(self):
        self.assertNotIn("blind100-",build("testing-only-seed-01")[0]["id"])
        with self.assertRaises(ValueError):
            build("")
        with self.assertRaises(ValueError):
            build("A"*300)

if __name__=="__main__":
    unittest.main()
