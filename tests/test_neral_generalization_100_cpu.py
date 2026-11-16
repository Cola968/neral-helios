"""CPU checks for the fresh transfer gate; never require Ollama, CUDA or training."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.neral_generalization_100_cases import (
    CATEGORIES, SEED, TRAIN_SHA256, build, disjoint, normalized, sha,
)
from scripts.neral_qlora_dataset import compact, generate, load_holdout

ROOT=Path(__file__).resolve().parents[1]
HOLDOUT=ROOT/"datasets/neral_qwen35_holdout_v1.jsonl"

class FreshGeneralizationCpu(unittest.TestCase):
    def test_exact_100_four_categories_and_five_subfamilies(self):
        rows=build()
        self.assertEqual(len(rows),100)
        self.assertEqual(build(seed=SEED),rows)
        self.assertEqual(len(set(x["question_sha256"] for x in rows)),100)
        for category in CATEGORIES:
            group=[x for x in rows if x["category"]==category]
            self.assertEqual(len(group),25)
            self.assertEqual(len(set(x["subfamily"] for x in group)),
                             1 if category=="programming" else 5)
        for x in rows:
            self.assertEqual(x["question_sha256"],sha(x["question"]))
            self.assertIs(type(x["expected_integer"]),int)
            self.assertLessEqual(abs(x["expected_integer"]),100_000_000)

    def test_hash_exclusion_from_real_frozen_train_and_holdout(self):
        excluded,_=load_holdout(HOLDOUT)
        train=generate(pairs=360,exclusion_questions=excluded)
        raw="".join(compact(row)+"\n" for row in train)
        self.assertEqual(hashlib.sha256(raw.encode("utf8")).hexdigest(),TRAIN_SHA256)
        with tempfile.TemporaryDirectory() as dirname:
            path=Path(dirname)/"train.jsonl"
            path.write_text(raw,encoding="utf8")
            result=disjoint(build(),path,HOLDOUT)
            self.assertEqual(result["case_count"],100)
            self.assertEqual(result["categories"],{c:25 for c in CATEGORIES})
            bad=build()
            bad[0]={**bad[0],"question_sha256":sha(train[0]["messages"][1]["content"]),
                    "question":train[0]["messages"][1]["content"]}
            with self.assertRaisesRegex(ValueError,"contamination"):
                disjoint(bad,path,HOLDOUT)
            bad2=build()
            case=json.loads(HOLDOUT.read_text(encoding="utf8").splitlines()[0])
            bad2[0]={**bad2[0],"question":case["question"],
                     "question_sha256":sha(case["question"])}
            with self.assertRaisesRegex(ValueError,"contamination"):
                disjoint(bad2,path,HOLDOUT)

    def test_no_earlier_corpus_questions_even_normalized(self):
        rows=build()
        old={normalized(json.loads(l)["question"])
             for l in HOLDOUT.read_text(encoding="utf8").splitlines()}
        self.assertFalse(old.intersection({normalized(x["question"]) for x in rows}))

if __name__=="__main__":
    unittest.main()
