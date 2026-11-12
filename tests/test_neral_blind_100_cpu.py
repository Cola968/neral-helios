"""Blind100 red/green oracle, leakage, exact API paired scoring and hash evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.neral_blind_100_cases import build,serialize,exclusion
from scripts.neral_blind_100_gpu import score_pairs,parse_answer
from scripts.neral_qlora_dataset import compact,generate,load_holdout

ROOT=Path(__file__).resolve().parents[1]
HOLDOUT=ROOT/"datasets/neral_qwen35_holdout_v1.jsonl"

class FrozenBlindCases(unittest.TestCase):
    def test_five_new_families_per_group_and_exact_100_unique(self):
        a=build()
        self.assertEqual(len(a),100)
        self.assertEqual(a,build())
        self.assertEqual(len({x["question_sha256"] for x in a}),100)
        for cat in ("mathematics","logic","programming","mixed"):
            sub=[x for x in a if x["category"]==cat]
            self.assertEqual(len(sub),25)
            self.assertEqual(len({x["family"] for x in sub}),1 if cat=="programming" else 5)
            for x in sub:
                self.assertEqual(hashlib.sha256(x["question"].encode("utf8")).hexdigest(),x["question_sha256"])
                self.assertIs(type(x["gold_integer"]),int)
                self.assertLess(abs(x["gold_integer"]),10**10)
        self.assertEqual(build(),[json.loads(s) for s in serialize(a).decode().splitlines()])
    def test_disjoint_against_exact_prior_train_24_and_100(self):
        excluded,_=load_holdout(HOLDOUT)
        train=generate(pairs=360,exclusion_questions=excluded)
        blob="".join(compact(x)+"\n" for x in train)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"train.jsonl"
            path.write_bytes(blob.encode("utf8"))
            receipt=exclusion(build(),path,HOLDOUT)
            self.assertEqual(receipt["count"],100)
            self.assertEqual(receipt["categories"],{
                "mathematics":25,"logic":25,"programming":25,"mixed":25})
            self.assertEqual(receipt["blind100_sha256"],hashlib.sha256(serialize(build())).hexdigest())
            mutated=build()
            from scripts.neral_generalization_100_cases import build as previous
            mutated[0]={**mutated[0],
                "question":previous()[0]["question"],
                "question_sha256":previous()[0]["question_sha256"]}
            with self.assertRaisesRegex(ValueError,"leakage"):
                exclusion(mutated,path,HOLDOUT)
            mutated2=build()
            question=train[0]["messages"][1]["content"]
            mutated2[0]={**mutated2[0],"question":question,
                         "question_sha256":hashlib.sha256(question.encode()).hexdigest()}
            with self.assertRaisesRegex(ValueError,"leakage"):
                exclusion(mutated2,path,HOLDOUT)
    def test_score_strict_integer_full_pairs_only(self):
        cases=build()
        results=[{"id":x["id"],"category":x["category"],
                  "base_correct":i%4==0,
                  "dual_correct":i%4==0 or i in (1,2,3,4,5,6,7,8),
                  "policy":"raw_qwen_already_selected",
                  "base_seconds":1.2,"dual_seconds":2.4} for i,x in enumerate(cases)]
        score=score_pairs(cases,results)
        self.assertEqual(score["base_correct"],25)
        self.assertEqual(score["dual_correct"],31)
        self.assertEqual(score["gained"],6)
        self.assertEqual(score["lost"],0)
        self.assertFalse(score["promotion_gate_passed"])
        self.assertEqual(score["by_category"]["mathematics"]["total"],25)
        self.assertIsNone(parse_answer("I think 42"))
        self.assertEqual(parse_answer("FINAL_ANSWER: -23"),-23)
        with self.assertRaises(ValueError):
            score_pairs(cases,results[:99])
if __name__=="__main__":
    unittest.main()
