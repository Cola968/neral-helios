"""CPU RED/GREEN safety and reproducibility contracts for NERAL logic weight pilot."""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
import random
import unittest

from scripts.neral_v7_logic_pilot import (
    STATUS,TRAIN_FAMILIES,TRANSFER_FAMILIES,AUTH,SEEDS,COUNTS,
    cases,independent_answer,build_logic_experiment,evaluation_cases,_bytes
)


class LogicPilotContracts(unittest.TestCase):
    def test_exact_560_replay_240_logic_train_and_128_val(self):
        train,val,meta=build_logic_experiment()
        self.assertEqual((len(train),len(val)),(800,128))
        self.assertEqual(meta["source_counts"],{"original_replay":560,"new_logic_verified":240})
        self.assertEqual(meta["status"],STATUS)
        self.assertFalse(meta["training_performed"])
        self.assertFalse(meta["release_validated"])
        self.assertFalse(meta["adapter_promoted"])
        self.assertFalse(meta["claimed_score_gain"])
        self.assertEqual(meta["steps"],120)
        self.assertEqual(meta["learning_rate"],3e-6)
        self.assertEqual(meta["train_sha256"],sha256(_bytes(train)).hexdigest())
        self.assertEqual(meta["validation_sha256"],sha256(_bytes(val)).hexdigest())

    def test_all_new_train_val_and_eval_oracles_independently_replay(self):
        for split in SEEDS:
            expected=len((TRANSFER_FAMILIES if split=="transfer" else TRAIN_FAMILIES))*COUNTS[split]
            subset=cases(split)
            self.assertEqual(len(subset),expected)
            for case in subset:
                self.assertEqual(independent_answer(case),int(case["answer"]))
                self.assertLess(len(case["check"]),2500)
                self.assertLess(len(case["prompt"]),2500)
                self.assertEqual(case["domain"],"reasoning")

    def test_wrong_gold_detectable_without_model(self):
        row=deepcopy(cases("transfer")[0])
        row["answer"]=str(int(row["answer"])+1)
        self.assertNotEqual(independent_answer(row),int(row["answer"]))

    def test_train_val_retention_transfer_are_exactly_disjoint(self):
        train,val,_=build_logic_experiment()
        prompts=[r["messages"][1]["content"] for r in train+val]
        used=cases("retention")+cases("transfer")
        prompts.extend(c["prompt"] for c in used)
        self.assertEqual(len(prompts),len(set(prompts)))
        self.assertEqual(len({r["id"] for r in used}),72)
        self.assertEqual(len({r["id"] for r in evaluation_cases()}),72)
        self.assertFalse(set(TRAIN_FAMILIES)&set(TRANSFER_FAMILIES))
        self.assertEqual(len([r for r in used if r["part"]=="retention"]),24)
        self.assertEqual(len([r for r in used if r["part"]=="transfer"]),48)

    def test_deterministic_generation_and_no_untrusted_data_download(self):
        a=cases("train")
        b=cases("train")
        self.assertEqual(_bytes(a),_bytes(b))
        self.assertNotIn("https://",str(a))
        self.assertNotIn("http://",str(a))
        self.assertNotIn("import ",str(a))

    def test_reject_bad_splits_and_coordinates(self):
        for bad in ("future","final_holdout","dev72","../secrets"):
            with self.assertRaises(ValueError):
                cases(bad)
        from scripts.neral_v7_logic_pilot import _row
        with self.assertRaises(ValueError):
            _row("xor_many",0,"transfer")
        with self.assertRaises(ValueError):
            _row("dag_path_count",0,"train")
        with self.assertRaises(ValueError):
            _row("xor_many",9999,"train")

    def test_no_automatic_promotion_or_training_without_authorization(self):
        train,val,meta=build_logic_experiment()
        self.assertTrue(AUTH.startswith("TRAIN_NERAL_V7_LOGIC"))
        self.assertFalse(meta["training_allowed_without_manual_gpu_optin"])
        self.assertFalse(meta["independent_transfer_proven"])

    def test_frozen_eval_cannot_be_incorporated_into_sft(self):
        train,val,_=build_logic_experiment()
        training_prompts={x["messages"][1]["content"] for x in train+val}
        self.assertTrue(all(r["prompt"] not in training_prompts
                            for r in evaluation_cases()))
        self.assertEqual(set(x["family"] for x in cases("transfer")),
                         set(TRANSFER_FAMILIES))


if __name__=="__main__":
    unittest.main()
