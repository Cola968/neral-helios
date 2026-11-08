"""NERAL v8: CPU contracts before any manually approved RTX4070 update.

This proves source properties, NOT improvement of learned model weights.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
import unittest

from scripts.neral_v8_selfverify_data import (
    V8_AUTH, V8_STATUS, V8_FAMILIES, V8_SEEDS, TRAIN_PER_FAMILY,
    VAL_PER_FAMILY, build_selfverify_experiment, new_logic_case,
    validate_new_logic_case, _as_bytes,
)
from scripts.neral_v8_selfverify_train import MODEL_STATUS
from scripts.neral_training_checkpoint_policy import options_for_training

class V8DataContracts(unittest.TestCase):
    def test_training_and_validation_exact_rows_and_hashes(self):
        train, val, meta=build_selfverify_experiment()
        self.assertEqual((len(train),len(val)),(960,128))
        self.assertEqual(meta["source_counts"],{
            "verified_v5_v6_replay":560,
            "independently_checked_v7_direct":240,
            "independently_checked_new_families":160,
        })
        self.assertEqual(meta["status"],V8_STATUS)
        self.assertEqual(meta["steps"],240)
        self.assertEqual(meta["optimizer_sample_visits"],960)
        self.assertEqual(meta["learning_rate"],4e-6)
        self.assertEqual(meta["train_sha256"],sha256(_as_bytes(train)).hexdigest())
        self.assertEqual(meta["validation_sha256"],sha256(_as_bytes(val)).hexdigest())
        self.assertFalse(meta["training_performed"])
        self.assertFalse(meta["claimed_score_gain"])
        self.assertFalse(meta["independent_transfer_proven"])
        self.assertFalse(meta["adapter_promoted"])
        self.assertFalse(meta["release_validated"])

    def test_every_new_gold_recomputed_by_independent_oracle(self):
        for split, count in (("train",TRAIN_PER_FAMILY),("val",VAL_PER_FAMILY)):
            for family in V8_FAMILIES:
                for index in range(count):
                    with self.subTest(split=split,family=family,index=index):
                        row=new_logic_case(family,split,index)
                        self.assertTrue(validate_new_logic_case(row))
                        self.assertEqual(row["split"],split)

    def test_exactly_100_wrong_claim_corrections(self):
        train,_,meta=build_selfverify_experiment()
        claimed=[row for row in train if "Proposed answer: " in row["messages"][1]["content"]]
        self.assertEqual(len(claimed),100)
        self.assertEqual(meta["self_verification_wrong_proposal_examples"],100)
        for row in claimed:
            self.assertIn("VERIFY:",row["messages"][-1]["content"])
            self.assertIn("FINAL_ANSWER: ",row["messages"][-1]["content"])
            self.assertIn("does not satisfy this check",row["messages"][-1]["content"])

    def test_train_val_disjoint_and_no_consumed_v7_transfer_prompts(self):
        train,val,_=build_selfverify_experiment()
        users=[r["messages"][1]["content"] for r in train+val]
        self.assertEqual(len(set(users)),1088)
        from scripts.neral_v7_logic_pilot import cases
        used={r["prompt"] for r in cases("transfer")}
        self.assertTrue(used.isdisjoint(users))
        for item in train+val:
            self.assertEqual([r["role"] for r in item["messages"]],
                             ["system","user","assistant"])

    def test_train_data_deterministic_and_known_good(self):
        a,b,meta=build_selfverify_experiment()
        aa,bb,other=build_selfverify_experiment()
        self.assertEqual(_as_bytes(a),_as_bytes(aa))
        self.assertEqual(_as_bytes(b),_as_bytes(bb))
        self.assertEqual(meta["train_sha256"],other["train_sha256"])

    def test_reject_mutated_gold(self):
        data=new_logic_case("modular_inverse","train",1)
        self.assertTrue(validate_new_logic_case(data))
        data["answer"]=str(int(data["answer"])+1)
        self.assertFalse(validate_new_logic_case(data))
        with self.assertRaises(ValueError):
            new_logic_case("unknown","train",0)
        with self.assertRaises(ValueError):
            new_logic_case(V8_FAMILIES[0],"transfer",0)

    def test_manual_boundaries_checkpoint_best_validation_and_no_promotion(self):
        self.assertEqual(V8_AUTH,"TRAIN_NERAL_V8_SELFVERIFY_R16_240_EXPERIMENTAL")
        self.assertEqual(MODEL_STATUS,"NERAL_V8_SELFVERIFY_R16_240_UNRELEASED")
        opts=options_for_training(steps=240,interval=40,max_steps_limit=240)
        self.assertEqual(opts["eval_steps"],40)
        self.assertEqual(opts["save_steps"],40)
        self.assertTrue(opts["load_best_model_at_end"])
        self.assertEqual(opts["metric_for_best_model"],"eval_loss")
        with self.assertRaises(ValueError):
            options_for_training(steps=241,interval=40,max_steps_limit=240)

if __name__=="__main__":
    unittest.main()
