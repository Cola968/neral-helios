"""RED/GREEN: NERAL v8 SFT reasoning verification + retention + GPU bounds."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import unittest

from scripts.neral_v8_selfverify_data import (
    build_selfverify_experiment, new_logic_case, validate_new_logic_case,
    V8_FAMILIES, V8_SEEDS, V8_AUTH, V8_STATUS,
)
from scripts.neral_training_checkpoint_policy import options_for_training


class V8SelfVerifyContracts(unittest.TestCase):
    def test_training_runs_one_nominal_epoch_not_six_tenths(self):
        train,val,meta=build_selfverify_experiment()
        self.assertEqual((len(train),len(val)),(960,128))
        self.assertEqual(meta["steps"],240)
        self.assertEqual(meta["gradient_accumulation"],4)
        self.assertEqual(meta["optimizer_sample_visits"],960)
        self.assertEqual(meta["source_counts"],{
            "verified_v5_v6_replay":560,
            "independently_checked_v7_direct":240,
            "independently_checked_new_families":160,
        })
        self.assertFalse(meta["training_performed"])
        self.assertFalse(meta["release_validated"])
        self.assertFalse(meta["adapter_promoted"])
        self.assertFalse(meta["independent_transfer_proven"])
        self.assertFalse(meta["claimed_score_gain"])
        self.assertEqual(meta["status"],V8_STATUS)

    def test_all_new_evidence_can_be_recomputed_from_numeric_params(self):
        for split in V8_SEEDS:
            for family in V8_FAMILIES:
                for index in range(40 if split=="train" else 8):
                    c=new_logic_case(family,split,index)
                    self.assertTrue(validate_new_logic_case(c))
                    self.assertIn("CHECK:",c["check"])
                    self.assertIn("VERIFY:",c["check"])
                    self.assertLessEqual(len(c["prompt"]),2200)
                    self.assertLessEqual(len(c["check"]),1500)

    def test_faulty_gold_is_not_silently_accepted(self):
        import copy
        for family in V8_FAMILIES:
            c=copy.deepcopy(new_logic_case(family,"train",1))
            c["answer"]=str(int(c["answer"])+1)
            self.assertFalse(validate_new_logic_case(c))

    def test_train_validation_and_transfer_source_separation(self):
        from scripts.neral_v7_logic_pilot import cases
        train,val,meta=build_selfverify_experiment()
        prompts=[r["messages"][1]["content"] for r in train+val]
        self.assertEqual(len(prompts),len(set(prompts)))
        held=set(r["prompt"] for r in cases("transfer"))
        self.assertTrue(held.isdisjoint(prompts))
        self.assertEqual(meta["v7_transfer_development_cases_excluded"],48)
        self.assertEqual(len(meta["train_sha256"]),64)
        self.assertEqual(len(meta["validation_sha256"]),64)
        self.assertEqual(
            sha256(("".join(json.dumps(r,sort_keys=True,
                      ensure_ascii=False,separators=(",",":"))+"\n"
                      for r in train)).encode()).hexdigest(),
            meta["train_sha256"],
        )

    def test_correction_tasks_actually_contain_an_explicit_wrong_claim(self):
        train,val,meta=build_selfverify_experiment()
        corrected=[r for r in train if
                  "Proposed answer: " in r["messages"][1]["content"]]
        self.assertEqual(len(corrected),100)
        for example in corrected:
            system,user,assistant=example["messages"]
            proposal=user["content"].rsplit("Proposed answer: ",1)[-1].split(".",1)[0]
            final=assistant["content"].split("FINAL_ANSWER: ")[-1].strip()
            self.assertNotEqual(proposal,final)
            self.assertTrue(assistant["content"].startswith("CHECK:"))
            self.assertIn("VERIFY:",assistant["content"])
            self.assertEqual(assistant["content"].count("FINAL_ANSWER:"),1)

    def test_families_span_multiple_reasoning_methods(self):
        self.assertEqual(set(V8_FAMILIES),
                         {"modular_inverse","euclidean_gcd",
                          "coin_change_combinations","weighted_dag_shortest"})
        c=Counter(new_logic_case(f,"train",j)["family"]
                  for f in V8_FAMILIES for j in range(40))
        self.assertTrue(all(n==40 for n in c.values()))

    def test_prior_120_step_default_remains_strict(self):
        with self.assertRaises(ValueError):
            options_for_training(steps=240,interval=40)
        opt=options_for_training(steps=240,interval=40,max_steps_limit=240)
        self.assertEqual(opt["save_steps"],40)
        self.assertTrue(opt["load_best_model_at_end"])
        for limit in (0,121,239,241,300,320):
            with self.assertRaises(ValueError):
                options_for_training(steps=240,interval=40,max_steps_limit=limit)
        with self.assertRaises(ValueError):
            options_for_training(steps=241,interval=40,max_steps_limit=240)
        self.assertEqual(V8_AUTH,"TRAIN_NERAL_V8_SELFVERIFY_R16_240_EXPERIMENTAL")

    def test_deterministic_without_network_or_model_calls(self):
        x=build_selfverify_experiment()
        y=build_selfverify_experiment()
        self.assertEqual(x,y)
        for r in x[0][:15]:
            self.assertEqual([p["role"] for p in r["messages"]],
                             ["system","user","assistant"])
            self.assertTrue(r["messages"][-1]["content"])
        self.assertEqual(x[2]["base_adapter"],"v5-r16-480")


if __name__=="__main__":
    unittest.main()
