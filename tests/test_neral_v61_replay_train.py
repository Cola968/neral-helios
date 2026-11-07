"""NERAL v6.1 replay regularization: data and GPU contract tests (CPU)."""
import json
import inspect
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v61_replay_train import (
    build_replay_experiment, materialize_replay, AUTH,
    LEARNING_RATE, TRAIN_STEPS, V5_COUNTS, V6_COUNTS,
)
from scripts.neral_v6_r16_gpu import _gpu_train


class V61ReplayTests(unittest.TestCase):
    def test_source_replay_is_both_diverse_and_train_only(self):
        train, val, meta=build_replay_experiment()
        self.assertEqual(len(train),800)
        self.assertEqual(len(val),140)
        self.assertEqual(meta["source_counts"],{"v5_train_replay":560,"v6_train_multistep":240})
        self.assertEqual(meta["v5_families"],V5_COUNTS)
        self.assertEqual(meta["v6_families"],V6_COUNTS)
        self.assertTrue(meta["strict_holdout_exclusion"])
        self.assertFalse(meta["release_validated"])
        self.assertFalse(meta["training_performed"])
        self.assertFalse(meta["actual_generalization_improved"])
        self.assertEqual(meta["steps"],120)
        self.assertEqual(meta["learning_rate"],2e-6)
        self.assertEqual(len({r["messages"][1]["content"] for r in train+val}),940)
        self.assertEqual(len({r["messages"][2]["content"] for r in train}),len({r["messages"][2]["content"] for r in train}))
        self.assertEqual((train,val,meta),build_replay_experiment())
        self.assertTrue(all([m["role"] for m in row["messages"]]==["system","user","assistant"] for row in train+val))
        self.assertTrue(all(not row["messages"][1]["content"].endswith("FINAL_ANSWER: 987654") for row in train))
        
    def test_72_case_consumed_development_suitedoes_not_enter_training(self):
        from scripts.neral_v6_fresh_dev_eval import build_cases, check_source_isolation
        train, val, meta=build_replay_experiment()
        self.assertTrue(check_source_isolation(build_cases()))
        from scripts.neral_decontamination import normalize,shingles
        tasks=build_cases()
        controls=[row["messages"][1]["content"] for row in train+val]
        control_set={normalize(x) for x in controls}
        grams=set().union(*(shingles(x,n=10) for x in controls))
        for case in tasks:
            self.assertNotIn(normalize(case["prompt"]),control_set)
            self.assertFalse(grams.intersection(shingles(case["prompt"],n=10)))

    def test_old_trainer_default_is_unchanged_and_new_call_is_configurable(self):
        spec=inspect.signature(_gpu_train)
        self.assertEqual(spec.parameters["steps"].default,120)
        self.assertEqual(spec.parameters["lr"].default,1e-5)
        self.assertEqual(spec.parameters["authorize"].default,"TRAIN_NERAL_V6_R16_120_EXPERIMENTAL")
        self.assertEqual(TRAIN_STEPS,120)
        self.assertEqual(LEARNING_RATE,2e-6)
        self.assertNotEqual(AUTH,spec.parameters["authorize"].default)
        source=inspect.getsource(_gpu_train)
        self.assertIn("max_steps=steps",source)
        self.assertIn("learning_rate=lr",source)
        self.assertIn("data_factory()",source)

    def test_private_materialization_fails_closed_on_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"run"
            manifest=materialize_replay(out)
            self.assertEqual(set(x.name for x in out.iterdir()),
                             {"train.jsonl","validation.jsonl","manifest.json"})
            self.assertEqual(manifest["train_rows"],800)
            self.assertFalse(manifest["release_validated"])
            with self.assertRaises(ValueError):
                materialize_replay(out)


if __name__=="__main__":
    unittest.main()
