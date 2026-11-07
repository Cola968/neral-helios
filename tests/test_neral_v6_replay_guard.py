"""CPU-only regression contracts for verified train-only anti-forgetting replay."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v6_replay_guard import build_candidate, export_candidate, ReplayRejected
from scripts.neral_v6_compositional import make_case
from scripts.neral_v6_r16_gpu import build_experiment
from scripts.neral_v5_transfer_curriculum import _generated

def fixture():
    v6, _, _ = build_experiment()
    v5 = [json.loads(s) for s in _generated(20261108)["train"].decode("utf8").splitlines()]
    pairs=[]
    for i in range(10):
        family=("money_order","inventory_flow")[i % 2]
        index=i//2
        case=make_case(family,"train",index)
        expected=case["messages"][2]["content"]
        wrong="999999,99 EUR" if family=="money_order" else "999999 Stück"
        pairs.append({
          "id":f"v6-{family}-train-{index:03}",
          "source_id":f"v6-original-{family}-train-{index:03}",
          "family":family, "split":"train",
          "prompt":case["messages"][1]["content"],"chosen":expected,
          "rejected":wrong,"chosen_verified":True,
          "rejected_verified":False,"verifier_type":"numeric_oracle",
          "verifier_sha256":"c"*64,
        })
    return v6,v5,pairs


class ReplayGuardTests(unittest.TestCase):
    def test_real_generator_inputs_produce_balanced_train_only(self):
        v6,v5,pairs=fixture()
        records,report=build_candidate(v6,v5,pairs,protected_prompts=["a genuinely separate task"])
        self.assertEqual(len(records),50)
        self.assertEqual(report["source_counts"],{"v6_clean":25,"v5_replay":15,"verified_correction":10})
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["general_improvement_proven"])

    def test_deterministic_order_and_reproducible_hash(self):
        args=fixture()
        a,pa=build_candidate(*args,protected_prompts=["unrelated holdout"],seed=9)
        b,pb=build_candidate(*args,protected_prompts=["unrelated holdout"],seed=9)
        self.assertEqual(a,b)
        self.assertEqual(pa["candidate_sha256"],pb["candidate_sha256"])

    def test_seed_changes_sample_not_labels(self):
        args=fixture()
        a,_=build_candidate(*args,protected_prompts=["private out of scope"],seed=9)
        b,_=build_candidate(*args,protected_prompts=["private out of scope"],seed=10)
        self.assertNotEqual(a,b)

    def test_candidate_entries_have_only_messages_and_provenance_not_wrong_as_target(self):
        rows,_=build_candidate(*fixture(),protected_prompts=["external holdout"])
        for item in rows:
            self.assertEqual([m["role"] for m in item["messages"]],["system","user","assistant"])
            self.assertNotIn("999999",item["messages"][-1]["content"])
            self.assertIn(item["source"],("v6_clean","v5_replay","verified_correction"))

    def test_no_pairs_is_not_invented(self):
        v6,v5,_=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,[],protected_prompts=["external holdout"])

    def test_validation_pairs_rejected(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,[{**pairs[0],"split":"validation"}]+pairs[1:],
                            protected_prompts=["external holdout"])

    def test_fake_verified_label_rejected(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,[{**pairs[0],"chosen_verified":False}]+pairs[1:],
                            protected_prompts=["external holdout"])

    def test_fabricated_rejected_answer_not_accepted(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,[{**pairs[0],"rejected":"I don't know"}]+pairs[1:],
                            protected_prompts=["external holdout"])

    def test_correct_rejected_answer_not_accepted(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,[{**pairs[0],"rejected":pairs[0]["chosen"]}]+pairs[1:],
                            protected_prompts=["external holdout"])

    def test_changed_chosen_solution_rejected(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,[{**pairs[0],"chosen":"123 EUR"}]+pairs[1:],
                            protected_prompts=["external holdout"])

    def test_duplicate_pair_source_rejected(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,pairs+[pairs[0]],protected_prompts=["external holdout"])

    def test_holdout_exact_prompt_blocks_export(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,pairs,protected_prompts=[pairs[0]["prompt"]])

    def test_holdout_near_duplicate_10gram_blocks(self):
        v6,v5,pairs=fixture()
        with self.assertRaises(ReplayRejected):
            build_candidate(v6,v5,pairs,
              protected_prompts=["IMPORTANT " + pairs[0]["prompt"] + " nothing else"])

    def test_missing_holdout_is_fail_closed(self):
        with self.assertRaises(ReplayRejected):
            build_candidate(*fixture(),protected_prompts=[])

    def test_private_create_only_no_overwrite(self):
        rows,report=build_candidate(*fixture(),protected_prompts=["new task"])
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/"train_only_candidate"
            saved=export_candidate(dest,rows,report)
            self.assertEqual(len((dest/"candidate.jsonl").read_text("utf8").splitlines()),50)
            self.assertEqual(saved["candidate_sha256"],hashlib.sha256(
                (dest/"candidate.jsonl").read_bytes()).hexdigest())
            with self.assertRaises(ReplayRejected):
                export_candidate(dest,rows,report)

    def test_no_protected_prompt_is_written_to_output(self):
        protected="private protected user question about 109 and 203 units"
        rows,report=build_candidate(*fixture(),protected_prompts=[protected])
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"candidate"
            export_candidate(dest,rows,report)
            for p in dest.iterdir():
                self.assertNotIn(protected,p.read_text("utf8"))
            self.assertNotIn("trained_checkpoint_sha256",report)

if __name__=="__main__":
    unittest.main()
