"""NERAL v6 experimental v5-continuation: CPU data and release-denial contracts."""
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from scripts.neral_v6_r16_gpu import (
    build_experiment, training_contract, require_explicit_gpu_authorization,
    tokenized_completion, _compact_trace,
)
from scripts.neral_reasoning_train_candidate import candidate, verify_case


class V6ExperimentContracts(unittest.TestCase):
    def test_verified_train_replay_and_no_holdout_entry(self):
        train, val, meta = build_experiment()
        self.assertEqual(len(train), 728)
        self.assertEqual(len(val), 100)
        self.assertEqual(meta["source_counts"], {"v6_composition":600,"reasoning_counterfactual":128})
        self.assertFalse(meta["training_performed"])
        self.assertFalse(meta["release_validated"])
        self.assertFalse(meta["independent_transfer_proven"])
        self.assertFalse(meta["human_semantic_review_performed"])
        self.assertEqual(len({x["messages"][1]["content"] for x in train+val}), 828)
        self.assertTrue(all(list(z.keys())==["messages"] for z in train+val))
        self.assertEqual(meta["holdout_excluded"],100)
        self.assertEqual(len(meta["train_sha256"]),64)
        self.assertEqual((train,val,meta), build_experiment())

    def test_counterfactual_trace_not_unverified_or_made_up(self):
        rows, proofs = candidate()
        self.assertEqual(len(rows),128)
        for row, proof in zip(rows,proofs):
            self.assertTrue(verify_case(proof,row))
            compact = _compact_trace(proof,row)
            self.assertIn("CHECK:", compact)
            self.assertIn("FINAL_ANSWER:", compact)
            self.assertEqual(compact.count("FINAL_ANSWER:"),1)
            self.assertIn(proof["answer"],compact)
            self.assertNotIn("???",compact)

    def test_exact_completion_boundary_and_truncation_rejected(self):
        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                t=[]
                for item in messages:
                    t+=list(item["role"].encode())+list(item["content"].encode())+[0]
                if kwargs.get("add_generation_prompt"):
                    t+=[255]  # deliberately wrong assistant-token prefix
                # To match real chat template, completion removes the artificial
                # end-of-message preceding the assistant generation prompt.
                if not kwargs.get("add_generation_prompt"):
                    pos=len(t)
                    return t[:pos] if not kwargs.get("return_tensors") else None
                return t
        # Bad tokenizer alignment must fail, never silently truncate.
        example=build_experiment()[0][0]
        with self.assertRaises(ValueError):
            tokenized_completion(example,Tok(),chat_template="fixed",max_length=512)

    def test_dry_run_materializes_new_private_train_and_val_only(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"new"
            report=training_contract(dest)
            self.assertEqual({x.name for x in dest.iterdir()},
              {"train.jsonl","validation.jsonl","manifest.json"})
            self.assertFalse(report["training_allowed_without_manual_gpu_optin"])
            self.assertFalse(report["human_semantic_review_performed"])
            self.assertFalse(report["release_validated"])
            self.assertNotIn("holdout.jsonl",str(list(dest.iterdir())))
            with self.assertRaises(ValueError):
                training_contract(dest)

    def test_output_dir_is_reserved_before_hf_trainer_may_autocreate_subdirectory(self):
        # Real Windows run 42407216024 failed because TrainingArguments/Trainer
        # created dest/trainer-temp *before* dest.mkdir(exist_ok=False).
        import inspect
        from scripts.neral_v6_r16_gpu import _gpu_train
        source=inspect.getsource(_gpu_train)
        self.assertIn("dest.mkdir(exist_ok=False)",source)
        self.assertIn("config=TrainingArguments(",source)
        self.assertLess(source.index("dest.mkdir(exist_ok=False)"),
                        source.index("config=TrainingArguments("))
        self.assertEqual(source.count("dest.mkdir(exist_ok=False)"),1)

    def test_manual_120_step_gpu_only_and_strict_identity(self):
        with self.assertRaises(ValueError):
            require_explicit_gpu_authorization("")
        with self.assertRaises(ValueError):
            require_explicit_gpu_authorization("TRAIN_NERAL_V5_R16_480_TRANSFER")
        self.assertTrue(require_explicit_gpu_authorization("TRAIN_NERAL_V6_R16_120_EXPERIMENTAL"))
        cfg=build_experiment()[2]
        self.assertEqual(cfg["steps"],120)
        self.assertEqual(cfg["max_length"],768)
        self.assertEqual(cfg["lora_rank"],16)
        self.assertEqual(cfg["adapter_lineage"],"v5-r16-480")
        self.assertEqual(cfg["optimizer"],"paged_adamw_8bit")


if __name__=="__main__":
    unittest.main()
