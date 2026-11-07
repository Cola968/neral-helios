"""NERAL v6 experimental v5-continuation: CPU data and release-denial contracts."""
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from scripts.neral_v6_r16_gpu import (
    build_experiment, training_contract, require_explicit_gpu_authorization,
    tokenized_completion, _compact_trace, VERIFIED_STEPS_AUTH, AUTH,
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
        self.assertNotIn("counterfactual_target", cfg)

    def test_verified_steps_restore_arithmetic_and_keep_answers(self):
        compact_train, compact_val, compact_meta = build_experiment()
        full_train, full_val, full_meta = build_experiment(
            counterfactual_target="verified_steps")
        self.assertEqual(
            [row["messages"][1]["content"] for row in compact_train],
            [row["messages"][1]["content"] for row in full_train])
        self.assertEqual(compact_val, full_val)
        self.assertEqual(compact_meta["learning_rate"], full_meta["learning_rate"])
        self.assertEqual(compact_meta["steps"], full_meta["steps"])
        self.assertEqual(compact_meta["train_rows"], full_meta["train_rows"])
        self.assertNotEqual(compact_meta["train_sha256"], full_meta["train_sha256"])
        self.assertFalse(full_meta["claimed_score_gain"])
        self.assertFalse(full_meta["training_performed"])
        changed = 0
        for old, new in zip(compact_train, full_train):
            old_text = old["messages"][2]["content"]
            new_text = new["messages"][2]["content"]
            self.assertEqual(
                old_text.split("FINAL_ANSWER:")[-1].strip(),
                new_text.split("FINAL_ANSWER:")[-1].strip())
            if old_text != new_text:
                changed += 1
                self.assertNotIn("*", old_text)
                self.assertIn("*", new_text)
                self.assertIn("+", new_text)
        self.assertEqual(changed, 128)
        self.assertEqual((full_train, full_val, full_meta), build_experiment(
            counterfactual_target="verified_steps"))

    def test_verified_step_permit_is_not_the_compact_permit(self):
        import contextlib
        import io
        from scripts.neral_v6_r16_gpu import main
        for target, permit in (
                ("verified_steps", AUTH),
                ("compact", VERIFIED_STEPS_AUTH)):
            with self.subTest(target=target):
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(io.StringIO()):
                        main(["--mode","train-local-gpu","--out","/tmp/unused-neral-out",
                              "--base","/tmp","--v5-adapter","/tmp",
                              "--counterfactual-target", target,
                              "--authorize", permit])
        self.assertNotEqual(AUTH, VERIFIED_STEPS_AUTH)

    def test_phase2_gate_blocks_historical_protocol_before_any_training(self):
        from scripts.neral_v6_fresh_dev_eval import phase2_training_gate
        counts={"strict_correct":70,"wrong_readable_number":1,"unreadable":1,
                "incomplete":0,"correct":70,"ungraded_completed":1,"total":72}
        report={"prompt_protocol":"historical","release_validated":False,
                "adapter_promoted":False,"weights_promoted":False,
                "claimed_score_gain":False,"general_improvement_proven":False,
                "total":72,"scores":{name:dict(counts) for name in
                                     ("qwen_fewshot","neral_v5","neral_v6")}}
        gate=phase2_training_gate(report)
        self.assertEqual(gate["decision"],"refuse")
        self.assertIn("historical_protocol_is_not_phase2_evidence",gate["reasons"])
        self.assertFalse(gate["training_authorized"])
        self.assertFalse(gate["claimed_score_gain"])
        matched=phase2_training_gate(dict(report,prompt_protocol="matched_fewshot"))
        self.assertEqual(matched["decision"],"ok")
        self.assertFalse(matched["claimed_capability_gain"])

    def test_verified_step_workflow_cannot_start_from_a_pull_request(self):
        workflow = Path(__file__).resolve().parents[1].joinpath(
            ".github","workflows","neral-v6-verified-steps-train.yml"
        ).read_text(encoding="utf8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn(VERIFIED_STEPS_AUTH, workflow)
        self.assertIn("--counterfactual-target verified_steps", workflow)
        self.assertIn("v6-verified-steps-120-", workflow)
        self.assertNotIn("v6-r16-120-$env:GITHUB_RUN_ID", workflow)
        self.assertNotIn("--authorize " + AUTH, workflow)


if __name__=="__main__":
    unittest.main()
