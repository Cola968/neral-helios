"""NERAL v6 experimental v5-continuation: CPU data and release-denial contracts."""
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.neral_v6_r16_gpu import (
    build_experiment, training_contract, require_explicit_gpu_authorization,
    tokenized_completion, _compact_trace, VERIFIED_STEPS_AUTH, AUTH,
    assert_verified_steps_fit_max_length, prove_qwen_completion_bound,
    _assert_arithmetic_preserved, _fit_verified_assistant,
    load_pinned_qwen_tokenizer, audit_verified_step_token_lengths,
    PINNED_TOKENIZER_SHA256,
)
from scripts.neral_reasoning_train_candidate import candidate, verify_case


def bound_verified_experiment():
    """Bound-path corpus. A local Qwen tokenizer must not change this result."""
    return build_experiment(
        counterfactual_target="verified_steps", length_source="bound")


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
        full_train, full_val, full_meta = bound_verified_experiment()
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
        self.assertEqual((full_train, full_val, full_meta), bound_verified_experiment())
        self.assertEqual(full_meta["length_preflight"]["length_source"], "bound")

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

    def test_verified_steps_byte_bound_fits_and_records_the_prior_overflow(self):
        train, val, meta = bound_verified_experiment()
        report = assert_verified_steps_fit_max_length(train + val)
        self.assertEqual(report["method"], "qwen2_byte_bpe_upper_bound")
        self.assertFalse(report["offline_qwen_tokenizer"])
        self.assertEqual(report["rows"], 828)
        self.assertEqual(report["rows_over_max_length"], 0)
        self.assertLessEqual(report["max_upper_bound_tokens"], 768)
        self.assertGreaterEqual(report["min_assistant_pieces"], 2)
        self.assertFalse(report["claimed_score_gain"])
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["training_authorized"])
        stored = dict(meta["length_preflight"])
        before = stored.pop("before_boilerplate_trim")
        self.assertEqual(report, stored)
        self.assertEqual(before["counterfactual_rows"], 128)
        self.assertEqual(before["rows_over_max_length"], 70)
        self.assertEqual(before["max_upper_bound_tokens"], 933)
        self.assertEqual(before["by_family"], {
            "ordered_price": 6, "segmented_fuel": 64})
        self.assertEqual(before["length_method"], "qwen2_byte_bpe_upper_bound")
        self.assertFalse(before["tokenizer_pinned"])
        self.assertEqual(before["boilerplate_trim_levels"], {
            "verified_text": 58,
            "drop_rounding_sentence": 6,
            "drop_intermediate_units": 0,
            "drop_node_labels": 6,
            "drop_parentheses": 42,
            "squeeze_operator_spaces": 16,
        })
        self.assertEqual(report["max_upper_bound_tokens"], 768)
        self.assertEqual(report["rows_at_max_length"], 2)
        rows, proofs = candidate()
        over = {}
        highest = 0
        for row, proof in zip(rows, proofs):
            bound = prove_qwen_completion_bound(row)["upper_bound_tokens"]
            highest = max(highest, bound)
            if bound > 768:
                over[proof["family"]] = over.get(proof["family"], 0) + 1
        self.assertEqual(highest, 933)
        self.assertEqual(over, {"ordered_price": 6, "segmented_fuel": 64})
        with self.assertRaises(ValueError) as caught:
            assert_verified_steps_fit_max_length(rows)
        self.assertIn("70 row(s)", str(caught.exception))
        self.assertIn("933", str(caught.exception))

    def test_verified_trim_keeps_numbers_operators_and_finals(self):
        import re
        rows, _proofs = candidate()
        train, _val, meta = bound_verified_experiment()
        fitted = {row["messages"][1]["content"]: row["messages"][2]["content"]
                  for row in train}
        self.assertFalse(meta["claimed_score_gain"])
        self.assertFalse(meta["release_validated"])
        self.assertNotIn("length_preflight", build_experiment()[2])
        for raw in rows:
            original = raw["messages"][2]["content"]
            new = fitted[raw["messages"][1]["content"]]
            self.assertEqual(
                original.split("FINAL_ANSWER:", 1)[1].strip(),
                new.split("FINAL_ANSWER:", 1)[1].strip())
            self.assertEqual(re.findall(r"\d+", original), re.findall(r"\d+", new))
            self.assertEqual(original.count("*"), new.count("*"))
            self.assertEqual(original.count("+"), new.count("+"))
            self.assertEqual(original.count("-"), new.count("-"))
            self.assertEqual(
                original.replace(" kg/min", "").count("/"),
                new.replace(" kg/min", "").count("/"))
            self.assertIn("*", new)
            self.assertIn("+", new)

    def test_pinned_chat_byte_bound_and_oversized_assistant_fails(self):
        tiny = {"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]}
        proof = prove_qwen_completion_bound(tiny)
        self.assertEqual(proof["upper_bound_tokens"], 34)
        self.assertGreaterEqual(proof["assistant_pieces"], 2)
        huge = {"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "1+" * 2000},
        ]}
        with self.assertRaises(ValueError) as caught:
            assert_verified_steps_fit_max_length([huge])
        self.assertIn("1 row(s)", str(caught.exception))
        self.assertIn("other", str(caught.exception))
        passed = assert_verified_steps_fit_max_length([tiny])
        self.assertLessEqual(passed["max_upper_bound_tokens"], 768)
        self.assertFalse(passed["training_authorized"])

    def test_injected_tokenizer_uses_tokenized_completion_rules(self):
        row = {"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "1+1"},
        ]}

        class Aligned:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("add_generation_prompt"):
                    return [7, 8]
                return [7, 8, 9, 10]

        class Misaligned:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("add_generation_prompt"):
                    return [1, 2, 3]
                return [4, 5, 6, 7]

        class LongerThanBound:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("add_generation_prompt"):
                    return list(range(4))
                return list(range(100))

        class OverBudget:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("add_generation_prompt"):
                    return list(range(4))
                return list(range(900))

        report = assert_verified_steps_fit_max_length([row], tokenizer=Aligned())
        self.assertEqual(report["method"], "offline_qwen_tokenizer")
        self.assertEqual(report["max_upper_bound_tokens"], 4)
        self.assertFalse(report["claimed_score_gain"])
        self.assertFalse(report["training_authorized"])
        with self.assertRaises(ValueError):
            assert_verified_steps_fit_max_length([row], tokenizer=Misaligned())
        with self.assertRaises(ValueError) as under:
            assert_verified_steps_fit_max_length([row], tokenizer=LongerThanBound())
        self.assertIn("under-estimated", str(under.exception))
        with self.assertRaises(ValueError):
            assert_verified_steps_fit_max_length([row], tokenizer=OverBudget())

    def test_verified_contract_materializes_length_preflight_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "verified"
            report = training_contract(
                dest, counterfactual_target="verified_steps", length_source="bound")
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["release_validated"])
            self.assertFalse(report["claimed_score_gain"])
            self.assertEqual(report["length_preflight"]["rows_over_max_length"], 0)
            self.assertLessEqual(report["length_preflight"]["max_upper_bound_tokens"], 768)
            manifest = json.loads((dest / "manifest.json").read_text(encoding="utf8"))
            self.assertEqual(manifest["length_preflight"]["max_length"], 768)
            self.assertFalse(manifest["length_preflight"]["training_authorized"])

    def test_missing_snapshot_env_does_not_download(self):
        import os
        from scripts.neral_v6_r16_gpu import _load_offline_qwen_tokenizer
        old = os.environ.get("NERAL_QWEN_SNAPSHOT")
        os.environ["NERAL_QWEN_SNAPSHOT"] = "/tmp/neral-no-such-qwen-snapshot"
        try:
            with self.assertRaises(ValueError):
                _load_offline_qwen_tokenizer()
        finally:
            if old is None:
                os.environ.pop("NERAL_QWEN_SNAPSHOT", None)
            else:
                os.environ["NERAL_QWEN_SNAPSHOT"] = old
        self.assertFalse(Path("/tmp/neral-no-such-qwen-snapshot").exists())

    def test_bound_length_source_ignores_an_injected_tokenizer(self):
        class Boom:
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("bound path consulted a tokenizer")

        _train, _val, meta = build_experiment(
            counterfactual_target="verified_steps", tokenizer=Boom(),
            length_source="bound")
        preflight = meta["length_preflight"]
        self.assertEqual(preflight["method"], "qwen2_byte_bpe_upper_bound")
        self.assertEqual(preflight["length_source"], "bound")
        self.assertFalse(preflight["tokenizer_pinned"])
        self.assertFalse(preflight["offline_qwen_tokenizer"])

    def test_auto_without_a_local_tokenizer_uses_the_bound(self):
        with patch("scripts.neral_v6_r16_gpu.load_pinned_qwen_tokenizer", return_value=None):
            _train, _val, meta = build_experiment(
                counterfactual_target="verified_steps", length_source="auto")
        before = meta["length_preflight"]["before_boilerplate_trim"]
        self.assertEqual(meta["length_preflight"]["length_source"], "bound")
        self.assertEqual(before["rows_over_max_length"], 70)
        self.assertEqual(before["max_upper_bound_tokens"], 933)

    def test_exact_length_source_requires_a_tokenizer(self):
        with patch("scripts.neral_v6_r16_gpu.load_pinned_qwen_tokenizer", return_value=None):
            with self.assertRaises(ValueError) as caught:
                build_experiment(
                    counterfactual_target="verified_steps", length_source="exact")
        self.assertIn("tokenizer", str(caught.exception).lower())

    def test_compaction_runs_only_when_the_measured_length_exceeds_budget(self):
        rows, _proofs = candidate()
        raw = next(row for row in rows
                   if prove_qwen_completion_bound(row)["upper_bound_tokens"] > 768)
        prefix = raw["messages"][:2]
        text = raw["messages"][2]["content"]
        kept, level = _fit_verified_assistant(prefix, text, length_of=lambda _row: 10)
        self.assertEqual(level, "verified_text")
        self.assertEqual(kept, text)
        fitted, trimmed = _fit_verified_assistant(prefix, text)
        self.assertNotEqual(trimmed, "verified_text")
        self.assertNotEqual(fitted, text)
        _assert_arithmetic_preserved(text, fitted)

    def test_dropping_parentheses_that_change_the_value_is_rejected(self):
        import re
        original = "CHECK:\nstep: (2) * (3 + 4) = 14\nFINAL_ANSWER: 14"
        changed = "CHECK:\nstep: 2 * 3 + 4 = 14\nFINAL_ANSWER: 14"
        self.assertEqual(re.findall(r"\d+", original), re.findall(r"\d+", changed))
        self.assertEqual(original.count("*"), changed.count("*"))
        self.assertEqual(original.count("+"), changed.count("+"))
        with self.assertRaises(ValueError) as caught:
            _assert_arithmetic_preserved(original, changed)
        self.assertIn("equation value", str(caught.exception))
        _assert_arithmetic_preserved(
            original, "CHECK:\nstep: 2 * (3 + 4) = 14\nFINAL_ANSWER: 14")
        prefix = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        line = "step: (2) * (3 + 4) = 14"
        long_text = "CHECK:\n" + "\n".join(line for _index in range(40)) + "\nFINAL_ANSWER: 14"
        with self.assertRaises(ValueError) as forced:
            _fit_verified_assistant(prefix, long_text, max_length=80)
        self.assertIn("equation value", str(forced.exception))

    def test_token_audit_artifact_records_exact_fit_without_a_score_claim(self):
        audit = json.loads(Path(__file__).resolve().parents[1].joinpath(
            "docs", "v6_verified_steps_token_audit.json").read_text(encoding="utf8"))
        self.assertEqual(audit["status"], "NERAL_V6_VERIFIED_STEP_TOKEN_AUDIT")
        self.assertEqual(audit["base_model"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(
            audit["base_revision"], "c03e6d358207e414f1eca0bb1891e29f1db0e242")
        self.assertEqual(audit["tokenizer_files_sha256"], dict(PINNED_TOKENIZER_SHA256))
        self.assertEqual(audit["untouched"]["rows"], 828)
        self.assertEqual(audit["untouched"]["rows_over_768"], 0)
        self.assertEqual(audit["untouched"]["max"], 451)
        self.assertEqual(
            audit["byte_bound_on_untouched_counterfactual"]["rows_over_768"], 70)
        self.assertEqual(audit["byte_bound_on_untouched_counterfactual"]["max"], 933)
        self.assertEqual(
            audit["byte_bound_on_untouched_counterfactual"][
                "rows_where_bound_underestimates_exact"], 0)
        self.assertEqual(audit["exact_decision"]["rows_compacted"], 0)
        self.assertEqual(audit["exact_decision"]["kept_verbatim"], 128)
        self.assertEqual(audit["bound_decision"]["rows_over_768_before_fit"], 70)
        self.assertFalse(audit["claimed_score_gain"])
        self.assertFalse(audit["training_performed"])
        self.assertFalse(audit["training_authorized"])
        self.assertFalse(audit["release_validated"])
        self.assertIn("matched_fewshot", audit["capability_effect"])

    def test_pinned_tokenizer_keeps_rows_verbatim_and_matches_the_audit(self):
        tokenizer = load_pinned_qwen_tokenizer()
        if tokenizer is None:
            self.skipTest(
                "Pinned Qwen tokenizer files for revision "
                "c03e6d358207e414f1eca0bb1891e29f1db0e242 are absent"
            )
        fresh = audit_verified_step_token_lengths(tokenizer)
        committed = json.loads(Path(__file__).resolve().parents[1].joinpath(
            "docs", "v6_verified_steps_token_audit.json").read_text(encoding="utf8"))
        for key in (
                "untouched", "untouched_by_split",
                "untouched_counterfactual_by_family",
                "bound_compacted_corpus", "bound_compacted_counterfactual",
                "byte_bound_on_untouched_counterfactual",
                "exact_decision", "bound_decision", "counterfactual_rows",
                "tokenizer_files_sha256", "chat_template_sha256",
                "model_chat_template_matches_training_template"):
            self.assertEqual(fresh[key], committed[key], key)
        self.assertEqual(fresh["exact_decision"]["kept_verbatim"], 128)
        self.assertEqual(fresh["untouched"]["rows_over_768"], 0)
        self.assertLessEqual(fresh["untouched"]["max"], 768)
        self.assertFalse(fresh["claimed_score_gain"])
        rows, _proofs = candidate()
        _train, _val, meta = build_experiment(
            counterfactual_target="verified_steps", tokenizer=tokenizer,
            length_source="exact")
        fitted = {row["messages"][1]["content"]: row["messages"][2]["content"]
                  for row in _train}
        for raw in rows:
            self.assertEqual(
                fitted[raw["messages"][1]["content"]],
                raw["messages"][2]["content"])
        self.assertTrue(meta["length_preflight"]["tokenizer_pinned"])
        self.assertEqual(meta["length_preflight"]["length_source"], "exact")
        self.assertEqual(
            meta["length_preflight"]["before_boilerplate_trim"]["rows_over_max_length"],
            0)

    def test_verified_training_entry_uses_the_pinned_tokenizer(self):
        import inspect
        from scripts.neral_v6_r16_gpu import main
        source = inspect.getsource(main)
        self.assertIn("load_pinned_qwen_tokenizer(args.base, required=True)", source)
        self.assertIn('length_source="exact"', source)


if __name__=="__main__":
    unittest.main()
