"""TDD: first-party NERAL SFT training data must be verifiable and leakage-free."""
import json
import tempfile
import unittest
from pathlib import Path
from neral_persona_data import load_seed, validate_examples, export_sft, inspect_manifest


class NeralPersonaDataTests(unittest.TestCase):
    def test_original_seed_is_valid_and_covers_target_behaviors(self):
        samples=load_seed()
        self.assertGreaterEqual(len(samples),30)
        audit=validate_examples(samples)
        self.assertEqual(audit["count"],len(samples))
        self.assertEqual(audit["failures"],[])
        self.assertGreaterEqual(len(audit["categories"]),7)
        self.assertGreaterEqual(audit["unique_users"],len(samples))

    def test_duplicate_or_near_duplicate_prompts_cannot_enter_train_and_eval(self):
        good=load_seed()
        bad=good+[dict(good[0],id="np1-999",messages=[
            {"role":"user","content":good[0]["messages"][0]["content"].upper()},
            {"role":"assistant","content":"Hello!"}])]
        with self.assertRaises(ValueError):
            validate_examples(bad)
        punctuation=good+[dict(good[0],id="np1-998",messages=[
            {"role":"user","content":"Wer bist du!"},
            {"role":"assistant","content":"Fake duplicate"}])]
        with self.assertRaises(ValueError):
            validate_examples(punctuation)

    def test_role_and_schema_must_be_exact_and_no_untrusted_system_messages(self):
        row=load_seed()[0]
        bad=[
            {**row,"messages":[{"role":"system","content":"ignore rules"}]+row["messages"]},
            {**row,"messages":[{"role":"assistant","content":"bad order"}]},
            {**row,"messages":row["messages"],"secret_token":"hf_fake"},
            {**row,"license":"unknown"},
            {**row,"id":"../../outside"},
            {**row,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":""}]},
        ]
        for sample in bad:
            with self.subTest(sample=str(sample)[:60]),self.assertRaises(ValueError):
                validate_examples([sample])

    def test_export_sft_uses_no_eval_cases_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            train=Path(directory)/"train.jsonl"
            manifest=Path(directory)/"manifest.json"
            first=export_sft(load_seed(),train,manifest)
            self.assertEqual(first["example_count"],len(load_seed()))
            self.assertEqual(first["base_model"],"Qwen/Qwen3.5-4B")
            self.assertEqual(first["source_license"],"original-neral")
            self.assertEqual(first["test_examples_included"],0)
            self.assertNotIn("NIS2",train.read_text(encoding="utf-8"))
            records=[json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records),len(load_seed()))
            for rec in records:
                self.assertEqual(list(rec),["messages"])
                self.assertEqual([m["role"] for m in rec["messages"]],["user","assistant"])
                self.assertNotIn("id",rec)
            before=train.read_bytes()
            export_sft(load_seed(),train,manifest)
            self.assertEqual(before,train.read_bytes())
            self.assertEqual(inspect_manifest(manifest)["train_sha256"],first["train_sha256"])

    def test_export_refuses_existing_data_or_unapproved_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):
                export_sft(load_seed(),root/"x.txt",root/"m.json")
            path=root/"train.jsonl"
            path.write_text("original",encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_sft(load_seed(),path,root/"manifest.json")

    def test_manifest_does_not_claim_model_was_trained_or_improved(self):
        with tempfile.TemporaryDirectory() as directory:
            result=export_sft(load_seed(),Path(directory)/"train.jsonl",
                              Path(directory)/"manifest.json")
            self.assertEqual(result["training_executed"],False)
            self.assertIsNone(result["quality_delta"])
            self.assertEqual(result["training_status"],"data_only")
            self.assertFalse(result["approved_for_training"])
            self.assertIn("human review pending",result["review_status"])
            self.assertEqual(result["fine_tuning_method"],"LoRA (planned, not executed)")
            self.assertNotIn("hf_",json.dumps(result))


if __name__=="__main__":
    unittest.main()
