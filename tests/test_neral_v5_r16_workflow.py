"""NERAL v5 manual GPU trial safety and data-integrity gate, no GPU executed in CI."""
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v4_full_epoch_patcher import adapt_trainer, PatchError
from scripts.neral_v5_transfer_curriculum import build, verify
from scripts.neral_v5_trainer_bridge import prepare, check_export


class NeralV5ManualGpuContract(unittest.TestCase):
    def test_exact_pinned_original_480_optimizer_qlora(self):
        src=Path("tests/fixtures/neral_qlora_source_fixture.py").read_text(encoding="utf8")
        modified=adapt_trainer(src)
        self.assertIn("if args.steps != 480 or args.max_length != 512:",modified)
        self.assertIn("if args.lora_rank != 16:",modified)
        self.assertIn("learning_rate=5e-5, bf16=True",modified)
        self.assertIn("gradient_accumulation_steps=4",modified)
        self.assertIn("per_device_train_batch_size=1",modified)
        self.assertIn("completion_only_loss=True",modified)
        self.assertIn('save_strategy="no"',modified)
        self.assertIn('eval_strategy="no"',modified)

    def test_all_1920_new_math_transfer_rows_are_review_hash_verified(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            source=root/"source"
            target=root/"trainer"
            build(source)
            self.assertTrue(verify(source)["valid"])
            prep=prepare(source,target)
            self.assertEqual(prep["rows"],{"train":1920,"validation":320})
            self.assertEqual(prep["test_rows_excluded"],320)
            self.assertFalse(prep["training_allowed"])
            self.assertEqual(check_export(source,target)["rows"],prep["rows"])
            self.assertLess((target/"train.jsonl").stat().st_size,1_048_576)

    def test_only_explicit_manual_run_trains_480_steps_in_new_folder(self):
        flow=Path(".github/workflows/neral-v5-r16-480-training.yml").read_text(encoding="utf8")
        for required in (
            "workflow_dispatch",
            "TRAIN_NERAL_V5_R16_480_TRANSFER",
            "refs/heads/main",
            "runs-on: [self-hosted, windows, x64, neral-gpu]",
            "neral-qwen7b-r32-gpu-exclusive",
            "a082738f9145722dec443141e5d7801a3ead93af",
            "neral_math_v2_restore.py",
            "neral_v5_transfer_curriculum.py",
            "neral_v5_transfer_curriculum",
            "neral_v5_trainer_bridge",
            "neral_v5_token_preflight",
            "neral_v4_full_epoch_patcher",
            "neral_qlora_v5_r16_480.py",
            "v5-r16-480-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT",
            "TRAINED_NOT_DEPLOYED",
            "--check-review",
            "--steps 480",
            "--max-length 512",
            "--lora-rank 16",
            "--run-local-gpu",
            "nvidia-smi",
            "10000",
            "NERAL_V5_R16_480",
            "adapter_model.safetensors",
            "adapter_sha256",
            "upload-artifact",
        ):
            with self.subTest(required=required):
                self.assertIn(required,flow)
        for forbidden in (
            "schedule:",
            "workflow_run:",
            "git push",
            "--steps 200",
            "--lora-rank 32",
            "v4-r16-480-$env:GITHUB_RUN_ID",
            "--variant $variant",
            "neral_v4_novel_eval",
            "gh run",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden,flow)
        self.assertEqual(flow.count("--run-local-gpu"),1)
        self.assertLess(
            flow.index("Verify sealed adapter"),
            flow.index("Upload ONLY"),
        )

    def test_workflow_never_uploads_model_weights_or_private_training_data(self):
        flow=Path(".github/workflows/neral-v5-r16-480-training.yml").read_text(encoding="utf8")
        self.assertIn("path: control/v5_training_scalar/summary.json",flow)
        self.assertNotIn("path: $env:V5_RUN_ROOT",flow)
        self.assertIn("no_weights_or_prompts_uploaded",flow)
        self.assertIn("quality_improvement_proven=$false",flow)
        self.assertIn("adapter_promoted=$false",flow)
        self.assertIn("release_validated=$false",flow)


if __name__=="__main__":
    unittest.main()
