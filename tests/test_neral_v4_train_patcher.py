"""CPU-only proof of the separately authorized NERAL v4 rank-16 trial."""
import ast
import tempfile
import unittest
from pathlib import Path
from scripts.neral_rank32_copy import PatchError
from scripts.neral_v4_train_patcher import adapt_trainer, write_variant

ROOT=Path(__file__).resolve().parents[1]
PINNED=ROOT/"tests/fixtures/neral_qlora_source_fixture.py"


class V4TrainPatcher(unittest.TestCase):
    def test_exact_pinned_source_produces_200_step_rank16_only_policy(self):
        original=PINNED.read_text(encoding="utf8")
        edited=adapt_trainer(original)
        ast.parse(edited)
        self.assertIn("if args.steps != 200 or args.max_length != 512:",edited)
        self.assertIn("learning_rate=5e-5, bf16=True",edited)
        self.assertIn("gradient_accumulation_steps=4",edited)
        self.assertIn("completion_only_loss=True",edited)
        self.assertIn("if args.lora_rank == 16:",edited)
        self.assertIn("40_370_176",edited)
        self.assertIn("Only rank 16 allowed for NERAL v4 pilot",edited)
        self.assertIn('\"lora_rank\": args.lora_rank',edited)
        self.assertNotIn("if args.steps != 80 or args.max_length != 512:",edited)
        self.assertNotIn("eval_strategy=\"steps\"",edited)
        self.assertNotIn("save_strategy=\"steps\"",edited)

    def test_unreviewed_or_corrupted_pinned_source_rejected(self):
        with self.assertRaises(PatchError):
            adapt_trainer("print('untrusted training')")
        source=PINNED.read_text(encoding="utf8")
        with self.assertRaises(PatchError):
            adapt_trainer(source.replace('learning_rate=2e-4','learning_rate=3e-4'))

    def test_scratch_only_copy_never_overwrites_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=root/"original.py"
            original.write_bytes(PINNED.read_bytes())
            copy=root/"trial.py"
            report=write_variant(original,copy)
            self.assertEqual(report["steps"],200)
            self.assertEqual(report["lora_rank"],16)
            self.assertFalse(report["training_performed"])
            self.assertEqual(original.read_bytes(),PINNED.read_bytes())
            self.assertTrue(copy.is_file())
            with self.assertRaises((PatchError,ValueError)):
                write_variant(original,copy)

    def test_workflow_requires_manual_user_approval_and_cannot_auto_promote(self):
        src=(ROOT/".github/workflows/neral-v4-r16-training.yml").read_text(encoding="utf8")
        self.assertIn("TRAIN_NERAL_V4_R16_200_EXPERIMENTAL",src)
        self.assertIn("workflow_dispatch",src)
        self.assertNotIn("schedule:",src)
        self.assertNotIn("workflow_run:",src)
        self.assertIn("runs-on: [self-hosted, windows, x64, neral-gpu]",src)
        self.assertIn("HF_HUB_OFFLINE",src)
        self.assertIn("--run-local-gpu",src)
        self.assertIn("--steps 200",src)
        self.assertIn("--check-review",src)
        self.assertIn("neral_v4_token_preflight",src)
        self.assertIn("neral_v4_trainer_bridge",src)
        self.assertIn("nvidia-smi",src)
        self.assertNotIn("release_validated: true",src)
        self.assertNotIn("models/latest",src)
        self.assertNotIn("huggingface-cli upload",src)
        self.assertNotIn("git push",src)


if __name__=="__main__":
    unittest.main()
