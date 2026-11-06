"""Tests for immutable v3 R32 training source transformation."""
import ast
import os
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v3_rank32_patch import patch, write_checked, PatchError


class Rank32TrainerContract(unittest.TestCase):
    def source(self):
        folder = Path(os.environ.get("NERAL_PINNED_TRAINER_ROOT", "trainer"))
        path = folder / "scripts" / "neral_qlora_pilot.py"
        if not path.is_file():
            self.skipTest("Original pinned QLoRA source unavailable in this checkout")
        return path.read_text(encoding="utf-8")

    def test_original_pinned_trainer_only(self):
        generated = patch(self.source())
        ast.parse(generated)
        for required in ("80_740_352", "r=args.lora_rank",
                         "learning_rate=5e-5", "max_steps=args.steps",
                         "gradient_checkpointing=True", 'optim="paged_adamw_8bit"',
                         "TRAIN_NERAL_V3_R32_120_STEPS", "11.5 * 1024 ** 3"):
            self.assertIn(required, generated)

    def test_mutated_trainer_rejected(self):
        with self.assertRaises(PatchError):
            patch(self.source() + "\n# unreviewed source\n")

    def test_new_file_only_and_no_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "original.py"
            original.write_text(self.source(), encoding="utf-8", newline="\n")
            output = root / "candidate.py"
            summary = write_checked(original, output)
            self.assertEqual(summary["status"], "SOURCE_PATCH_ONLY_NOT_TRAINED")
            self.assertEqual(summary["rank"], 32)
            self.assertEqual(summary["steps"], 120)
            with self.assertRaises(PatchError):
                write_checked(original, output)
