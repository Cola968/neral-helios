"""Exact full-epoch NERAL v4 pilot: 480 optimizer steps, rank16, no promotion."""
import ast
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v4_full_epoch_patcher import adapt_trainer, write_variant
from scripts.neral_rank32_copy import PatchError

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"tests/fixtures/neral_qlora_source_fixture.py"


class FullEpoch480Tests(unittest.TestCase):
    def test_exact_480_steps_and_original_pinned_model(self):
        patched=adapt_trainer(ORIGINAL.read_text(encoding="utf8"))
        ast.parse(patched)
        self.assertIn("if args.steps != 480 or args.max_length != 512:",patched)
        self.assertIn("if not 1 <= args.steps <= 480",patched)
        self.assertIn("Only rank 16 allowed for NERAL v4 pilot",patched)
        self.assertIn("40_370_176",patched)
        self.assertIn("learning_rate=5e-5, bf16=True",patched)
        self.assertIn("gradient_accumulation_steps=4",patched)
        self.assertIn("per_device_train_batch_size=1",patched)
        self.assertIn("completion_only_loss=True",patched)
        self.assertNotIn("if args.steps != 200",patched)
        self.assertNotIn("if not 1 <= args.steps <= 200",patched)
        self.assertNotIn("eval_strategy=\"steps\"",patched)

    def test_untrusted_source_and_destination_rejected(self):
        with self.assertRaises(PatchError):
            adapt_trainer("print('not-the-pinned-trainer')")
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/"original.py"
            src.write_bytes(ORIGINAL.read_bytes())
            target=root/"full_epoch.py"
            metadata=write_variant(src,target)
            self.assertEqual(metadata["steps"],480)
            self.assertEqual(metadata["examples_per_epoch"],1920)
            self.assertEqual(metadata["optimizer_steps_per_epoch"],480)
            self.assertEqual(metadata["rank"],16)
            self.assertFalse(metadata["training_performed"])
            self.assertEqual(src.read_bytes(),ORIGINAL.read_bytes())
            with self.assertRaises(PatchError):
                write_variant(src,target)

    def test_manual_full_epoch_workflow_stays_away_from_old_adapters(self):
        w=(ROOT/".github/workflows/neral-v4-full-epoch.yml").read_text(encoding="utf8")
        for text in (
            "workflow_dispatch","TRAIN_NERAL_V4_R16_480_FULL_EPOCH",
            "runs-on: [self-hosted, windows, x64, neral-gpu]",
            "--steps 480","--lora-rank 16","--run-local-gpu",
            "neral_v4_full_epoch_patcher","neral_v4_token_preflight",
            "neral_v4_trainer_bridge","neral_v4_novel_eval",
            "--variant $variant","--combine","480 optimizer steps",
            "neral-v4-novel-128-paired-scores"):
            self.assertIn(text,w)
        self.assertNotIn("schedule:",w)
        self.assertNotIn("workflow_run:",w)
        self.assertNotIn("git push",w)
        self.assertNotIn("release_validated=$true",w)
        self.assertNotIn("adapter_promoted=$true",w)
        self.assertNotIn("--steps 200",w)
        self.assertIn("v4-r16-480-$env:GITHUB_RUN_ID",w)

    def test_v4_train_inventory_mathematically_matches_one_epoch(self):
        self.assertEqual(480*4*1,1920)
        from scripts.neral_v4_curriculum import ROWS
        self.assertEqual(ROWS["train"],1920)


if __name__=="__main__":
    unittest.main()
