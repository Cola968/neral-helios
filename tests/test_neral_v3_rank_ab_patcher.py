"""CPU contract tests for reviewed, fixed-hyperparameter NERAL v3 rank 16/32 trainers."""
import ast
import unittest

from scripts.neral_v3_rank_ab_patcher import adapt_trainer, V3_TRAIN_STEPS, V3_LR
from scripts.neral_rank32_copy import patched_trainer, PatchError


class RankedV3PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        # Tests load original bytes only when installed in the sealed source checkout.
        cls.source = Path(__file__).resolve().parents[1]/'scripts'/'neral_qlora_source_fixture.py'

    def test_reviewed_original_patch_has_no_conditional_production_promotion(self):
        if not self.source.is_file():
            self.skipTest('Pinned trainer fixture is not in ordinary checkout')
        s=adapt_trainer(self.source.read_text(encoding='utf8'))
        ast.parse(s)
        self.assertIn('if args.steps != 80 or args.max_length != 512:',s)
        self.assertIn('learning_rate=5e-5, bf16=True',s)
        self.assertIn('if args.lora_rank == 32:',s)
        self.assertIn('if args.lora_rank == 16:',s)
        self.assertIn('from neral_adapter_candidate_r32 import seal_adapter',s)
        self.assertIn('from neral_adapter_candidate import seal_adapter',s)

    def test_parameters_are_frozen_identically_for_both_ranks(self):
        self.assertEqual(V3_TRAIN_STEPS,80)
        self.assertEqual(V3_LR,0.00005)

    def test_unrelated_source_rejected_without_writing(self):
        with self.assertRaises(PatchError):
            adapt_trainer('print("unapproved")')

    def test_exact_original_qlora_rank32_smoke_guard_cannot_be_ignored(self):
        if not self.source.is_file():
            self.skipTest('Pinned trainer fixture is not in ordinary checkout')
        before=patched_trainer(self.source.read_text(encoding='utf8'))
        self.assertIn('rank 32 is limited to the authorized 3-step smoke',before)
        after=adapt_trainer(self.source.read_text(encoding='utf8'))
        self.assertNotIn('rank 32 is limited to the authorized 3-step smoke',after)
        self.assertIn('exactly 80 steps',after)


if __name__=='__main__':
    unittest.main()
