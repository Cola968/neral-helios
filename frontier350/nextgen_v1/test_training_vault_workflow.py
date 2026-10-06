"""CPU-only source contract: no automatic GPU run or unprotected anchor."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github' / 'workflows' / 'neral-nextgen-v1-bounded-quality-train.yml'


class TrainingVaultWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding='utf-8')

    def test_training_requires_explicit_manual_dispatch(self):
        self.assertIn('on:\n  workflow_dispatch:', self.text)
        self.assertNotIn('  push:', self.text)
        self.assertNotIn('  schedule:', self.text)

    def test_verified_backup_precedes_data_and_training(self):
        vault = self.text.index('Require independently verified SAFE source AND distinct-volume immutable backup')
        preflight = self.text.index('Check tokenizer, shards, GPU, available disk')
        training = self.text.index('Bounded isolated research training ONLY')
        self.assertLess(vault, preflight)
        self.assertLess(preflight, training)
        self.assertIn('checkpoint_vault.py verify --backup', self.text)
        self.assertIn('SAFE backup missing; run dedicated vault workflow first', self.text)
        self.assertIn('source and replica share one volume', self.text)

    def test_source_pin_and_no_checkout_cleaning(self):
        self.assertIn('ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c', self.text)
        self.assertIn('clean: false', self.text)
        self.assertIn('cancel-in-progress: false', self.text)
        self.assertIn('persist-credentials: false', self.text)

    def test_quality_regression_rejects_without_promotion(self):
        self.assertIn('MEASURED QUALITY REGRESSION: candidate rejected', self.text)
        self.assertIn('development.no_regression_in_this_dev_suite', self.text)
        self.assertIn('generation_health_repeated_trigram_pct.delta_candidate_minus_safe -gt 0', self.text)
        self.assertNotIn('torch.save(', self.text)
        self.assertIn('NEVER weights, optimizer or corpus', self.text)


if __name__ == '__main__':
    unittest.main()