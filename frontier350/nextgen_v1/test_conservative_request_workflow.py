"""CPU-only audit of conservative pilot workflow; does not train or access GPUs."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/neral-nextgen-v1-conservative-request.yml'


class ConservativeRequestWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding='utf8')

    def test_request_only_trigger_and_bounded_runtime(self):
        self.assertIn("paths: ['frontier350_nextgen_requests/conservative_pilot_20260916.json']", self.text)
        self.assertNotIn('  workflow_dispatch:', self.text)
        self.assertNotIn('  schedule:', self.text)
        self.assertIn('conservative-v1-20260916', self.text)
        self.assertIn('[int]$r.steps -ne 20', self.text)
        self.assertIn('[long]$r.tokens -ne 491520', self.text)

    def test_preflight_before_training(self):
        order = [self.text.index(x) for x in (
            'Validate one explicit, bounded local-only user request',
            'Verify SHA-pinned C original, D vault, corpus and free GPU BEFORE training',
            'Baseline on protected fixed validation and development suites',
            'Train isolated conservative 20 updates',
            'Evaluate separate 10-step checkpoint',
            'Compare pinned checkpoints')]
        self.assertEqual(order, sorted(order))
        self.assertIn('checkpoint_vault.py verify --backup', self.text)
        self.assertIn('real_gpu_preflight.py --anchor', self.text)
        self.assertIn('manifest_sha256 -ne $manifestSha', self.text)

    def test_non_destructive_and_quality_protection(self):
        self.assertIn('clean: false', self.text)
        self.assertIn('persist-credentials: false', self.text)
        self.assertIn('cancel-in-progress: false', self.text)
        self.assertIn('--snapshot-fraction 0.5', self.text)
        self.assertIn('midpoint_sha256 -ne $midSha', self.text)
        self.assertIn('checkpoint_sha256 -ne $finalSha', self.text)
        self.assertIn('promotion_allowed=$false', self.text)
        self.assertIn('no deletes', self.text)
        self.assertNotIn('Remove-Item', self.text)
        self.assertNotIn('torch.save(', self.text)
        self.assertIn('never checkpoint, optimizer or data bytes', self.text)

    def test_no_release_suite_used_to_tune(self):
        self.assertNotIn('nis2_core350.py', self.text)
        self.assertIn('eval_validation_loss350.py', self.text)
        self.assertIn('eval_recovery_dev_v2.py', self.text)
        self.assertIn('--lr 0.000005', self.text)
        self.assertIn('--replay-boost 1.0', self.text)


if __name__ == '__main__':
    unittest.main()
