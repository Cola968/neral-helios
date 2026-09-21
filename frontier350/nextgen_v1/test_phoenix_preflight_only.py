"""Synthetic CPU tests: no real private files, GPU, model or training."""
from __future__ import annotations

import io
from contextlib import redirect_stdout
import json
import sys
import types
import unittest
from unittest.mock import patch

pilot = types.ModuleType('phoenix_weight_only_pilot')
pilot.parser_args = lambda argv=None: types.SimpleNamespace()
pilot.prepare = lambda args: {'out': 'synthetic'}
launcher = types.ModuleType('phoenix_audited_pilot_launcher')
launcher.verify_snapshot = lambda args, plan: {
    'all_token_ids_in_vocab': True, 'train': {'shards': 2},
    'validation': {'shards': 1}}
holdout = types.ModuleType('phoenix_holdout_gate')
holdout.require_zero_exact_overlap = lambda args, plan: {
    'exact_overlap_gate_passed': True}

with patch.dict(sys.modules, {
    'phoenix_weight_only_pilot': pilot,
    'phoenix_audited_pilot_launcher': launcher,
    'phoenix_holdout_gate': holdout,
}):
    import phoenix_preflight_only as check


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {
            'phoenix_weight_only_pilot': pilot,
            'phoenix_audited_pilot_launcher': launcher,
            'phoenix_holdout_gate': holdout,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_pass_is_not_training_or_gpu_proof(self):
        result = check.preflight([])
        self.assertEqual(result['status'], 'CPU_PREFLIGHT_PASS_GPU_NOT_CHECKED')
        self.assertFalse(result['training_started'])
        self.assertFalse(result['checkpoint_written'])
        self.assertFalse(result['gpu_fit_or_training_speed_verified'])
        self.assertFalse(result['independent_human_review_verified_by_script'])
        self.assertEqual(result['new_tokens_processed'], 0)

    def test_native_prepare_failure_refuses_without_scanning(self):
        with patch.object(check, 'prepare', side_effect=ValueError('private_path')):
            with patch.object(launcher, 'verify_snapshot', side_effect=AssertionError('reached')):
                with redirect_stdout(io.StringIO()) as output:
                    status = check.main([])
        self.assertEqual(status, 1)
        self.assertNotIn('private_path', output.getvalue())
        self.assertEqual(json.loads(output.getvalue())['status'], 'CPU_PREFLIGHT_FAIL')

    def test_missing_vocab_pass_is_rejected(self):
        with patch.object(launcher, 'verify_snapshot', return_value={
            'all_token_ids_in_vocab': False}):
            with self.assertRaisesRegex(ValueError, 'must explicitly pass'):
                check.preflight([])

    def test_holdout_gate_failure_is_rejected(self):
        with patch.object(holdout, 'require_zero_exact_overlap', return_value={
            'exact_overlap_gate_passed': False}):
            with self.assertRaisesRegex(ValueError, 'must explicitly pass'):
                check.preflight([])

    def test_json_success_and_exit_zero(self):
        with redirect_stdout(io.StringIO()) as output:
            status = check.main([])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())['native_manifest_consistency_checked'])


if __name__ == '__main__':
    unittest.main()
