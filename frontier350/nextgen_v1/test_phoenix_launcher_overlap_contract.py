"""Mocked launcher orchestration only; no trainer/GPU/owner approval."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import phoenix_audited_pilot_launcher as launcher


class LauncherGateContract(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace()
        self.plan = {}
        self.calls = []
        self.pilot = SimpleNamespace(
            parser_args=lambda argv: self.args,
            prepare=lambda args: self.calls.append('approve') or self.plan,
            main=lambda argv: self.calls.append('train'))
        self.sample_summary = {'train': {'shards': 1}, 'validation': {'shards': 1},
                               'all_token_ids_in_vocab': True}

    def test_no_training_when_exact_overlap_detected(self):
        with (patch.object(launcher.importlib, 'import_module', return_value=self.pilot),
              patch.object(launcher, 'verify_snapshot', side_effect=lambda *a: self.calls.append('audit') or self.sample_summary),
              patch.object(launcher, 'require_zero_exact_overlap', side_effect=ValueError('pilot refused'))):
            with self.assertRaisesRegex(ValueError, 'pilot refused'):
                launcher.main([])
        self.assertEqual(self.calls, ['approve', 'audit'])

    def test_no_overlap_runs_original_pilot_only_after_both_checks(self):
        def gate(*args):
            self.calls.append('overlap')
            return {'exact_overlap_gate_passed': True}
        with (patch.object(launcher.importlib, 'import_module', return_value=self.pilot),
              patch.object(launcher, 'verify_snapshot', side_effect=lambda *a: self.calls.append('audit') or self.sample_summary),
              patch.object(launcher, 'require_zero_exact_overlap', side_effect=gate)):
            launcher.main([])
        self.assertEqual(self.calls, ['approve', 'audit', 'overlap', 'train'])

    def test_no_overlap_scan_when_original_approval_fails(self):
        self.pilot.prepare = lambda args: (_ for _ in ()).throw(ValueError('unapproved'))
        with (patch.object(launcher.importlib, 'import_module', return_value=self.pilot),
              patch.object(launcher, 'verify_snapshot', side_effect=AssertionError('should not audit')),
              patch.object(launcher, 'require_zero_exact_overlap', side_effect=AssertionError('should not scan'))):
            with self.assertRaisesRegex(ValueError, 'unapproved'):
                launcher.main([])
        self.assertEqual(self.calls, [])
