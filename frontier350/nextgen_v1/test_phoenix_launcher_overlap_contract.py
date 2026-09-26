"""Static and mocked wrapper/native contract; never import Torch or train."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import phoenix_audited_pilot_launcher as launcher


class LauncherGateContract(unittest.TestCase):
    def test_real_native_main_contains_guards_before_cuda(self):
        path = Path(__file__).with_name('phoenix_weight_only_pilot.py')
        tree = ast.parse(path.read_text(encoding='utf8'), filename=str(path))
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        def import_index(name):
            return next(i for i, node in enumerate(main.body)
                        if isinstance(node, ast.ImportFrom) and node.module == name)
        audit = import_index('phoenix_audited_pilot_launcher')
        overlap = import_index('phoenix_holdout_gate')
        gpu = next(i for i, node in enumerate(main.body)
                   if isinstance(node, ast.Import) and
                   any(n.name == 'phoenix_cuda_diagnostic' for n in node.names))
        self.assertLess(audit, overlap)
        self.assertLess(overlap, gpu)
        native_text = ast.get_source_segment(path.read_text(encoding='utf8'), main)
        self.assertIn('verify_snapshot(args, plan)', native_text)
        self.assertIn('require_zero_exact_overlap(args, plan)', native_text)

    def test_wrapper_delegates_exactly_once_without_preflight_duplication(self):
        calls = []
        pilot = SimpleNamespace(main=lambda argv: calls.append(argv),
                                parser_args=lambda argv: self.fail('duplicate parse'),
                                prepare=lambda args: self.fail('duplicate approval'))
        with (patch.object(launcher.importlib, 'import_module', return_value=pilot),
              patch.object(launcher, 'verify_snapshot', side_effect=AssertionError('duplicate scan'))):
            launcher.main(['--resume', '/synthetic/resume.pt'])
        self.assertEqual(calls, [['--resume', '/synthetic/resume.pt']])

    def test_native_approval_failure_propagates_without_wrapper_retries(self):
        calls = []
        def reject(argv):
            calls.append(argv)
            raise ValueError('unapproved')
        with patch.object(launcher.importlib, 'import_module',
                          return_value=SimpleNamespace(main=reject)):
            with self.assertRaisesRegex(ValueError, 'unapproved'):
                launcher.main(['synthetic'])
        self.assertEqual(calls, [['synthetic']])

    def test_native_overlap_failure_propagates_without_wrapper_retries(self):
        calls = []
        def reject(argv):
            calls.append(argv)
            raise ValueError('frozen holdout exact-token overlap')
        with patch.object(launcher.importlib, 'import_module',
                          return_value=SimpleNamespace(main=reject)):
            with self.assertRaisesRegex(ValueError, 'exact-token overlap'):
                launcher.main(['synthetic'])
        self.assertEqual(calls, [['synthetic']])


if __name__ == '__main__':
    unittest.main()
