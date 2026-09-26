"""Synthetic tests for the offline runner; no model weights or training."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

import phoenix_offline_cpu_check as check


class OfflineCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in (*check.SOURCES, *check.TESTS):
            (self.root / name).write_text('pass\n', encoding='utf8')

    def test_compilation_all_allowlisted_files(self):
        self.assertEqual(check.verify_sources(self.root), len(check.SOURCES) + len(check.TESTS))

    def test_missing_source_fails_before_test_execution(self):
        (self.root / check.SOURCES[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'missing, symlinked'):
            check.run(self.root, runner=lambda *a, **kw: self.fail('test runner launched'))

    def test_syntax_error_fails_before_test_execution(self):
        (self.root / check.SOURCES[0]).write_text('def broken(\n', encoding='utf8')
        with self.assertRaises(SyntaxError):
            check.run(self.root, runner=lambda *a, **kw: self.fail('test runner launched'))

    def test_cpu_isolation_and_exact_module_allowlist(self):
        seen = []
        def fake(command, **kwargs):
            seen.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, '', 'Ran 37 tests in 0.020s\n\nOK\n')
        result = check.run(self.root, runner=fake)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['tests_run'], 37)
        self.assertEqual(result['test_modules_selected'], 15)
        command, kwargs = seen[0]
        self.assertEqual(command[:4], [check.sys.executable, '-B', '-m', 'unittest'])
        self.assertEqual(command[5:], [name[:-3] for name in check.TESTS])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertTrue(kwargs['capture_output'])

    def test_fail_closed_on_nonzero_exit(self):
        fake = lambda command, **kw: subprocess.CompletedProcess(command, 1, '', 'Ran 3 tests in 0.1s\nFAILED\n')
        result = check.run(self.root, runner=fake)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['exit_code'], 1)

    def test_fail_closed_on_missing_test_summary(self):
        fake = lambda command, **kw: subprocess.CompletedProcess(command, 0, '', '')
        self.assertEqual(check.run(self.root, runner=fake)['status'], 'FAIL')

    def test_fail_closed_on_skipped_cases(self):
        fake = lambda command, **kw: subprocess.CompletedProcess(command, 0, '', 'Ran 37 tests in 0.1s\nOK (skipped=1)\n')
        self.assertEqual(check.run(self.root, runner=fake)['status'], 'FAIL')

    def test_timeout_returns_safe_failure(self):
        def fake(command, **kw):
            raise subprocess.TimeoutExpired(command, 180)
        self.assertEqual(check.run(self.root, runner=fake)['status'], 'TIMEOUT')

    def test_expanded_scope_covers_answer_export_and_evaluation(self):
        self.assertEqual(len(check.TESTS), 15)
        self.assertEqual(len(set(check.SOURCES + check.TESTS)), 26)
        for name in ('test_phoenix_real_answer_export.py',
                     'test_phoenix_candidate_loader_contract.py',
                     'test_phoenix_blind_ab_unblind.py',
                     'test_phoenix_paired_evidence.py'):
            self.assertIn(name, check.TESTS)

    def test_incomplete_test_collection_fails_even_with_ok(self):
        fake = lambda command, **kw: subprocess.CompletedProcess(command, 0, '', 'Ran 2 tests in 0.1s\n\nOK\n')
        self.assertEqual(check.run(self.root, runner=fake)['status'], 'FAIL')

    def test_new_source_syntax_error_fails_before_child(self):
        (self.root / 'phoenix_real_answer_export.py').write_text('def broken(\n')
        with self.assertRaises(SyntaxError):
            check.run(self.root, runner=lambda *a, **kw: self.fail('test runner launched'))


if __name__ == '__main__':
    unittest.main()
