"""CPU-only regression tests: model loading is deliberately not imported."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.neral_r32_compare import (
    TASKS, extract_code, docker_command, redact_runtime_result, summary_from_rows,
)


class CompareContracts(unittest.TestCase):
    def test_fixed_six_task_suite(self):
        self.assertEqual(len(TASKS), 6)
        self.assertEqual(len({t[0] for t in TASKS}), 6)

    def test_fenced_code_extraction(self):
        self.assertEqual(extract_code('```python\ndef test():\n    return 1\n```'),
                         'def test():\n    return 1\n')
        self.assertEqual(extract_code('x = 2'), 'x = 2\n')

    def test_docker_is_offline_and_bounded(self):
        with TemporaryDirectory() as directory:
            cmd = docker_command('python:3.11-alpine', Path(directory), 'safe-container', 1)
        self.assertIn('--network', cmd)
        self.assertIn('none', cmd)
        self.assertIn('--read-only', cmd)
        self.assertIn('--cap-drop', cmd)
        self.assertIn('ALL', cmd)
        self.assertIn('--pull', cmd)
        self.assertIn('never', cmd)
        self.assertNotIn('--privileged', cmd)
        self.assertNotIn('host', cmd)

    def test_unavailable_docker_is_never_success(self):
        rows = [{'task': 'even', 'passed': None, 'syntax_valid': True, 'execution_status': 'docker_unavailable'}]
        summary = summary_from_rows({'basis': rows, 'neral': rows})
        self.assertEqual(summary['runs']['basis']['tested'], 0)
        self.assertIsNone(summary['runs']['basis']['pass_rate'])
        self.assertIsNone(summary['runs']['basis']['passed'])
        self.assertFalse(summary['quality_gain_proven'])

    def test_counts_include_only_executed_tests(self):
        results = {'basis': [
            {'task': 'a', 'passed': True, 'syntax_valid': True, 'execution_status': 'executed'},
            {'task': 'b', 'passed': False, 'syntax_valid': False, 'execution_status': 'executed'},
        ], 'neral': [
            {'task': 'a', 'passed': False, 'syntax_valid': True, 'execution_status': 'executed'},
            {'task': 'b', 'passed': True, 'syntax_valid': True, 'execution_status': 'executed'},
        ]}
        summary = summary_from_rows(results)
        self.assertEqual(summary['paired']['baseline_only'], 1)
        self.assertEqual(summary['paired']['adapter_only'], 1)
        self.assertEqual(summary['paired']['both_pass'], 0)
        self.assertEqual(summary['paired']['neither_pass'], 0)
        self.assertFalse(summary['quality_gain_proven'])

    def test_missing_pair_not_passing(self):
        res = {'basis': [{'task': 'a', 'passed': True, 'execution_status': 'executed'}],
               'neral': [{'task': 'a', 'passed': None, 'execution_status': 'not_run'}]}
        self.assertEqual(summary_from_rows(res)['paired']['comparable'], 0)

    def test_runtime_error_redacted(self):
        row = redact_runtime_result({'output': 'SECRET', 'passed': False, 'returncode': 2})
        self.assertNotIn('SECRET', str(row))
        self.assertEqual(row['returncode'], 2)


if __name__ == '__main__':
    unittest.main()