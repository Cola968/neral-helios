"""Synthetic, CPU-only checks of native fresh-stop safety; never train."""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from fresh_stop_boundary350 import require_fresh_stop_boundary as check


class FreshStopBoundaryTests(unittest.TestCase):
    def test_aligned_intermediate_stop(self):
        self.assertEqual(check(target=10_000, stop=4096, seq=128,
                               microbatch=2, grad_accum=4), 1024)

    def test_unaligned_intermediate_stop_refused(self):
        with self.assertRaisesRegex(ValueError, 'splits optimizer update'):
            check(target=10_000, stop=4100, seq=128, microbatch=2, grad_accum=4)

    def test_terminal_partial_update_permitted(self):
        self.assertEqual(check(target=4100, stop=4100, seq=128,
                               microbatch=2, grad_accum=4), 1024)

    def test_invalid_dimensions_and_limits(self):
        for args in (
            {'target': 0, 'stop': 1, 'seq': 1, 'microbatch': 1, 'grad_accum': 1},
            {'target': 5, 'stop': 6, 'seq': 1, 'microbatch': 1, 'grad_accum': 1},
            {'target': 5, 'stop': 4, 'seq': True, 'microbatch': 1, 'grad_accum': 1},
            {'target': 5, 'stop': 4, 'seq': 1, 'microbatch': 0, 'grad_accum': 1},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                check(**args)

    def native_prefix(self, filename):
        root = Path(__file__).resolve().parent
        tree = ast.parse((root / filename).read_text(encoding='utf8'))
        self.assertTrue(any(isinstance(node, ast.ImportFrom)
                            and node.module == 'fresh_stop_boundary350'
                            and any(alias.name == 'require_fresh_stop_boundary'
                                    for alias in node.names) for node in tree.body))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        found = [i for i, stmt in enumerate(main.body)
                 if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                 and isinstance(stmt.value.func, ast.Name)
                 and stmt.value.func.id == 'require_fresh_stop_boundary']
        self.assertEqual(len(found), 1)
        guard_idx = found[0]
        subsequent = main.body[guard_idx + 1:]
        calls_after = [ast.unparse(call.func) for stmt in subsequent
                       for call in ast.walk(stmt) if isinstance(call, ast.Call)]
        self.assertIn('torch.cuda.is_available', calls_after)
        self.assertIn('out.mkdir', calls_after)
        prefix = ast.fix_missing_locations(ast.Module(body=main.body[:guard_idx + 1],
                                                        type_ignores=[]))
        return compile(prefix, filename, 'exec')

    def test_both_native_entrypoints_call_guard_before_cuda_or_output(self):
        for filename in ('train350.py', 'train350_compiled.py'):
            with self.subTest(filename=filename):
                self.native_prefix(filename)

    def test_both_native_entrypoints_refuse_unaligned_before_side_effects(self):
        for filename in ('train350.py', 'train350_compiled.py'):
            with self.subTest(filename=filename):
                code = self.native_prefix(filename)
                argv = [filename, '--train-data', 'no-dataset', '--valid-data', 'no-valid',
                        '--out', 'no-output', '--tokenizer-sha256', 'synthetic',
                        '--seq', '128', '--microbatch', '2', '--grad-accum', '4',
                        '--target-tokens', '10000', '--stop-after-tokens', '4100']
                namespace = {'argparse': argparse, 'math': math,
                             'FRONTIER_350': types.SimpleNamespace(max_seq_len=512),
                             'require_fresh_stop_boundary': check}
                with patch.object(sys, 'argv', argv), self.assertRaisesRegex(
                        ValueError, 'splits optimizer update'):
                    exec(code, namespace)

    def test_both_native_entrypoints_accept_aligned_without_gpu(self):
        for filename in ('train350.py', 'train350_compiled.py'):
            with self.subTest(filename=filename):
                code = self.native_prefix(filename)
                argv = [filename, '--train-data', 'no-dataset', '--valid-data', 'no-valid',
                        '--out', 'no-output', '--tokenizer-sha256', 'synthetic',
                        '--seq', '128', '--microbatch', '2', '--grad-accum', '4',
                        '--target-tokens', '10000', '--stop-after-tokens', '4096']
                namespace = {'argparse': argparse, 'math': math,
                             'FRONTIER_350': types.SimpleNamespace(max_seq_len=512),
                             'require_fresh_stop_boundary': check}
                with patch.object(sys, 'argv', argv):
                    exec(code, namespace)


if __name__ == '__main__':
    unittest.main()
