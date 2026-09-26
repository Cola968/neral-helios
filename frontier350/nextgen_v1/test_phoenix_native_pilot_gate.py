"""Read and execute only the real pilot's pre-CUDA AST with fake inputs.

No native pilot import, Torch, checkpoint, user corpus, GPU or training.
Approval, corpus and overlap failures must precede GPU diagnostic import.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('phoenix_weight_only_pilot.py')


def isolated_prologue():
    tree = ast.parse(SOURCE.read_text(encoding='utf8'), filename=str(SOURCE))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    cuda = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Import)
                and any(alias.name == 'phoenix_cuda_diagnostic' for alias in n.names))
    prefix = main.body[:cuda]
    imports = {n.module for n in prefix if isinstance(n, ast.ImportFrom)}
    if len(prefix) < 8 or not {'phoenix_audited_pilot_launcher', 'phoenix_holdout_gate'} <= imports:
        raise AssertionError('original pilot missing mandatory corpus/holdout gates before CUDA')
    main.body = [*prefix, ast.Return(value=None)]
    main.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
    env = {'json': json}
    exec(compile(module, str(SOURCE), 'exec'), env)
    return env


class DirectPilotGateTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.args = SimpleNamespace(resume=None)
        self.plan = {'lineage': {'holdout_sha256': '0' * 64}}
        self.audit = ModuleType('phoenix_audited_pilot_launcher')
        self.gate = ModuleType('phoenix_holdout_gate')
        self.audit.verify_snapshot = self.good_audit
        self.gate.require_zero_exact_overlap = self.good_gate
        self.env = isolated_prologue()
        self.env['parser_args'] = lambda argv: self.mark('parse', self.args)
        self.env['prepare'] = lambda args: self.mark('approval', self.plan)

    def mark(self, event, value):
        self.calls.append(event)
        return value

    def good_audit(self, args, plan):
        return self.mark('audit', {'all_token_ids_in_vocab': True,
                                   'train': {'shards': 1}, 'validation': {'shards': 1}})

    def good_gate(self, args, plan):
        return self.mark('gate', {'exact_overlap_gate_passed': True})

    def invoke(self):
        with patch.dict(sys.modules, {'phoenix_audited_pilot_launcher': self.audit,
                                      'phoenix_holdout_gate': self.gate}):
            return self.env['main'](['test-only'])

    def test_direct_trainer_has_required_order(self):
        self.invoke()
        self.assertEqual(self.calls, ['parse', 'approval', 'audit', 'gate'])

    def test_resume_has_the_same_required_order(self):
        self.args.resume = Path('/synthetic/resume.pt')
        self.invoke()
        self.assertEqual(self.calls, ['parse', 'approval', 'audit', 'gate'])

    def test_approval_failure_does_not_scan_or_train(self):
        self.env['prepare'] = lambda args: self.refuse('approval', 'not approved')
        with self.assertRaisesRegex(ValueError, 'not approved'):
            self.invoke()
        self.assertEqual(self.calls, ['parse', 'approval'])

    def test_bad_shard_does_not_reach_holdout_or_gpu(self):
        self.audit.verify_snapshot = lambda a, p: self.refuse('audit', 'invalid shard')
        with self.assertRaisesRegex(ValueError, 'invalid shard'):
            self.invoke()
        self.assertEqual(self.calls, ['parse', 'approval', 'audit'])

    def test_holdout_hit_does_not_reach_gpu(self):
        self.gate.require_zero_exact_overlap = lambda a, p: self.refuse('gate', 'exact-token overlap')
        with self.assertRaisesRegex(ValueError, 'exact-token overlap'):
            self.invoke()
        self.assertEqual(self.calls, ['parse', 'approval', 'audit', 'gate'])

    def test_false_shard_report_refused(self):
        self.audit.verify_snapshot = lambda a, p: {'all_token_ids_in_vocab': False}
        with self.assertRaisesRegex(ValueError, 'mandatory pilot'):
            self.invoke()

    def test_false_overlap_status_refused(self):
        self.gate.require_zero_exact_overlap = lambda a, p: {'exact_overlap_gate_passed': False}
        with self.assertRaisesRegex(ValueError, 'mandatory pilot'):
            self.invoke()

    def refuse(self, name, message):
        self.calls.append(name)
        raise ValueError(message)


if __name__ == '__main__':
    unittest.main()
