"""CPU-only source-contract tests for the ACTUAL native pilot main entrypoint.

Extract only the pre-CUDA prologue from the real trainer AST. Execute that
prologue with mocked approval/audit/gate; never import Torch, invoke a GPU,
read owner data, or issue a training authorization. This does not test an actual
checkpoint run or the installed tokenizer dependency.
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
    source = SOURCE.read_text(encoding='utf8')
    tree = ast.parse(source, filename=str(SOURCE))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    cuda_index = next(i for i, node in enumerate(main.body)
                      if isinstance(node, ast.Import) and
                      any(name.name == 'phoenix_cuda_diagnostic' for name in node.names))
    prefix = main.body[:cuda_index]
    if len(prefix) < 9 or not any(isinstance(node, ast.ImportFrom) and
                                  node.module == 'phoenix_holdout_gate' for node in prefix):
        raise AssertionError('missing holdout gate before CUDA in original trainer')
    if not any(isinstance(node, ast.ImportFrom) and
               node.module == 'phoenix_audited_pilot_launcher' for node in prefix):
        raise AssertionError('missing shard audit before CUDA in original trainer')
    clone = ast.FunctionDef(name='main', args=main.args, body=[*prefix, ast.Return(value=None)],
                            decorator_list=[], returns=main.returns, type_comment=None)
    isolated = ast.fix_missing_locations(ast.Module(body=[clone], type_ignores=[]))
    namespace = {'json': json}
    exec(compile(isolated, str(SOURCE), 'exec'), namespace)
    return namespace


class DirectPilotGateTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.args = SimpleNamespace(resume=None)
        self.plan = {'lineage': {'holdout_sha256': '0' * 64}}
        self.audit = ModuleType('phoenix_audited_pilot_launcher')
        self.gate = ModuleType('phoenix_holdout_gate')
        self.audit.verify_snapshot = lambda args, plan: self.record_audit(args, plan)
        self.gate.require_zero_exact_overlap = lambda args, plan: self.record_gate(args, plan)
        self.namespace = isolated_prologue()
        self.namespace['parser_args'] = self.parser
        self.namespace['prepare'] = self.prepare

    def parser(self, argv):
        self.events.append('parse')
        return self.args

    def prepare(self, args):
        self.events.append('approval')
        return self.plan

    def record_audit(self, args, plan):
        self.events.append('audit')
        return {'all_token_ids_in_vocab': True,
                'train': {'shards': 1}, 'validation': {'shards': 1}}

    def record_gate(self, args, plan):
        self.events.append('gate')
        return {'exact_overlap_gate_passed': True}

    def invoke(self):
        with patch.dict(sys.modules, {'phoenix_audited_pilot_launcher': self.audit,
                                      'phoenix_holdout_gate': self.gate}):
            return self.namespace['main'](['synthetic'])

    def test_direct_entrypoint_requires_ordered_checks_before_cuda(self):
        self.invoke()
        self.assertEqual(self.events, ['parse', 'approval', 'audit', 'gate'])

    def test_explicit_resume_cannot_skip_integrity(self):
        self.args.resume = Path('/synthetic/resume.pt')
        self.invoke()
        self.assertEqual(self.events, ['parse', 'approval', 'audit', 'gate'])

    def test_rejected_operator_approval_prevents_both_scans(self):
        def refuse(args):
            self.events.append('approval')
            raise ValueError('no real approval')
        self.namespace['prepare'] = refuse
        with self.assertRaisesRegex(ValueError, 'no real approval'):
            self.invoke()
        self.assertEqual(self.events, ['parse', 'approval'])

    def test_corpus_failure_prevents_holdout_gate_and_cuda(self):
        def refuse(args, plan):
            self.events.append('audit')
            raise ValueError('invalid shard')
        self.audit.verify_snapshot = refuse
        with self.assertRaisesRegex(ValueError, 'invalid shard'):
            self.invoke()
        self.assertEqual(self.events, ['parse', 'approval', 'audit'])

    def test_detected_holdout_overlap_prevents_cuda(self):
        def refuse(args, plan):
            self.events.append('gate')
            raise ValueError('frozen holdout exact-token overlap')
        self.gate.require_zero_exact_overlap = refuse
        with self.assertRaisesRegex(ValueError, 'exact-token overlap'):
            self.invoke()
        self.assertEqual(self.events, ['parse', 'approval', 'audit', 'gate'])

    def test_false_gate_result_fails_closed(self):
        self.gate.require_zero_exact_overlap = lambda args, plan: {'exact_overlap_gate_passed': False}
        with self.assertRaisesRegex(ValueError, 'mandatory pilot'):
            self.invoke()

    def test_false_corpus_status_fails_closed(self):
        self.audit.verify_snapshot = lambda args, plan: {'all_token_ids_in_vocab': False}
        with self.assertRaisesRegex(ValueError, 'mandatory pilot'):
            self.invoke()


if __name__ == '__main__':
    unittest.main()
