"""Synthetic contracts only: no owner data, real weights, GPU or approval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import phoenix_holdout_gate as gate


class FrozenGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.holdout = root / 'frozen.json'
        self.data = {'schema': 1, 'cases': [
            {'id': f'q{i}', 'language': 'de' if i % 2 else 'en',
             'domain': 'synthetic', 'prompt': f'Toy question {i}?', 'rubric': 'Toy rubric'}
            for i in range(8)]}
        raw = json.dumps(self.data).encode('utf-8')
        self.holdout.write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()
        self.args = SimpleNamespace(frozen_holdout=self.holdout, assets_root=root,
                                    train_data=root / 'train', valid_data=root / 'valid', seq=4)
        self.plan = {'lineage': {'holdout_sha256': self.sha}}
        self.calls = []
        self.report = {'case_count': 8, 'pattern_count': 16,
                       'train_case_ids_with_exact_hits': [],
                       'validation_case_ids_with_exact_hits': [],
                       'holdout_independence_verified': False,
                       'training_approved': False,
                       'full_corpus_token_ids_valid': True}

    def fixtures(self, *, actual_sha=None, result=None, vocabulary=16_384):
        base = ModuleType('helios350_preview')
        base.trusted_asset = lambda root, path, sha: root / 'frontier350_assets' / 'tokenizer.json'
        blind = ModuleType('phoenix_blind_ab_audit')
        blind.load = lambda path: (self.data, self.sha if actual_sha is None else actual_sha)
        blind.validate_holdout = lambda data: data['cases']
        lineage = ModuleType('phoenix_candidate_lineage')
        lineage.TOKENIZER_SHA = 'a' * 64
        overlap = ModuleType('phoenix_holdout_overlap_audit')
        def scan(cases, tokenizer, train, valid, seq):
            self.calls.append(('scan', len(cases), train, valid, seq))
            return self.report if result is None else result
        overlap.audit = scan
        shard = ModuleType('phoenix_shard_distribution_audit')
        shard.VOCAB_SIZE = 16_384
        tokenizer = ModuleType('tokenizers')
        tokenizer.Tokenizer = SimpleNamespace(from_file=lambda path: SimpleNamespace(
            get_vocab_size=lambda: vocabulary))
        return patch.dict(sys.modules, {'helios350_preview': base,
             'phoenix_blind_ab_audit': blind, 'phoenix_candidate_lineage': lineage,
             'phoenix_holdout_overlap_audit': overlap,
             'phoenix_shard_distribution_audit': shard, 'tokenizers': tokenizer})

    def test_clear_synthetic_result_preserves_nonproof_flags(self):
        with self.fixtures():
            output = gate.require_zero_exact_overlap(self.args, self.plan)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(output['exact_overlap_gate_passed'])
        self.assertFalse(output['holdout_independence_verified'])
        self.assertFalse(output['training_approved'])

    def test_train_overlap_refused_without_private_case_id_in_error(self):
        hits = dict(self.report, train_case_ids_with_exact_hits=['secret_private_case'])
        with self.fixtures(result=hits):
            with self.assertRaisesRegex(ValueError, 'pilot refused') as error:
                gate.require_zero_exact_overlap(self.args, self.plan)
        self.assertNotIn('secret_private_case', str(error.exception))

    def test_validation_overlap_refused(self):
        hits = dict(self.report, validation_case_ids_with_exact_hits=['q1'])
        with self.fixtures(result=hits):
            with self.assertRaisesRegex(ValueError, 'pilot refused'):
                gate.require_zero_exact_overlap(self.args, self.plan)

    def test_modified_holdout_refused_before_scanning_corpus(self):
        with self.fixtures(actual_sha='b' * 64):
            with self.assertRaisesRegex(ValueError, 'frozen holdout bytes changed'):
                gate.require_zero_exact_overlap(self.args, self.plan)
        self.assertEqual(self.calls, [])

    def test_missing_provenance_or_wrong_tokenizer_refused(self):
        with self.fixtures():
            with self.assertRaisesRegex(ValueError, 'lineage'):
                gate.require_zero_exact_overlap(self.args, {})
        with self.fixtures(vocabulary=16_383):
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                gate.require_zero_exact_overlap(self.args, self.plan)
        self.assertEqual(self.calls, [])

    def test_scanner_cannot_claim_independence_or_approval(self):
        for change in ({'holdout_independence_verified': True},
                       {'training_approved': True},
                       {'full_corpus_token_ids_valid': False},
                       {'case_count': 7}, {'pattern_count': 15}):
            with self.subTest(change=change), self.fixtures(result=dict(self.report, **change)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    gate.require_zero_exact_overlap(self.args, self.plan)
