"""Synthetic exporter integration: refuse wrong pilot holdout before model creation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_phoenix_real_answer_export import load_synthetic_module
import test_phoenix_candidate_lineage as fixture_module


class CandidateLoaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.CandidateLineageTests(
            methodName='test_matching_synthetic_contract_passes_without_training')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.checkpoint = self.fixture.checkpoint
        self.checkpoint.write_bytes(b'fake candidate fixture; never a checkpoint')
        self.sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.mod = load_synthetic_module()
        self.fixture.state['config'] = {'vocab_size': 16_384}
        self.loads = []
        fake_torch = types.ModuleType('torch')
        def fake_load(*args, **kwargs):
            self.loads.append((args, kwargs))
            return self.fixture.state
        fake_torch.load = fake_load
        fake_tokens = types.ModuleType('tokenizers')
        class Tokenizer:
            @classmethod
            def from_file(cls, filename):
                raise RuntimeError('tokenizer reached after metadata gate')
        fake_tokens.Tokenizer = Tokenizer
        fake_model = types.ModuleType('model350')
        class Config:
            @staticmethod
            def to_dict():
                return {'vocab_size': 16_384}
        fake_model.FRONTIER_350 = Config()
        fake_model.Neral350M = lambda: self.fail('model constructed too early')
        self.overrides = patch.dict(sys.modules, {
            'torch': fake_torch, 'tokenizers': fake_tokens, 'model350': fake_model})
        self.overrides.start()
        self.addCleanup(self.overrides.stop)

    def test_different_real_holdout_refused_after_file_hash_before_model(self):
        with self.assertRaisesRegex(ValueError, 'holdout lineage mismatch'):
            self.mod.load_candidate(self.checkpoint, self.sha,
                self.checkpoint.parent, allow_legacy_pickle=False,
                expected_holdout_sha='e' * 64)
        self.assertEqual(len(self.loads), 1)
        self.assertTrue(self.loads[0][1]['weights_only'])

    def test_matching_metadata_reaches_tokenizer_only_after_gate(self):
        with self.assertRaisesRegex(RuntimeError, 'tokenizer reached'):
            self.mod.load_candidate(self.checkpoint, self.sha,
                self.checkpoint.parent, allow_legacy_pickle=False,
                expected_holdout_sha=self.fixture.holdout)
        self.assertEqual(len(self.loads), 1)


if __name__ == '__main__':
    unittest.main()
