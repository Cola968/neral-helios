"""Real Phoenix module integration with fabricated shards and an ASCII tokenizer.

The actual gate, holdout validator, exact matcher and shard auditor are imported
unmodified. Only the external tokenizer asset loader and Tokenizer.from_file are
substituted; no owner assets, Torch, GPU, grading or training are accessed.
"""
from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import helios350_preview as preview
import phoenix_holdout_gate as gate


class AsciiTokenizer:
    def get_vocab_size(self):
        return 16_384

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('unexpected tokenizer mode')
        return SimpleNamespace(ids=[ord(char) for char in text])


class RealModuleGateIntegration(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.train, self.valid = root / 'train', root / 'valid'
        self.train.mkdir()
        self.valid.mkdir()
        self.holdout = root / 'holdout.json'
        self.cases = [{'id': f'case_{index}',
                       'language': 'de' if index % 2 else 'en',
                       'domain': 'toy', 'prompt': f'Secret question {index}?',
                       'rubric': 'Dummy expected answer'} for index in range(8)]
        raw = json.dumps({'schema': 1, 'cases': self.cases}).encode('utf-8')
        self.holdout.write_bytes(raw)
        self.plan = {'lineage': {'holdout_sha256': hashlib.sha256(raw).hexdigest()}}
        self.args = SimpleNamespace(frozen_holdout=self.holdout,
                                    assets_root=root, train_data=self.train,
                                    valid_data=self.valid, seq=2)
        self.shard(self.train / 'train.bin', [1000, 1001, 1002, 1003])
        self.shard(self.valid / 'valid.bin', [1004, 1005, 1006, 1007])
        tok = ModuleType('tokenizers')
        tok.Tokenizer = SimpleNamespace(from_file=lambda path: AsciiTokenizer())
        self.stub_tokenizer = tok

    @staticmethod
    def shard(path, numbers):
        tokens = array('H', numbers)
        if sys.byteorder != 'little':
            tokens.byteswap()
        path.write_bytes(tokens.tobytes())

    def evaluate(self):
        with (patch.object(preview, 'trusted_asset', return_value=self.args.assets_root / 'tokenizer.json'),
              patch.dict(sys.modules, {'tokenizers': self.stub_tokenizer})):
            return gate.require_zero_exact_overlap(self.args, self.plan)

    def test_full_canonical_matcher_clear_tiny_shards(self):
        report = self.evaluate()
        self.assertTrue(report['exact_overlap_gate_passed'])
        self.assertFalse(report['holdout_independence_verified'])

    def test_full_canonical_matcher_refuses_training_contamination(self):
        text = self.cases[0]['prompt']
        self.shard(self.train / 'train.bin', [999, *[ord(c) for c in text], 998])
        with self.assertRaisesRegex(ValueError, 'pilot refused'):
            self.evaluate()

    def test_full_canonical_matcher_refuses_validation_contamination(self):
        text = 'User: ' + self.cases[1]['prompt'] + '\nAssistant:'
        self.shard(self.valid / 'valid.bin', [999, *[ord(c) for c in text], 998])
        with self.assertRaisesRegex(ValueError, 'pilot refused'):
            self.evaluate()

    def test_full_canonical_matcher_refuses_invalid_uint16(self):
        self.shard(self.train / 'train.bin', [1, 2, 16_384, 4])
        with self.assertRaisesRegex(ValueError, 'out-of-vocabulary'):
            self.evaluate()

    def test_actual_holdout_file_hash_change_refused(self):
        self.holdout.write_text(json.dumps({'schema': 1, 'cases': self.cases}, indent=2))
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.evaluate()


if __name__ == '__main__':
    unittest.main()
