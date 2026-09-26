"""Synthetic private-file CLI tests; model/tokenizer loading is stubbed, never trained."""
from __future__ import annotations

from array import array
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import phoenix_holdout_overlap_audit as overlap


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) for c in text])

    def get_vocab_size(self):
        return 16_384


class OverlapCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / 'NeralPrivateAssets' / 'phoenix'
        self.private.mkdir(parents=True)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.train = self.root / 'train'
        self.valid = self.root / 'valid'
        self.train.mkdir()
        self.valid.mkdir()
        cases = [{'id': f'q{i}', 'language': 'de' if i % 2 == 0 else 'en',
                  'domain': 'toy', 'prompt': f'Question {i}?', 'rubric': 'Manual grading.'}
                 for i in range(8)]
        self.holdout = self.private / 'frozen.json'
        self.holdout.write_text(json.dumps({'schema': 1, 'cases': cases}), encoding='utf-8')
        self.shard(self.train / 'a.bin', [700, *map(ord, 'Question 0?'), 701])
        self.shard(self.valid / 'v.bin', [702, *map(ord, 'unrelated'), 703])
        self.argv = ['overlap.py', '--execute', '--holdout', str(self.holdout),
                     '--assets-root', str(self.assets), '--train-data', str(self.train),
                     '--valid-data', str(self.valid), '--private-dir', str(self.private),
                     '--tag', 'toy', '--seq', '2']
        base = ModuleType('helios350_preview')
        base.TOKENIZER_SHA = 'a' * 64
        base.trusted_asset = lambda root, path, sha: path
        tokens = ModuleType('tokenizers')
        tokens.Tokenizer = SimpleNamespace(from_file=lambda filename: ToyTokenizer())
        self.modules = {'helios350_preview': base, 'tokenizers': tokens}

    @staticmethod
    def shard(path, values):
        contents = array('H', values)
        if sys.byteorder != 'little':
            contents.byteswap()
        path.write_bytes(contents.tobytes())

    def run_cli(self):
        output = StringIO()
        with patch.dict(sys.modules, self.modules), patch.object(sys, 'argv', self.argv), redirect_stdout(output):
            overlap.main()
        return output.getvalue()

    def test_private_report_and_aggregate_only_stdout(self):
        printed = self.run_cli()
        report = json.loads((self.private / 'phoenix_overlap_toy.json').read_text())
        self.assertEqual(report['train_case_ids_with_exact_hits'], ['q0'])
        self.assertFalse(report['holdout_independence_verified'])
        self.assertEqual(json.loads(printed)['train_cases_with_hits'], 1)
        self.assertNotIn('Question', printed)
        self.assertNotIn(str(self.private), printed)
        self.assertNotIn('Question', json.dumps(report))

    def test_create_only_report_refuses_overwrite(self):
        first = self.run_cli()
        original = (self.private / 'phoenix_overlap_toy.json').read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'not be overwritten'):
            self.run_cli()
        self.assertEqual(original, (self.private / 'phoenix_overlap_toy.json').read_bytes())
        self.assertIn('private_exact_overlap_audit_complete', first)

    def test_missing_execute_refuses_before_private_input(self):
        self.argv.remove('--execute')
        with self.assertRaises(SystemExit) as error:
            self.run_cli()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.private / 'phoenix_overlap_toy.json').exists())

    def test_wrong_tokenizer_vocab_refused(self):
        self.modules['tokenizers'].Tokenizer = SimpleNamespace(
            from_file=lambda filename: SimpleNamespace(get_vocab_size=lambda: 99))
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            self.run_cli()
        self.assertFalse((self.private / 'phoenix_overlap_toy.json').exists())


if __name__ == '__main__':
    unittest.main()
