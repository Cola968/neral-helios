"""Synthetic CPU-only contracts; no real weights or Torch imports required."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoenix_continuity_cpu_probe import (EXPECTED_TOKENS, check_metadata,
                                          require_pinned_checkpoint)
from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA


class ContinuityCPUProbeContracts(unittest.TestCase):
    def test_token_target_and_pinned_source_exist(self):
        self.assertEqual(EXPECTED_TOKENS, 2_073_161_216)
        self.assertEqual(len(CONTINUITY_SHA), 64)

    def test_missing_and_relative_checkpoint_refused(self):
        with self.assertRaises(ValueError):
            require_pinned_checkpoint(Path('relative.pt'))
        with self.assertRaises(ValueError):
            require_pinned_checkpoint(Path('/nonexistent-2b.pt'))

    def test_mismatch_rejected_without_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'own.pt'
            path.write_bytes(b'wrong synthetic bytes only')
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CONTINUITY_SHA)
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                require_pinned_checkpoint(path)

    def test_matching_source_hash_can_be_checked_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'own.pt'
            path.write_bytes(b'synthetic only')
            with patch('phoenix_continuity_cpu_probe.CONTINUITY_SHA', hashlib.sha256(b'synthetic only').hexdigest()):
                self.assertEqual(require_pinned_checkpoint(path), path)

    def test_checkpoint_metadata_must_match_architecture_tokenizer_and_tokens(self):
        config = {'vocab': 16000}
        valid = {'config': config, 'tokenizer_sha256': TOKENIZER_SHA,
                 'processed_tokens': EXPECTED_TOKENS, 'model': {'weight': 1}}
        check_metadata(valid, config)
        for changed in ({'processed_tokens': EXPECTED_TOKENS - 1},
                        {'processed_tokens': True}, {'tokenizer_sha256': 'wrong'},
                        {'config': {}}, {'model': None}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    check_metadata({**valid, **changed}, config)


if __name__ == '__main__':
    unittest.main()
