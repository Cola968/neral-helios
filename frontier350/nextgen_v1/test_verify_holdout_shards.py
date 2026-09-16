"""Synthetic shard fixtures only; no actual Neral data or GPU."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_holdout_shards import inspect


class HoldoutIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.valid = self.root / 'valid'
        self.valid.mkdir()
        self.shard = self.valid / 'valid_000.bin'
        self.shard.write_bytes(b'\0\0' * 70000)
        self.tok_sha = 'a' * 64
        self.manifest = {'benchmark_content_included': False,
                         'tokenizer': {'tokenizer_sha256': self.tok_sha},
                         'validation_tokens': 70000,
                         'valid_files': [{'path': self.shard.name, 'tokens': 70000,
                                          'sha256': hashlib.sha256(self.shard.read_bytes()).hexdigest()}]}
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf8')

    def test_admitted_holdout_receipt_is_not_training(self):
        receipt = inspect(self.root, self.tok_sha)
        self.assertEqual(receipt['validation_tokens'], 70000)
        self.assertTrue(receipt['holdout_only'])
        self.assertFalse(receipt['model_training_started'])
        self.assertFalse(receipt['full_nis2_verified'])

    def test_modified_shard_rejected(self):
        self.shard.write_bytes(b'\1\0' * 70000)
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            inspect(self.root, self.tok_sha)

    def test_unlisted_file_rejected(self):
        (self.valid / 'extra.bin').write_bytes(b'\0\0' * 80000)
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            inspect(self.root, self.tok_sha)

    def test_wrong_tokenizer_or_unattested_separation(self):
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            inspect(self.root, 'b' * 64)
        self.manifest['benchmark_content_included'] = None
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'separation'):
            inspect(self.root, self.tok_sha)

    def test_traversal_and_duplicate_names_rejected(self):
        self.manifest['valid_files'][0]['path'] = '../escape.bin'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'basename'):
            inspect(self.root, self.tok_sha)
        self.manifest['valid_files'][0]['path'] = self.shard.name
        self.manifest['valid_files'].append(dict(self.manifest['valid_files'][0]))
        self.manifest['validation_tokens'] = 140000
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inspect(self.root, self.tok_sha)

    def test_manifest_duplicate_json_keys_rejected(self):
        (self.root / 'manifest.json').write_text('{"benchmark_content_included":false,"benchmark_content_included":false}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inspect(self.root, self.tok_sha)

    def test_symlink_shard_rejected(self):
        other = self.root / 'untrusted.bin'
        other.write_bytes(self.shard.read_bytes())
        self.shard.unlink()
        try:
            self.shard.symlink_to(other)
        except (OSError, NotImplementedError):
            self.skipTest('symlinks unavailable')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            inspect(self.root, self.tok_sha)


if __name__ == '__main__':
    unittest.main()
