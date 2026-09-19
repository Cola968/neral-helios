"""Fabricated tiny shards and mocked native pilot; no GPU or model weights."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import phoenix_audited_pilot_launcher as launch
import phoenix_shard_distribution_audit as audit


class AuditedLaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.train = root / 'train'
        self.valid = root / 'valid'
        self.train.mkdir()
        self.valid.mkdir()
        (self.train / 'a.bin').write_bytes((1).to_bytes(2, 'little') * 10)
        (self.valid / 'b.bin').write_bytes((2).to_bytes(2, 'little') * 10)
        self.args = SimpleNamespace(train_data=self.train, valid_data=self.valid,
                                    seq=4, microbatch=1, eval_batches=2)
        self.plan = {'train_files': [self.train / 'a.bin'],
                     'valid_files': [self.valid / 'b.bin'],
                     'contract': {'seed': 35001}}
        self.plan['train_id'] = launch.corpus_digest(
            self.plan['train_files'], audit.inspect_shards(self.train, 4),
            seq=4, role='train', seed=35002)
        self.plan['val_id'] = launch.corpus_digest(
            self.plan['valid_files'], audit.inspect_shards(self.valid, 4),
            seq=4, role='valid', seed=36000, batch=1, batches=2)

    def test_matching_real_tiny_shard_bytes_pass(self):
        result = launch.verify_snapshot(self.args, self.plan)
        self.assertEqual(result['train']['valid_windows'], 6)
        self.assertTrue(result['all_token_ids_in_vocab'])

    def test_reject_changed_bytes_even_when_length_identical(self):
        (self.train / 'a.bin').write_bytes((3).to_bytes(2, 'little') * 10)
        with self.assertRaisesRegex(ValueError, 'audited shard bytes differ'):
            launch.verify_snapshot(self.args, self.plan)

    def test_reject_invalid_vocab_before_pilot_invocation(self):
        data = bytearray((self.train / 'a.bin').read_bytes())
        data[-2:] = (65535).to_bytes(2, 'little')
        (self.train / 'a.bin').write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'out-of-vocabulary'):
            launch.verify_snapshot(self.args, self.plan)

    def test_reject_mismatched_approval_corpus_identity(self):
        self.plan['val_id'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'approved native identities'):
            launch.verify_snapshot(self.args, self.plan)

    def test_wrapper_delegates_once_without_repeating_scans(self):
        calls = []
        fake = SimpleNamespace(main=lambda argv: calls.append(argv))
        with (patch.object(launch.importlib, 'import_module', return_value=fake),
              patch.object(launch, 'verify_snapshot', side_effect=AssertionError('double scan'))):
            launch.main(['synthetic'])
        self.assertEqual(calls, [['synthetic']])

    def test_wrapper_forwards_resume_arguments_unchanged(self):
        calls = []
        fake = SimpleNamespace(main=lambda argv: calls.append(argv))
        args = ['--resume', '/synthetic/resume.pt', '--resume-sha256', 'a' * 64]
        with patch.object(launch.importlib, 'import_module', return_value=fake):
            launch.main(args)
        self.assertEqual(calls, [args])

    def test_wrapper_preserves_original_approval_refusal(self):
        def native(argv):
            raise ValueError('unapproved')
        fake = SimpleNamespace(main=native)
        with patch.object(launch.importlib, 'import_module', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'unapproved'):
                launch.main([])

    def test_wrapper_preserves_native_holdout_refusal(self):
        def native(argv):
            raise ValueError('frozen holdout exact-token overlap')
        fake = SimpleNamespace(main=native)
        with patch.object(launch.importlib, 'import_module', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'exact-token overlap'):
                launch.main(['synthetic'])


if __name__ == '__main__':
    unittest.main()
