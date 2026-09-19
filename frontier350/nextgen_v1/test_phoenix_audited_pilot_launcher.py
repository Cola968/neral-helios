"""Fabricated tiny shards and fake pilot only; never imports GPU or trains."""
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

    def test_launcher_cannot_bypass_original_approval(self):
        fake = SimpleNamespace(parser_args=lambda argv: self.args,
                               prepare=lambda args: (_ for _ in ()).throw(ValueError('unapproved')),
                               main=lambda argv: self.fail('started without approval'))
        with (patch.object(launch.importlib, 'import_module', return_value=fake),
              patch.object(launch, 'require_zero_exact_overlap', side_effect=AssertionError('gate should not run'))):
            with self.assertRaisesRegex(ValueError, 'unapproved'):
                launch.main([])

    def test_launcher_delegates_only_after_matching_preflight_and_no_overlap(self):
        calls = []
        fake = SimpleNamespace(parser_args=lambda argv: self.args,
                               prepare=lambda args: self.plan,
                               main=lambda argv: calls.append('training'))
        def gate(args, plan):
            calls.append('overlap')
            return {'exact_overlap_gate_passed': True}
        with (patch.object(launch.importlib, 'import_module', return_value=fake),
              patch.object(launch, 'require_zero_exact_overlap', side_effect=gate)):
            launch.main(['synthetic'])
        self.assertEqual(calls, ['overlap', 'training'])

    def test_launcher_refuses_overlap_without_calling_trainer(self):
        calls = []
        fake = SimpleNamespace(parser_args=lambda argv: self.args,
                               prepare=lambda args: self.plan,
                               main=lambda argv: calls.append('training'))
        with (patch.object(launch.importlib, 'import_module', return_value=fake),
              patch.object(launch, 'require_zero_exact_overlap', side_effect=ValueError('pilot refused'))):
            with self.assertRaisesRegex(ValueError, 'pilot refused'):
                launch.main(['synthetic'])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
