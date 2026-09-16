"""Offline crash-fixture tests; no Torch, GPU or real checkpoint deserialization."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from checkpoint_journal import (CheckpointJournal, recover_to_new_file,
                                safe_save_every, sha256_file)
from train_full_with_journal import guarded_arguments


def state(tokens: int) -> dict:
    return {'processed_tokens': tokens, 'step': tokens // 16, 'model': {'weight': 1},
            'optimizer': {'state': {'weight': 1}}, 'torch_rng': 'cpu',
            'cuda_rng': 'gpu', 'tokenizer_sha256': 'native',
            'train_sampler_rng': 'sampler'}


def atomic_writer(value: dict, path: Path):
    temp = path.with_suffix('.pt.tmp')
    temp.write_text(json.dumps(value, sort_keys=True), encoding='utf8')
    os.replace(temp, path)


class CheckpointJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.resume = self.folder / 'resume.pt'
        self.resume.write_bytes(b'owned-original-complete-checkpoint')
        self.digest = sha256_file(self.resume)

    def journal(self, writer=atomic_writer):
        return CheckpointJournal(self.resume, initial_sha256=self.digest,
                                 initial_tokens=1_010_003_584, writer=writer)

    def test_cadence_full_model_defaults_below_one_million(self):
        self.assertEqual(safe_save_every(seq=512, microbatch=4, accumulation=8), 60)
        self.assertEqual(512 * 4 * 8 * 60, 983_040)
        self.assertEqual(safe_save_every(seq=512, microbatch=4, accumulation=12), 40)
        self.assertEqual(safe_save_every(seq=512, microbatch=4, accumulation=8, requested=20), 20)
        with self.assertRaises(ValueError):
            safe_save_every(seq=2048, microbatch=32, accumulation=16)
        with self.assertRaises(ValueError):
            safe_save_every(seq=512, microbatch=4, accumulation=8, max_rollback_tokens=1_000_001)
        with self.assertRaises(ValueError):
            safe_save_every(seq=512, microbatch=4, accumulation=8, requested=0)

    def test_commit_preserves_original_and_verifies_new_generation(self):
        journal = self.journal()
        source = self.folder / 'checkpoint_journal' / f'ckpt-1010003584-{self.digest}.pt'
        self.assertTrue(source.is_file())
        self.assertEqual(source.read_bytes(), b'owned-original-complete-checkpoint')
        receipt = journal.commit(state(1_010_003_584 + 983_040), self.resume)
        self.assertEqual(journal.verify(), receipt)
        self.assertEqual(source.read_bytes(), b'owned-original-complete-checkpoint')
        self.assertEqual(receipt['sha256'], sha256_file(self.resume))
        self.assertEqual(os.stat(self.resume).st_ino, os.stat(journal.root / receipt['name']).st_ino)
        journal.commit(state(1_010_003_584 + 983_040 + 983_040), self.resume)
        self.assertEqual(len(list(journal.root.glob('ckpt-*.pt'))), 3)

    def test_reject_unbounded_skip_without_writing(self):
        journal = self.journal()
        with self.assertRaisesRegex(ValueError, '1M'):
            journal.commit(state(1_010_003_584 + 1_000_001), self.resume)
        self.assertEqual(sha256_file(self.resume), self.digest)
        self.assertEqual(journal.verify()['processed_tokens'], 1_010_003_584)

    def test_reject_non_resumable_state(self):
        journal = self.journal()
        broken = state(1_010_003_585)
        broken.pop('train_sampler_rng')
        with self.assertRaisesRegex(ValueError, 'sampler'):
            journal.commit(broken, self.resume)
        broken = state(1_010_003_585)
        broken.pop('optimizer')
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            journal.commit(broken, self.resume)
        self.assertEqual(sha256_file(self.resume), self.digest)

    def test_atomic_writer_failure_retains_previous_checkpoint(self):
        def interrupt(_state, path):
            path.with_suffix('.pt.tmp').write_bytes(b'incomplete')
            raise OSError('simulated power loss before replacement')
        journal = self.journal(interrupt)
        with self.assertRaisesRegex(OSError, 'power loss'):
            journal.commit(state(1_010_003_585), self.resume)
        self.assertEqual(journal.verify()['sha256'], self.digest)
        self.assertEqual(self.resume.read_bytes(), b'owned-original-complete-checkpoint')

    def test_failure_after_replace_recovers_old_generation_non_destructively(self):
        def interrupt(value, path):
            atomic_writer(value, path)
            raise OSError('crash after atomic replace, before receipt')
        journal = self.journal(interrupt)
        with self.assertRaisesRegex(OSError, 'crash'):
            journal.commit(state(1_010_003_585), self.resume)
        self.assertNotEqual(sha256_file(self.resume), self.digest)
        with self.assertRaisesRegex(ValueError, 'pinned'):
            self.journal()
        restored = self.folder / 'restored.pt'
        receipt = recover_to_new_file(journal=journal.root, destination=restored)
        self.assertEqual(receipt['sha256'], self.digest)
        self.assertEqual(sha256_file(restored), self.digest)
        with self.assertRaises(FileExistsError):
            recover_to_new_file(journal=journal.root, destination=restored)

    def test_existing_journal_requires_exact_latest_resume(self):
        journal = self.journal()
        latest = journal.commit(state(1_010_003_585), self.resume)
        with self.assertRaises(ValueError):
            self.journal()
        continued = CheckpointJournal(self.resume, initial_sha256=latest['sha256'],
                                     initial_tokens=latest['processed_tokens'], writer=atomic_writer)
        self.assertEqual(continued.verify(), latest)

    def test_bad_generation_or_bad_initial_hash_rejected(self):
        journal = self.journal()
        with self.assertRaises(ValueError):
            CheckpointJournal(self.resume, initial_sha256='0' * 64,
                              initial_tokens=1_010_003_584, writer=atomic_writer)
        receipt = journal.verify()
        (journal.root / receipt['name']).unlink()
        with self.assertRaisesRegex(ValueError, 'absent or corrupt'):
            journal.verify()

    def test_wrapper_enforces_same_folder_and_overrides_unsafe_interval(self):
        args, path, window = guarded_arguments(
            ['--', '--out', str(self.folder), '--resume', str(self.resume),
             '--save-every', '100', '--seq', '512', '--microbatch', '4', '--grad-accum', '8'],
            max_rollback=1_000_000)
        self.assertEqual(path, self.resume.absolute())
        self.assertEqual(int(window), 983_040)
        self.assertEqual(args[-2:], ['--save-every', '60'])
        with self.assertRaises(ValueError):
            guarded_arguments(['--', '--out', str(self.folder), '--resume', str(self.folder / 'different.pt')],
                              max_rollback=1_000_000)
        with self.assertRaises(ValueError):
            guarded_arguments(['--', '--out', str(self.folder), '--resume', str(self.resume),
                               '--save-every', '5', '--save-every', '50'], max_rollback=1_000_000)


if __name__ == '__main__':
    unittest.main()
