"""Offline crash-fixture tests; no Torch, GPU or real checkpoint deserialization."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from checkpoint_journal import (CheckpointJournal, recover_to_new_file,
                                safe_save_every, sha256_file)
from train_full_with_journal import guarded_arguments, duplicate_terminal_save


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
        self.assertEqual(safe_save_every(seq=512, microbatch=4, accumulation=8), 61)
        self.assertEqual(512 * 4 * 8 * 61, 999_424)
        self.assertEqual(safe_save_every(seq=512, microbatch=4, accumulation=12), 40)
        self.assertEqual(512 * 4 * 12 * 40, 983_040)
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

    def test_retention_keeps_latest_three_and_records_retired_metadata(self):
        journal = self.journal()
        base = 1_010_003_584
        for offset in (100_000, 200_000, 300_000, 400_000):
            journal.commit(state(base + offset), self.resume)
        generations = list(journal.root.glob('ckpt-*.pt'))
        self.assertEqual(len(generations), 3)
        latest = journal.verify()
        self.assertEqual(latest['processed_tokens'], base + 400_000)
        self.assertTrue((journal.root / latest['name']).is_file())
        retired = (journal.root / 'retired.jsonl').read_text(encoding='utf8').splitlines()
        self.assertEqual(len(retired), 2)
        records = [json.loads(line) for line in retired]
        self.assertEqual([row['processed_tokens'] for row in records],
                         [base, base + 100_000])
        self.assertFalse(any((journal.root / row['name']).exists() for row in records))

    def test_quarantine_stale_temp_only_after_verified_latest(self):
        journal = self.journal()
        next_state = state(1_010_003_584 + 100_000)
        journal.commit(next_state, self.resume)
        temporary = self.resume.with_name(self.resume.name + '.tmp')
        temporary.write_bytes(b'interrupted-write-bytes')
        temp_sha = sha256_file(temporary)
        result = quarantine_stale_resume_temp(self.resume)
        self.assertTrue(result['quarantined'])
        self.assertEqual(result['temp_sha256'], temp_sha)
        self.assertFalse(temporary.exists())
        quarantine = self.resume.parent / result['quarantine_name']
        self.assertTrue(quarantine.is_file())
        self.assertEqual(sha256_file(quarantine), temp_sha)
        self.assertEqual(journal.verify()['sha256'], sha256_file(self.resume))

    def test_quarantine_refuses_when_resume_differs_from_latest(self):
        journal = self.journal()
        next_state = state(1_010_003_584 + 100_000)
        journal.commit(next_state, self.resume)
        temporary = self.resume.with_name(self.resume.name + '.tmp')
        temporary.write_bytes(b'interrupted-write-bytes')
        self.resume.write_bytes(b'corrupted-resume')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            quarantine_stale_resume_temp(self.resume)
        self.assertTrue(temporary.exists())

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
        self.assertEqual(int(window), 999_424)
        self.assertEqual(args[-2:], ['--save-every', '61'])
        with self.assertRaises(ValueError):
            guarded_arguments(['--', '--out', str(self.folder), '--resume', str(self.folder / 'different.pt')],
                              max_rollback=1_000_000)
        with self.assertRaises(ValueError):
            guarded_arguments(['--', '--out', str(self.folder), '--resume', str(self.resume),
                               '--save-every', '5', '--save-every', '50'], max_rollback=1_000_000)

    def test_redundant_final_save_does_not_count_as_new_progress(self):
        self.assertTrue(duplicate_terminal_save({'processed_tokens': 999, 'step': 12},
                                                last_tokens=999, last_step=12))
        self.assertFalse(duplicate_terminal_save({'processed_tokens': 1000, 'step': 13},
                                                 last_tokens=999, last_step=12))
        with self.assertRaisesRegex(ValueError, 'changed optimizer step'):
            duplicate_terminal_save({'processed_tokens': 999, 'step': 13},
                                    last_tokens=999, last_step=12)


if __name__ == '__main__':
    unittest.main()
