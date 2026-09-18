"""Regression: never publish a model-only checkpoint as fully resumable."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from checkpoint_journal import CheckpointJournal, sha256_file


class NoFalseResumeReceipt(unittest.TestCase):
    def test_empty_optimizer_state_rejected_before_writer_or_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            resume = Path(temp) / 'resume.pt'
            resume.write_bytes(b'synthetic-original-checkpoint')
            before = sha256_file(resume)
            writes = []

            def writer(state, target):
                writes.append(target)
                raise AssertionError('writer must never be called for empty optimizer')

            journal = CheckpointJournal(
                resume, initial_sha256=before, initial_tokens=100,
                writer=writer, max_rollback_tokens=1_000_000)
            original = journal.verify()
            self.assertEqual(original['kind'], 'pinned_initial_structure_unverified')
            synthetic = {
                'processed_tokens': 101, 'step': 10, 'model': {'synthetic': 1},
                'optimizer': {'state': {}}, 'torch_rng': 'synthetic',
                'cuda_rng': 'synthetic', 'train_sampler_rng': 'synthetic',
                'tokenizer_sha256': 'a' * 64,
            }
            with self.assertRaisesRegex(ValueError, 'nonempty AdamW optimizer moments'):
                journal.commit(synthetic, resume)
            self.assertEqual(writes, [])
            self.assertEqual(sha256_file(resume), before)
            self.assertEqual(journal.verify(), original)
            self.assertEqual(len(list(journal.root.glob('ckpt-*.pt'))), 1)

    def test_missing_state_field_also_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            resume = Path(temp) / 'resume.pt'
            resume.write_bytes(b'original')
            journal = CheckpointJournal(resume, initial_sha256=sha256_file(resume),
                                        initial_tokens=50, writer=lambda *_: None)
            broken = {'processed_tokens': 51, 'model': {'synthetic': 1},
                      'optimizer': {}, 'step': 2, 'torch_rng': 1,
                      'cuda_rng': 1, 'train_sampler_rng': 1, 'tokenizer_sha256': 'a'*64}
            with self.assertRaisesRegex(ValueError, 'nonempty AdamW'):
                journal.commit(broken, resume)
            self.assertEqual(journal.verify()['processed_tokens'], 50)


if __name__ == '__main__':
    unittest.main()
