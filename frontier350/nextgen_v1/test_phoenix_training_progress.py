"""Synthetic log-contract tests: no actual Neral weights or training."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from phoenix_training_progress import summarize


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, rows, summary=None):
        (self.root / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        if summary is not None:
            (self.root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')

    def start(self, tokens=0, step=0, target=2048):
        return {'event': 'separate_weight_only_pilot', 'lineage': {'new_target_tokens': target},
                'new_processed_tokens': tokens, 'new_step': step}

    def step(self, tokens=512, step=1, elapsed=2):
        return {'event': 'step', 'new_tokens': tokens, 'new_step': step,
                'loss': 4.1, 'grad_norm': 1.2, 'lr': 0.00002, 'elapsed_s': elapsed}

    def test_no_file_does_not_claim_training(self):
        with self.assertRaises(ValueError): summarize(self.root)

    def test_new_real_log_receipt_progress_and_speed(self):
        self.write([self.start(), {'event':'validation','new_tokens':512,'new_step':1,'loss':4.4}, self.step()])
        result = summarize(self.root)
        self.assertEqual(result['new_tokens'], 512)
        self.assertEqual(result['progress_percent'], 25)
        self.assertEqual(result['observed_wall_tokens_per_second_including_overheads'], 256)
        self.assertEqual(result['last_validation_loss'], 4.4)
        self.assertFalse(result['quality_improvement_verified'])

    def test_resume_recalculates_invocation_speed_not_historical(self):
        self.write([self.start(), self.step(), self.start(512, 1), self.step(1024, 2, 4)])
        self.assertEqual(summarize(self.root)['observed_wall_tokens_per_second_including_overheads'], 128)

    def test_resume_must_match_previous_receipt(self):
        self.write([self.start(), self.step(), self.start(700, 1)])
        with self.assertRaisesRegex(ValueError, 'resume receipt'): summarize(self.root)

    def test_reject_fabricated_step_jump(self):
        self.write([self.start(), self.step(512, 2)])
        with self.assertRaisesRegex(ValueError, 'nonmonotonic'): summarize(self.root)

    def test_reject_nonfinite_loss(self):
        bad = self.step(); bad['loss'] = float('nan')
        self.write([self.start(), bad])
        with self.assertRaisesRegex(ValueError, 'invalid complete training receipt'): summarize(self.root)

    def test_reject_invalid_complete_json(self):
        (self.root/'metrics.jsonl').write_text('{bad}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'invalid complete'): summarize(self.root)

    def test_ignore_only_incomplete_final_line(self):
        self.write([self.start(), self.step()])
        with (self.root/'metrics.jsonl').open('a') as file: file.write('{"event":"step"')
        value = summarize(self.root)
        self.assertTrue(value['incomplete_last_log_line_ignored'])
        self.assertEqual(value['new_tokens'], 512)

    def test_valid_summary_reports_not_checkpoint_verification(self):
        self.write([self.start(target=512), self.step()],
                   {'event':'weight_only_pilot_finished','source_unchanged': True,
                    'new_processed_tokens':512,'new_optimizer_steps':1,
                    'resume_sha256':'a'*64})
        r = summarize(self.root)
        self.assertEqual(r['status'], 'summary_reported_checkpoint_not_verified')
        self.assertFalse(r['checkpoint_bytes_or_gpu_verified'])

    def test_summary_without_matching_step_fails(self):
        self.write([self.start(), self.step()],
                   {'event':'weight_only_pilot_finished','source_unchanged': True,
                    'new_processed_tokens':1024,'new_optimizer_steps':1,
                    'resume_sha256':'a'*64})
        with self.assertRaisesRegex(ValueError, 'completion receipt'): summarize(self.root)

    def test_stale_summary_on_resume_never_claims_completion(self):
        self.write([self.start(target=2048), self.step(), self.start(512, 1)],
                   {'event':'weight_only_pilot_finished','source_unchanged': True,
                    'new_processed_tokens':512,'new_optimizer_steps':1,
                    'resume_sha256':'a'*64})
        result = summarize(self.root)
        self.assertEqual(result['status'], 'in_progress_or_interrupted')
        self.assertTrue(result['stale_summary_ignored'])

    def test_target_change_rejected(self):
        self.write([self.start(), self.step(), self.start(512,1,target=4096)])
        with self.assertRaisesRegex(ValueError, 'target changed'): summarize(self.root)

    def test_run_dir_symlink_refused(self):
        link = self.root.parent / (self.root.name + '_link')
        link.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(link.unlink)
        with self.assertRaisesRegex(ValueError, 'symlink'): summarize(link)


if __name__ == '__main__': unittest.main()