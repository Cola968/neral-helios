"""Synthetic grading and unblinding contract tests; no real model answers."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phoenix_blind_ab_audit import ORIGINAL_SHA, prepare
from phoenix_blind_ab_unblind import summarize, main
from unittest.mock import patch


class UnblindTests(unittest.TestCase):
    def setUp(self):
        holdout = {'schema': 1, 'cases': [
            {'id': f'case_{i}', 'language': 'de' if i % 2 else 'en',
             'domain': 'synthetic', 'prompt': f'Synthetic prompt {i}?',
             'rubric': 'A human must check exact answer'} for i in range(8)]}
        self.holdout_sha = 'c' * 64

        def report(sha, answer):
            return {'schema': 1, 'checkpoint_sha256': sha,
                    'tokenizer_sha256': 'e' * 64, 'holdout_sha256': self.holdout_sha,
                    'max_new_tokens': 16, 'decode': {'method': 'greedy'},
                    'cases': [{'id': c['id'],
                               'prompt_sha256': hashlib.sha256(c['prompt'].encode()).hexdigest(),
                               'input_ids_sha256': hashlib.sha256(c['id'].encode()).hexdigest(),
                               'answer': answer, 'generated_tokens': 2,
                               'finish_reason': 'eos'} for c in holdout['cases']]}
        self.packet, self.key = prepare(holdout, self.holdout_sha,
                                        report(ORIGINAL_SHA, 'old'),
                                        report('b' * 64, 'new'))
        self.packet_sha = hashlib.sha256(json.dumps(self.packet).encode()).hexdigest()
        self.grades = {'schema': 1, 'packet_sha256': self.packet_sha, 'items': [
            {'id': item['id'], 'A_correct': i % 2 == 0,
             'B_correct': i % 2 == 1, 'notes': 'Synthetic grading only'}
            for i, item in enumerate(self.packet['items'])]}

    def calc(self):
        return summarize(self.packet, self.packet_sha, self.grades, 'f' * 64, self.key)

    def test_paired_counts_balance(self):
        scored = self.calc()
        self.assertEqual(scored['n'], 8)
        self.assertEqual(scored['baseline_correct'] + scored['candidate_correct'], 8)
        self.assertEqual(scored['baseline_only'] + scored['candidate_only'], 8)
        self.assertEqual(scored['both_wrong'], 0)
        self.assertEqual(scored['both_correct'], 0)
        self.assertEqual(sum(group['n'] for group in scored['groups']['language'].values()), 8)
        self.assertFalse(scored['quality_improvement_proven'])
        self.assertNotIn('prompt', str(scored))
        self.assertNotIn('Synthetic grading', str(scored))

    def test_reject_tampered_packet(self):
        self.grades['packet_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'not pinned'):
            self.calc()

    def test_reject_missing_grading(self):
        self.grades['items'].pop()
        with self.assertRaisesRegex(ValueError, 'do not exactly match'):
            self.calc()

    def test_reject_duplicate_case(self):
        self.grades['items'][1]['id'] = self.grades['items'][0]['id']
        with self.assertRaisesRegex(ValueError, 'human grade'):
            self.calc()

    def test_reject_numeric_as_boolean(self):
        self.grades['items'][0]['A_correct'] = 1
        with self.assertRaisesRegex(ValueError, 'human grade'):
            self.calc()

    def test_reject_unexplained_grade(self):
        self.grades['items'][0]['notes'] = ' '
        with self.assertRaisesRegex(ValueError, 'human grade'):
            self.calc()

    def test_reject_corrupt_key(self):
        self.key['case_mapping'][0]['A'] = 'candidate'
        self.key['case_mapping'][0]['B'] = 'candidate'
        with self.assertRaisesRegex(ValueError, 'mapping inconsistent'):
            self.calc()

    def test_reject_changed_candidate_origin(self):
        self.key['candidate_checkpoint_sha256'] = ORIGINAL_SHA
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.calc()

    def test_real_private_file_flow_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'NeralPrivateAssets' / 'phoenix'
            root.mkdir(parents=True)
            paths = []
            for name, data in (('packet.json', self.packet),
                               ('grades.json', self.grades), ('key.json', self.key)):
                p = root / name
                p.write_text(json.dumps(data), encoding='utf-8')
                paths.append(p)
            # Hash exact bytes used by the CLI, not the guessed formatting.
            self.grades['packet_sha256'] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
            paths[1].write_text(json.dumps(self.grades), encoding='utf-8')
            argv = ['grade', '--execute', '--packet', str(paths[0]), '--grades',
                    str(paths[1]), '--secret-key', str(paths[2]),
                    '--private-dir', str(root), '--tag', 'synthetic']
            with patch('sys.argv', argv):
                main()
            result = root / 'phoenix_scored_synthetic.json'
            self.assertTrue(result.is_file())
            self.assertEqual(json.loads(result.read_text())['n'], 8)
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                main()


if __name__ == '__main__':
    unittest.main()
