"""Synthetic contract tests. Do not mistake fixtures for an independent holdout."""
import hashlib
import unittest

import phoenix_blind_ab_audit as audit


class BlindABTests(unittest.TestCase):
    def setUp(self):
        self.holdout = {'schema': 1, 'cases': [
            {'id': f'case_{i}', 'language': 'de' if i % 2 else 'en',
             'domain': 'synthetic', 'prompt': f'Synthetic prompt {i}?',
             'rubric': 'Synthetic human rubric'} for i in range(8)]}
        self.sha = 'a' * 64
        self.base = self.report(audit.ORIGINAL_SHA, 'original')
        self.candidate = self.report('b' * 64, 'candidate')

    def report(self, checkpoint, answer):
        return {'schema': 1, 'checkpoint_sha256': checkpoint,
                'tokenizer_sha256': 'c' * 64, 'holdout_sha256': self.sha,
                'max_new_tokens': 16, 'decode': {'temperature': 0},
                'cases': [{'id': c['id'],
                           'prompt_sha256': hashlib.sha256(c['prompt'].encode()).hexdigest(),
                           'input_ids_sha256': hashlib.sha256(c['id'].encode()).hexdigest(),
                           'answer': answer, 'generated_tokens': 1,
                           'finish_reason': 'eos'} for c in self.holdout['cases']]}

    def test_blind_packet_and_secret_key(self):
        packet, key = audit.prepare(self.holdout, self.sha, self.base, self.candidate)
        self.assertEqual(len(packet['items']), 8)
        self.assertEqual(len(key['case_mapping']), 8)
        self.assertNotIn('checkpoint_sha256', str(packet))
        for row, mapping in zip(packet['items'], key['case_mapping']):
            self.assertEqual(row['A'], self.base['cases'][0]['answer'] if mapping['A'] == 'baseline' else self.candidate['cases'][0]['answer'])
            self.assertNotEqual(mapping['A'], mapping['B'])

    def test_reject_changed_input(self):
        self.candidate['cases'][0]['input_ids_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'different tokenized input'):
            audit.prepare(self.holdout, self.sha, self.base, self.candidate)

    def test_reject_changed_holdout(self):
        self.candidate['holdout_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'different holdout'):
            audit.prepare(self.holdout, self.sha, self.base, self.candidate)

    def test_reject_same_weights(self):
        self.candidate['checkpoint_sha256'] = audit.ORIGINAL_SHA
        with self.assertRaisesRegex(ValueError, 'different checkpoint'):
            audit.prepare(self.holdout, self.sha, self.base, self.candidate)

    def test_reject_changed_decoding(self):
        self.candidate['decode']['temperature'] = 0.7
        with self.assertRaisesRegex(ValueError, 'setting mismatch'):
            audit.prepare(self.holdout, self.sha, self.base, self.candidate)

    def test_reject_duplicate_prompt(self):
        self.holdout['cases'][1]['prompt'] = self.holdout['cases'][0]['prompt'].swapcase()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.prepare(self.holdout, self.sha, self.base, self.candidate)

    def test_reject_boolean_token_count(self):
        self.candidate['cases'][0]['generated_tokens'] = True
        with self.assertRaisesRegex(ValueError, 'token count'):
            audit.prepare(self.holdout, self.sha, self.base, self.candidate)


if __name__ == '__main__':
    unittest.main()
