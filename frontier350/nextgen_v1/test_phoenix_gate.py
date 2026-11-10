"""CPU-only contract tests; fixtures do not constitute model-quality evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from phoenix_gate import compare, load


def fixture(checkpoint, answer='A short, useful answer.', reason='eos'):
    return {'schema': 1, 'checkpoint_sha256': checkpoint * 64,
            'protocol_sha256': 'a' * 64,
            'cases': [{'id': f'case-{i}', 'prompt_sha256': f'{i:064x}',
                       'answer': answer, 'finish_reason': reason,
                       'raw_model': True, 'tool_calls': 0,
                       'tokens_generated': 5, 'max_tokens': 128}
                      for i in range(8)]}


class PhoenixGateTests(unittest.TestCase):
    def setUp(self):
        self.before = fixture('b')
        self.after = fixture('c')

    def roundtrip(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'private.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            return load(path)

    def test_same_population_nonregression_never_promotes(self):
        result = compare(self.roundtrip(self.before), self.roundtrip(self.after))
        self.assertTrue(result['health_nonregression'])
        self.assertFalse(result['production_approved'])
        self.assertFalse(result['answer_correctness_evaluated'])
        self.assertNotIn('A short, useful answer.', json.dumps(result))

    def test_rejects_assisted_output(self):
        self.after['cases'][0]['tool_calls'] = 1
        with self.assertRaises(ValueError):
            self.roundtrip(self.after)

    def test_rejects_same_checkpoint(self):
        with self.assertRaises(ValueError):
            compare(self.before, self.before)

    def test_rejects_changed_protocol(self):
        self.after['protocol_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            compare(self.before, self.after)

    def test_rejects_changed_prompts(self):
        self.after['cases'][0]['prompt_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            compare(self.before, self.after)

    def test_rejects_changed_budget_even_if_protocol_claim_matches(self):
        self.after['cases'][0]['max_tokens'] = 256
        with self.assertRaises(ValueError):
            compare(self.before, self.after)

    def test_rejects_duplicate_case_ids(self):
        self.after['cases'][1]['id'] = 'case-0'
        with self.assertRaises(ValueError):
            self.roundtrip(self.after)

    def test_detects_eos_regression(self):
        self.after['cases'][0]['finish_reason'] = 'max_tokens'
        self.after['cases'][0]['tokens_generated'] = 128
        self.assertFalse(compare(self.roundtrip(self.before), self.roundtrip(self.after))['health_nonregression'])

    def test_detects_repetition(self):
        self.after['cases'][0]['answer'] = 'loop loop loop loop loop loop loop loop loop loop'
        self.assertFalse(compare(self.before, self.after)['health_nonregression'])

    def test_detects_role_bleed(self):
        self.after['cases'][0]['answer'] = 'Answer.\nUser: next invented message'
        self.assertFalse(compare(self.before, self.after)['health_nonregression'])

    def test_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.json'
            path.write_text('{"schema":1,"schema":1}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load(path)

    def test_rejects_falsified_eos_with_zero_generated_tokens(self):
        self.after['cases'][0]['answer'] = ''
        self.after['cases'][0]['tokens_generated'] = 0
        with self.assertRaises(ValueError):
            self.roundtrip(self.after)

    def test_first_token_eos_is_real_but_empty_answer_is_penalized(self):
        self.after['cases'][0]['answer'] = ''
        self.after['cases'][0]['tokens_generated'] = 1
        result = compare(self.roundtrip(self.before), self.roundtrip(self.after))
        self.assertEqual(result['candidate']['eos_rate'], 1.0)
        self.assertEqual(result['candidate']['nonempty_rate'], 0.875)
        self.assertFalse(result['health_nonregression'])

    def test_rejects_incomplete_max_token_claim(self):
        self.after['cases'][0]['finish_reason'] = 'max_tokens'
        with self.assertRaises(ValueError):
            self.roundtrip(self.after)

    def test_rejects_artificial_stop(self):
        self.after['cases'][0]['finish_reason'] = 'stop'
        with self.assertRaises(ValueError):
            self.roundtrip(self.after)


if __name__ == '__main__':
    unittest.main()
