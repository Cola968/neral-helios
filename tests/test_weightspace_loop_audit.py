"""Offline failure-profiling contracts: never run/download models or change weights."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_weightspace_loop_audit import (
    EOS_QWEN25, AuditError, assess_loop_signal, read_audit,
    repetition_fraction,
)


H = hashlib.sha256(b'fixture').hexdigest()
R = 'a' * 40
B = 'Qwen/Qwen2.5-Coder-7B-Instruct'


def row(i, *, token_ids=None, reason='eos', budget=48, **overrides):
    if token_ids is None:
        token_ids = list(range(i + 1, i + 11)) + [EOS_QWEN25]
    result = {
        'task_id': f'case-{i:03}', 'base_model_id': B, 'base_revision': R,
        'prompt_sha256': hashlib.sha256(f'prompt{i}'.encode()).hexdigest(),
        'tokenizer_sha256': H, 'generation_config_sha256': H,
        'adapter_sha256': None, 'max_new_tokens': budget,
        'finish_reason': reason, 'output_token_ids': token_ids,
    }
    result.update(overrides)
    return result


class LoopAuditTests(unittest.TestCase):
    def test_repetition_score_exactness(self):
        self.assertEqual(repetition_fraction([1, 2, 3], n=4), 0.0)
        self.assertEqual(repetition_fraction([1, 2, 3, 4, 5], n=4), 0.0)
        self.assertGreater(repetition_fraction([1, 2, 3, 4] * 12, n=4), .8)

    def test_length_and_repeated_ngrams_flag_only_a_review_candidate(self):
        looping = [1, 2, 3, 4] * 12
        entries = [row(i) for i in range(22)] + [row(40 + i, token_ids=looping, reason='length') for i in range(5)]
        report = assess_loop_signal(entries)
        self.assertEqual(report['review_candidates'], 5)
        self.assertEqual(report['length_limited'], 5)
        self.assertEqual(report['decision'], 'manual_loop_review_required')
        self.assertNotIn('model_improved', json.dumps(report))
        self.assertEqual(report['recommendation'], 'review_only_no_weight_edit')

    def test_token_limit_without_repeat_is_not_proof_of_loop(self):
        entries = [row(i, token_ids=list(range(i + 1, i + 49)), reason='length') for i in range(22)]
        report = assess_loop_signal(entries)
        self.assertEqual(report['length_limited'], 22)
        self.assertEqual(report['review_candidates'], 0)
        self.assertEqual(report['decision'], 'no_repeated_loop_signal')

    def test_few_cases_are_not_sufficient_for_inference(self):
        report = assess_loop_signal([row(1), row(2)])
        self.assertEqual(report['decision'], 'insufficient_cases')
        self.assertEqual(report['n_cases'], 2)

    def test_cancelled_and_failed_are_reported_not_successfully_scored(self):
        entries = [row(1, token_ids=[1, 2], reason='cancelled'), row(2, token_ids=[], reason='failed')]
        report = assess_loop_signal(entries)
        self.assertEqual(report['interrupted'], 2)
        self.assertEqual(report['n_eligible'], 0)
        self.assertEqual(report['decision'], 'insufficient_cases')

    def test_duplicate_and_mixed_model_versions_fail_closed(self):
        with self.assertRaisesRegex(AuditError, 'duplicate'):
            assess_loop_signal([row(1), row(1)])
        with self.assertRaisesRegex(AuditError, 'mixed'):
            assess_loop_signal([row(1), row(2, base_revision='b'*40)])

    def test_no_raw_text_or_prompts_allowed_in_baseline_records(self):
        for extra in ({'text': 'private'}, {'prompt': 'private'}, {'tool_calls': 1}):
            with self.subTest(extra=extra), self.assertRaisesRegex(AuditError, 'fields'):
                assess_loop_signal([row(1, **extra)])

    def test_bad_eos_and_token_budget_are_rejected(self):
        bad = [
            row(1, token_ids=[1, 2, 3], reason='eos'),
            row(1, token_ids=[1, 2, 3], reason='length'),
            row(1, token_ids=[1, EOS_QWEN25], reason='length', budget=2),
            row(1, output_token_ids=[1, True, EOS_QWEN25]),
        ]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(AuditError):
                assess_loop_signal([item])

    def test_jsonl_parse_and_private_data_not_returned(self):
        data = '\n'.join(json.dumps(row(i)) for i in range(4)) + '\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'audit.jsonl')
            path.write_text(data, encoding='utf-8')
            report = assess_loop_signal(read_audit(path))
            self.assertEqual(report['n_cases'], 4)
            self.assertNotIn('output_token_ids', report)
            self.assertNotIn('prompt_sha256', json.dumps(report))

    def test_rejects_extra_large_and_bad_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'bad.jsonl')
            path.write_text('{"task_id": 1, "task_id": 2}\n', encoding='utf-8')
            with self.assertRaises(AuditError):
                read_audit(path)
            path.write_bytes(b'0' * (2_097_153))
            with self.assertRaises(AuditError):
                read_audit(path)


if __name__ == '__main__':
    unittest.main()
