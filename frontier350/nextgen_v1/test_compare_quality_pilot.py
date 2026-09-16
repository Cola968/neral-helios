"""CPU-only synthetic checks; no model weights, GPU or inference required."""
import copy
import unittest

from compare_quality_pilot import compare

A = 'a' * 64
B = 'b' * 64
START = 1_010_003_584
EXTRA = 491_520


def fixtures():
    base_dev = {'checkpoint_sha256': A, 'checkpoint_processed_tokens': START,
                'suite': 'dev-v1', 'protocol_sha256': 'p' * 64,
                'tokenizer_sha256': 't' * 64, 'architecture': 'frontier350-v1',
                'cases': 20, 'overall_accuracy_pct': 60.0,
                'categories': {'math': {'total': 10, 'accuracy_pct': 50.0},
                               'language': {'total': 10, 'accuracy_pct': 70.0}}}
    cand_dev = copy.deepcopy(base_dev)
    cand_dev.update(checkpoint_sha256=B, checkpoint_processed_tokens=START+EXTRA,
                    overall_accuracy_pct=65.0)
    cand_dev['categories']['math']['accuracy_pct'] = 60.0
    base_nis = {'checkpoint_processed_tokens': START, 'benchmark': 'chat-v2',
                'suite_sha256': 's' * 64, 'architecture': 'frontier350-v1',
                'parameters': 349_825_152, 'case_counts': {'total': 140},
                'full_nis2': {'score': None},
                'scores': {'pair_overall_pct': 60.0,
                           'instruction_following_pct': 45.0,
                           'generation_health_degenerative_pct': 20.0,
                           'generation_health_repeated_trigram_pct': 10.0}}
    cand_nis = copy.deepcopy(base_nis)
    cand_nis['checkpoint_processed_tokens'] = START+EXTRA
    cand_nis['scores']['pair_overall_pct'] = 62.0
    train = {'complete': True, 'new_tokens': EXTRA, 'processed_tokens': START+EXTRA,
             'checkpoint_sha256': B, 'updates': 20, 'frozen_weights_unchanged': True,
             'promoted': False}
    return base_dev, cand_dev, base_nis, cand_nis, train


def run(data):
    return compare(*data, anchor_sha=A, candidate_sha=B,
                   start_tokens=START, extra_tokens=EXTRA)


class QualityPilotGuardTests(unittest.TestCase):
    def test_comparison_never_promotes(self):
        report = run(fixtures())
        self.assertEqual(report['status'], 'comparison_complete')
        self.assertTrue(report['development']['no_regression_in_this_dev_suite'])
        self.assertFalse(report['promotion_allowed'])
        self.assertFalse(report['nis2_chat_core_diagnostic_only']['canonical_full_nis2_verified'])
        self.assertEqual(report['nis2_chat_core_diagnostic_only']['metrics']['pair_overall_pct']['delta_candidate_minus_safe'], 2.0)

    def test_category_regression_is_reported(self):
        data = fixtures()
        data[1]['categories']['language']['accuracy_pct'] = 60.0
        report = run(data)
        self.assertFalse(report['development']['no_regression_in_this_dev_suite'])
        self.assertFalse(report['promotion_allowed'])

    def test_reject_changed_dev_protocol(self):
        data = fixtures()
        data[1]['protocol_sha256'] = 'x' * 64
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            run(data)

    def test_reject_wrong_candidate_checkpoint(self):
        data = fixtures()
        data[4]['checkpoint_sha256'] = 'z' * 64
        with self.assertRaisesRegex(ValueError, 'identity/update mismatch'):
            run(data)

    def test_reject_incomplete_training(self):
        data = fixtures()
        data[4]['complete'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            run(data)

    def test_reject_nis2_suite_change(self):
        data = fixtures()
        data[3]['suite_sha256'] = 'x' * 64
        with self.assertRaisesRegex(ValueError, 'NIS2 chat-core protocol mismatch'):
            run(data)

    def test_reject_claim_of_full_nis2_score(self):
        data = fixtures()
        data[3]['full_nis2']['score'] = 70
        with self.assertRaisesRegex(ValueError, 'canonical NIS2'):
            run(data)

    def test_reject_nonfinite_score(self):
        data = fixtures()
        data[1]['overall_accuracy_pct'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            run(data)

    def test_reject_frozen_weight_mutation(self):
        data = fixtures()
        data[4]['frozen_weights_unchanged'] = False
        with self.assertRaisesRegex(ValueError, 'frozen-weight'):
            run(data)


if __name__ == '__main__':
    unittest.main()
