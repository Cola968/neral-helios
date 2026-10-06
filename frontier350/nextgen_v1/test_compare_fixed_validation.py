"""Synthetic-only validator contracts; no real checkpoint data or torch imports."""
import unittest

from compare_fixed_validation import assess

BASE_SHA = 'a' * 64
CAND_SHA = 'b' * 64


def report(sha, tokens, loss):
    return dict(schema=1, architecture='frontier350-v1', checkpoint_sha256=sha,
                checkpoint_processed_tokens=tokens, tokenizer_sha256='t' * 64,
                protocol_sha256='c' * 64, seq=512, batch=4, batches=32, seed=351337,
                validation_loss=loss, validation_perplexity=2.718281828 ** loss)


class FixedValidationTests(unittest.TestCase):
    def compare(self, base=None, candidate=None):
        return assess(base or report(BASE_SHA, 100, 4.5),
                      candidate or report(CAND_SHA, 120, 4.3),
                      anchor_sha=BASE_SHA, candidate_sha=CAND_SHA,
                      anchor_tokens=100, candidate_tokens=120)

    def test_improvement_does_not_promote(self):
        comparison = self.compare()
        self.assertTrue(comparison['loss_not_worse_on_this_fixed_sample'])
        self.assertFalse(comparison['promotion_allowed'])
        self.assertFalse(comparison['full_nis2_verified'])
        self.assertEqual(comparison['validation_tokens_evaluated_per_model'], 65536)

    def test_regression_is_reported_without_promotion(self):
        comparison = self.compare(candidate=report(CAND_SHA, 120, 4.8))
        self.assertFalse(comparison['loss_not_worse_on_this_fixed_sample'])
        self.assertGreater(comparison['loss_delta_candidate_minus_safe'], 0)
        self.assertFalse(comparison['promotion_allowed'])

    def test_modified_validation_protocol_rejected(self):
        other = report(CAND_SHA, 120, 4.3)
        other['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.compare(candidate=other)
        other['seed'] -= 1
        other['protocol_sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'protocol_sha256'):
            self.compare(candidate=other)

    def test_checkpoint_hash_and_token_identity_rejected(self):
        other = report(CAND_SHA, 121, 4.3)
        with self.assertRaisesRegex(ValueError, 'checkpoint digest or token'):
            self.compare(candidate=other)
        other = report(CAND_SHA, 120, 4.3)
        other['checkpoint_sha256'] = BASE_SHA
        with self.assertRaisesRegex(ValueError, 'checkpoint digest or token'):
            self.compare(candidate=other)

    def test_nonfinite_and_missing_metrics_rejected(self):
        for bad in (float('nan'), float('inf'), -2.0, None, '4.3', True):
            with self.subTest(bad=bad):
                other = report(CAND_SHA, 120, 4.3)
                other['validation_loss'] = bad
                with self.assertRaises(ValueError):
                    self.compare(candidate=other)

    def test_wrong_schema_architecture_or_tokenizer_rejected(self):
        other = report(CAND_SHA, 120, 4.3)
        other['schema'] = 2
        with self.assertRaises(ValueError):
            self.compare(candidate=other)
        other = report(CAND_SHA, 120, 4.3)
        other['tokenizer_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'tokenizer_sha256'):
            self.compare(candidate=other)

    def test_nonadvancing_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'did not advance'):
            assess(report(BASE_SHA, 100, 4.5), report(CAND_SHA, 100, 4.3),
                   anchor_sha=BASE_SHA, candidate_sha=CAND_SHA,
                   anchor_tokens=100, candidate_tokens=100)


if __name__ == '__main__':
    unittest.main()
