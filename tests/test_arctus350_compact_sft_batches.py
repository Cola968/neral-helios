"""CPU-only TDD: proof of actual token-padding reduction, not model intelligence.

Synthetic encoded rows deliberately exclude private SFT text and native weights.
"""
import random
import unittest

from frontier350.sft_batching350 import (
    compact_epoch_order, compact_length_batches, epoch_order, padding_stats,
)


class Encoded:
    def __init__(self, length):
        self.input_ids = [length] * length


class CompactSFTBatchTests(unittest.TestCase):
    def test_exactly_once_per_epoch_without_drop_or_repeat(self):
        rows = [Encoded(i) for i in range(4, 84, 4)]  # 20 distinct rows
        order = compact_epoch_order(rows, random.Random(80), batch_size=4)
        self.assertEqual(len(order), len(rows))
        self.assertEqual(sorted(order), list(range(len(rows))))

    def test_repeatable_with_same_seed_and_changes_between_seeds(self):
        rows = [Encoded(i) for i in range(4, 132, 4)]
        a = compact_epoch_order(rows, random.Random(9), batch_size=4)
        b = compact_epoch_order(rows, random.Random(9), batch_size=4)
        c = compact_epoch_order(rows, random.Random(10), batch_size=4)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_compact_beats_current_bucket_padding_on_synthetic_spread(self):
        rows = [Encoded(i) for i in range(8, 264, 2)]  # 128 varied lengths
        rng_a, rng_b = random.Random(5), random.Random(5)
        legacy = epoch_order(rows, rng_a, batch_size=4, bucket_size=64)
        compact = compact_epoch_order(rows, rng_b, batch_size=4)
        def padded(order):
            return sum(
                padding_stats([rows[x] for x in order[j:j + 4]])["padding_tokens"]
                for j in range(0, len(order), 4)
            )
        old_waste, new_waste = padded(legacy), padded(compact)
        self.assertGreater(old_waste, 0)
        self.assertLess(new_waste, old_waste * .35)
        # This is a deterministic token-count result, not measured GPU speedup.
        self.assertEqual(sum(len(rows[i].input_ids) for i in legacy),
                         sum(len(rows[i].input_ids) for i in compact))

    def test_generator_no_lost_rows_across_epoch_boundary(self):
        rows = [Encoded(i) for i in range(6, 22)]  # 16, divisible by 4
        gen = compact_length_batches(rows, batch_size=4, rng=random.Random(37))
        first = [v.input_ids[0] for _ in range(4) for v in next(gen)]
        second = [v.input_ids[0] for _ in range(4) for v in next(gen)]
        self.assertEqual(sorted(first), list(range(6, 22)))
        self.assertEqual(sorted(second), list(range(6, 22)))

    def test_odd_dataset_makes_only_finite_batches_and_preserves_rows(self):
        rows = [Encoded(i) for i in range(5, 16)]  # 11, batch 4
        gen = compact_length_batches(rows, batch_size=4, rng=random.Random(4))
        batches = [next(gen) for _ in range(9)]
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        self.assertEqual(sorted({b.input_ids[0] for batch in batches for b in batch}),
                         list(range(5, 16)))

    def test_no_data_or_oversized_batch_is_not_silently_accepted(self):
        for rows, size in (([], 4), ([Encoded(3)], 4), ([Encoded(3)], 0)):
            with self.subTest(rows=len(rows), size=size), self.assertRaises(ValueError):
                next(compact_length_batches(rows, batch_size=size, rng=random.Random(1)))
        with self.assertRaises(ValueError):
            compact_epoch_order([Encoded(4)], random.Random(1), batch_size=-1)


if __name__ == "__main__":
    unittest.main()
