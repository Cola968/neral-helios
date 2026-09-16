"""Offline CPU-only regression tests; never load user data, weights or CUDA."""
import math
import unittest

from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices


class ValidationContractTests(unittest.TestCase):
    def test_small_population_evaluated_in_full(self):
        self.assertEqual(validation_indices(1), (0,))
        self.assertEqual(validation_indices(8), tuple(range(8)))
        self.assertEqual(validation_indices(16), tuple(range(16)))

    def test_spread_sample_covers_all_parts_not_first_eight(self):
        picked = validation_indices(100)
        self.assertEqual(len(picked), 16)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 99)
        self.assertEqual(len(set(picked)), 16)
        self.assertTrue(all(a < b for a, b in zip(picked, picked[1:])))
        self.assertTrue(any(index >= 50 for index in picked))
        self.assertEqual(validation_indices(100), picked)

    def test_bad_sampling_population_rejected(self):
        for total, limit in ((0, 16), (-1, 16), (True, 16), (20, 1), (20, True)):
            with self.subTest(total=total, limit=limit), self.assertRaises(ValueError):
                validation_indices(total, limit)

    def test_token_weighting_not_row_average(self):
        # Unequal answer lengths: unweighted row average would misleadingly be 2.0.
        self.assertAlmostEqual(token_weighted_loss([(1.0, 1), (3.0, 9)]), 2.8)
        self.assertEqual(token_weighted_loss([(2.0, 2), (2.0, 20)]), 2.0)

    def test_bad_loss_and_token_counts_rejected(self):
        for bad in ([], [(1.0, 0)], [(1.0, True)], [(float('nan'), 1)],
                    [(float('inf'), 1)], [(1.0, -2)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                token_weighted_loss(bad)

    def test_dynamic_padding_tracks_selected_rows(self):
        rows = [{"input_ids": [1, 4, 7]}, {"input_ids": [1, 4, 8, 9, 10]}]
        self.assertEqual(dynamic_batch_width(rows, 512), 5)
        self.assertEqual(dynamic_batch_width([rows[0]], 512), 3)
        self.assertNotEqual(dynamic_batch_width(rows, 512), 512)

    def test_dynamic_padding_rejects_invalid_shapes(self):
        for rows, limit in (([], 512), ([{"input_ids": [1]}], 512),
                            ([{"input_ids": [1, 2, 3]}], 2),
                            ([{"input_ids": [1, 2]}], True)):
            with self.subTest(rows=rows, limit=limit), self.assertRaises(ValueError):
                dynamic_batch_width(rows, limit)


if __name__ == "__main__":
    unittest.main()
