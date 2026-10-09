from __future__ import annotations
import random
import unittest
from frontier350.sft_stratified350 import stratified_batches, cohort_summary


class Row:
    def __init__(self, i, length):
        self.id = i
        self.input_ids = [i] * length


class StratifiedBatchTests(unittest.TestCase):
    def setUp(self):
        self.rows = [Row(i, i * 2 + 5) for i in range(30)]
        self.tags = [i % 3 == 0 for i in range(30)]

    def test_human_quota_every_batch(self):
        g = stratified_batches(self.rows, self.tags, 4, random.Random(123),
                               human_per_batch=2, bucket_size=8)
        for _ in range(20):
            b = next(g)
            self.assertEqual(len(b), 4)
            self.assertEqual(sum(self.tags[x.id] for x in b), 2)

    def test_deterministic_across_restarts(self):
        a = stratified_batches(self.rows, self.tags, 4, random.Random(29),
                               human_per_batch=2, bucket_size=8)
        b = stratified_batches(self.rows, self.tags, 4, random.Random(29),
                               human_per_batch=2, bucket_size=8)
        self.assertEqual([[x.id for x in next(a)] for _ in range(20)],
                         [[x.id for x in next(b)] for _ in range(20)])

    def test_no_repeat_before_cohort_exhausted(self):
        g = stratified_batches(self.rows, self.tags, 4, random.Random(11),
                               human_per_batch=2, bucket_size=8)
        observed = []
        for _ in range(5):
            observed.extend(x.id for x in next(g) if self.tags[x.id])
        self.assertEqual(len(set(observed)), 10)

    def test_distinct_nonhuman_first_epoch(self):
        g = stratified_batches(self.rows, self.tags, 4, random.Random(8),
                               human_per_batch=2, bucket_size=8)
        observed=[]
        for _ in range(10):
            observed.extend(x.id for x in next(g) if not self.tags[x.id])
        self.assertEqual(len(set(observed)), 20)

    def test_invalid_config_fails_closed(self):
        with self.assertRaises(ValueError):
            next(stratified_batches(self.rows, self.tags[:5], 4, random.Random(1)))
        with self.assertRaises(ValueError):
            next(stratified_batches(self.rows, self.tags, 4, random.Random(1),
                                    human_per_batch=4))
        self.assertEqual(cohort_summary(self.tags)["human_rows"], 10)


if __name__ == "__main__":
    unittest.main()
