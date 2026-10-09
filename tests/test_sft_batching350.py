from __future__ import annotations
import random,unittest
from frontier350.sft_batching350 import epoch_order,length_bucket_batches,padding_stats

class R:
    def __init__(self,n):self.input_ids=list(range(n))

class BatchingTests(unittest.TestCase):
    def test_epoch_exact(self):
        rows=[R(n) for n in [20,3,18,4,17,5,16,6,15,7,14,8,13,9,12,10]]
        o=epoch_order(rows,random.Random(1),batch_size=4,bucket_size=8)
        self.assertEqual(sorted(o),list(range(len(rows))))

    def test_deterministic(self):
        rows=[R(i+2) for i in range(20)]
        a=epoch_order(rows,random.Random(9),batch_size=4,bucket_size=8)
        b=epoch_order(rows,random.Random(9),batch_size=4,bucket_size=8)
        self.assertEqual(a,b)

    def test_bucket_reduces_padding_on_spread(self):
        rows=[R(n) for n in range(4,84,4)]
        g=length_bucket_batches(rows,4,random.Random(7),bucket_size=8)
        stats=[padding_stats(next(g)) for _ in range(5)]
        self.assertLess(sum(x["padding_fraction"] for x in stats)/len(stats),0.35)

    def test_nondivisible_dataset(self):
        rows=[R(i+2) for i in range(11)]
        g=length_bucket_batches(rows,4,random.Random(3),bucket_size=8)
        self.assertEqual([len(next(g)) for _ in range(4)],[4,4,4,4])

    def test_invalid_bucket(self):
        with self.assertRaises(ValueError):
            epoch_order([R(2)]*8,random.Random(1),batch_size=4,bucket_size=6)

if __name__=="__main__":
    unittest.main()
