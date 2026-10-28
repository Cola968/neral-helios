"""Bounded native Arctus350 FP32 SFT pilot: CPU-only safety contracts."""
from __future__ import annotations
import unittest
from frontier350.arctus350_clean_fp32_pilot import admit_pilot,choose_balanced


class CleanFP32PilotTests(unittest.TestCase):
    def test_strict_hw_and_authorization(self):
        self.assertTrue(admit_pilot(True,10000,10000,.68,64,128))
        for args in [
            (False,10000,10000,.68,64,128),
            (True,8000,10000,.68,64,128),
            (True,10000,8000,.68,64,128),
            (True,10000,10000,.85,64,128),
            (True,10000,10000,.68,65,128),
            (True,10000,10000,.68,64,256),
        ]:
            with self.subTest(args=args):
                self.assertFalse(admit_pilot(*args))
        for args in [(True,True,10000,.68,64,128),(True,-1,10000,.68,64,128),
                     (True,10000,10000,float("nan"),64,128)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                admit_pilot(*args)

    def test_balanced_selection_exact_count_and_stable_order(self):
        rows=[(f"d{i}","de",i) for i in range(8)]+[(f"e{i}","en",i) for i in range(8)]
        out=choose_balanced(rows,3)
        self.assertEqual(len(out),6)
        self.assertEqual({x[1] for x in out},{"de","en"})
        self.assertEqual(out,choose_balanced(list(reversed(rows)),3))
        self.assertEqual(len(set(x[0] for x in out)),6)

    def test_not_enough_short_bilingual_examples_fails_closed(self):
        rows=[("d","de","a"),("e","en","b")]
        with self.assertRaises(ValueError):
            choose_balanced(rows,2)

    def test_duplicated_example_ids_rejected(self):
        with self.assertRaises(ValueError):
            choose_balanced([("same","de",1),("same","en",2)],1)


if __name__=="__main__":
    unittest.main()
