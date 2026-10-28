"""Bounded 64->96 native SFT update contracts (no GPU in unit tests)."""
import unittest
from frontier350.arctus350_targeted_exact_pilot import (
    admit_targeted, choose_schedule, candidate_worth_saving,
)

class TargetedPilotContracts(unittest.TestCase):
    def test_no_unbounded_or_unsafe_gpu_training(self):
        self.assertTrue(admit_targeted(True,10000,10000,.65,96,128))
        for x in [(False,10000,10000,.65,96,128),
                  (True,8000,10000,.65,96,128),
                  (True,10000,8000,.65,96,128),
                  (True,10000,10000,.80,96,128),
                  (True,10000,10000,.65,97,128),
                  (True,10000,10000,.65,96,256)]:
            with self.subTest(values=x):
                self.assertFalse(admit_targeted(*x))
        with self.assertRaises(ValueError):
            admit_targeted(True,10000,10000,float("nan"),96,128)

    def test_deterministic_disjoint_balanced_training(self):
        a=[("exact-%d"%i,"data%d"%i) for i in range(40)]
        b=[("math-%d"%i,"data%d"%i) for i in range(40)]
        c=[("human-%d"%i,"data%d"%i) for i in range(40)]
        result=choose_schedule(a,b,c,32)
        self.assertEqual(len(result),96)
        self.assertEqual(len({r[0] for r in result}),96)
        self.assertEqual(result,choose_schedule(list(reversed(a)),list(reversed(b)),list(reversed(c)),32))
        self.assertTrue(result[0][0].startswith("exact-"))
        self.assertTrue(result[1][0].startswith("math-"))
        self.assertTrue(result[2][0].startswith("human-"))

    def test_duplicate_examples_and_unavailable_cohort_refused(self):
        a=[("same",1)]*3
        b=[("m0",2),("m1",2),("m2",2)]
        c=[("h0",3),("h1",3),("h2",3)]
        with self.assertRaises(ValueError):choose_schedule(a,b,c,2)
        with self.assertRaises(ValueError):choose_schedule(b[:1],b,c,2)

    def test_no_candidate_promotion_from_lower_nll_alone(self):
        self.assertFalse(candidate_worth_saving(3.,1.,2.,2.,0))
        self.assertFalse(candidate_worth_saving(3.,1.,2.,2.5,1))
        self.assertFalse(candidate_worth_saving(3.,2.99,2.,2.,1))
        self.assertTrue(candidate_worth_saving(3.,2.0,2.,2.1,1))

if __name__=="__main__":
    unittest.main()
