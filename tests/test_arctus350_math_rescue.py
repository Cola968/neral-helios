"""Prove arithmetic rescue cannot silently erase native exact-output skill."""
import unittest
from frontier350.arctus350_math_rescue import (
    choose_repair_schedule, admit_math_rescue, keep_candidate
)

class RescueContracts(unittest.TestCase):
    def test_disjoint_from_prior_96_and_deterministic(self):
        groups={
          "small_math":[("m%d"%i,i) for i in range(150)],
          "exact":[("e%d"%i,i) for i in range(90)],
          "chat":[("c%d"%i,i) for i in range(90)],
        }
        old={"m1","e2","c3"}
        a=choose_repair_schedule(groups,{"small_math":128,"exact":64,"chat":64},old)
        b=choose_repair_schedule({k:list(reversed(v)) for k,v in groups.items()},
                                 {"small_math":128,"exact":64,"chat":64},old)
        self.assertEqual(a,b)
        self.assertEqual(len(a),256)
        self.assertEqual(len(set(x[0] for x in a)),256)
        self.assertFalse(any(x[0] in old for x in a))
        self.assertEqual([x[0][0] for x in a[:3]],["m","e","c"])
        with self.assertRaises(ValueError):
            choose_repair_schedule(groups,{"small_math":250,"exact":64,"chat":64},old)

    def test_math_must_improve_and_all_exact_words_remain(self):
        self.assertTrue(keep_candidate(3,7,32,32,76,76,0.0,0.0,3.0,3.05,0.07,0.07,0.917,0.917,50,50))
        self.assertFalse(keep_candidate(3,3,32,32,76,76,0.0,0.0,3.0,3.05,0.07,0.07,0.917,0.917,50,50))
        self.assertFalse(keep_candidate(3,7,32,31,76,76,0.0,0.0,3.0,3.05,0.07,0.07,0.917,0.917,50,50))
        self.assertFalse(keep_candidate(3,7,32,32,76,75,0.0,0.0,3.0,3.05,0.07,0.07,0.917,0.917,50,50))
        self.assertFalse(keep_candidate(3,7,32,32,76,76,0.0,0.02,3.0,3.05,0.07,0.07,0.917,0.917,50,50))
        self.assertFalse(keep_candidate(3,7,32,32,76,76,0.0,0.0,3.0,3.28,0.07,0.07,0.917,0.917,50,50))
        self.assertFalse(keep_candidate(3,7,32,32,76,76,0.0,0.0,3.0,3.05,0.07,0.09,0.917,0.917,50,50))

    def test_hardware_clock_and_explicit_owner_gate(self):
        self.assertTrue(admit_math_rescue(True,10000,10000,54,"2026-10-28T18:00:00+01:00",1200))
        self.assertFalse(admit_math_rescue(False,10000,10000,54,"2026-10-28T18:00:00+01:00",1200))
        self.assertFalse(admit_math_rescue(True,8000,10000,54,"2026-10-28T18:00:00+01:00",1200))
        self.assertFalse(admit_math_rescue(True,10000,10000,80,"2026-10-28T18:00:00+01:00",1200))
        self.assertFalse(admit_math_rescue(True,10000,10000,54,"2026-10-29T01:00:00+01:00",1200))
        self.assertFalse(admit_math_rescue(True,10000,10000,54,"2026-10-28T20:55:00+01:00",1200))
        with self.assertRaises(ValueError):
            admit_math_rescue(True,10000,10000,float("nan"),"2026-10-28T18:00:00+01:00",1200)
if __name__=="__main__":
    unittest.main()
