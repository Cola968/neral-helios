"""RED pure CPU contracts: balanced, new-pair math rescue, no skill regression."""
import unittest
from frontier350.arctus350_balanced_math_v2_pilot import (
    balanced_unique_selection, keep_v2_candidate, admit_today
)

class MathV2PilotContracts(unittest.TestCase):
    def test_balanced_distinct_operand_pairs_stable_order(self):
        rows=[]
        for lang in ("de","en"):
            for parity in (0,1):
                for n in range(25):
                    rows.append((
                        f"{lang}-{parity}-{n}",lang,parity,
                        (parity*100+n,n+20),
                        ("mock",lang,parity,n),
                    ))
        selected=balanced_unique_selection(rows,3)
        self.assertEqual(len(selected),12)
        self.assertEqual(len({r[3] for r in selected}),12)
        self.assertEqual(selected,balanced_unique_selection(list(reversed(rows)),3))
        from collections import Counter
        self.assertEqual(Counter((r[1],r[2]) for r in selected),
                         {("de",0):3,("de",1):3,("en",0):3,("en",1):3})
        with self.assertRaises(ValueError):
            balanced_unique_selection(rows,40)

    def test_duplicate_id_and_no_tiny_cohort_is_accepted(self):
        rows=[("same","de",0,(0,2),"a"),("same","en",0,(2,3),"b")]
        with self.assertRaises(ValueError):balanced_unique_selection(rows,1)
        with self.assertRaises(ValueError):balanced_unique_selection([],1)

    def test_new_unseen_math_and_all_previous_skills_are_required(self):
        good=dict(
            new_before=0,new_after=4,old_math_before=5,old_math_after=5,
            old_words_before=32,old_words_after=32,old_eos_before=76,old_eos_after=76,
            old_repeat_before=0.,old_repeat_after=0.,
            human_before=3.0,human_after=3.08,
            dev_eos_before=1.,dev_eos_after=1.,
            dev_repeat_before=.019,dev_repeat_after=.019,
            dev_exact_before=50.,dev_exact_after=50.,
        )
        self.assertTrue(keep_v2_candidate(**good))
        for overrides in (
            {"new_after":2},
            {"old_math_after":4},
            {"old_words_after":31},
            {"old_eos_after":75},
            {"old_repeat_after":0.002},
            {"human_after":3.3},
            {"dev_eos_after":.91667},
            {"dev_repeat_after":.020},
            {"dev_exact_after":0},
        ):
            self.assertFalse(keep_v2_candidate(**{**good,**overrides}))

    def test_hard_wednesday_only_deadline_gpu_and_duration(self):
        self.assertTrue(admit_today(True,10000,10000,59,"2026-10-28T19:15:00+01:00",1200))
        self.assertFalse(admit_today(True,10000,10000,59,"2026-10-28T20:35:00+01:00",1200))
        self.assertFalse(admit_today(True,10000,10000,59,"2026-10-29T01:00:00+01:00",1200))
        self.assertFalse(admit_today(True,8000,10000,59,"2026-10-28T19:15:00+01:00",1200))
        self.assertFalse(admit_today(True,10000,10000,80,"2026-10-28T19:15:00+01:00",1200))
        self.assertFalse(admit_today(False,10000,10000,59,"2026-10-28T19:15:00+01:00",1200))
        with self.assertRaises(ValueError):
            admit_today(True,10000,10000,float("nan"),"2026-10-28T19:15:00+01:00",1200)

if __name__=="__main__":
    unittest.main()
