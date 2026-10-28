"""RED: math-specialist isolates general model, deduplicates operand pairs, and gates results."""
import unittest
from frontier350.arctus350_math_specialist import (
    choose_unique_math_pairs, accept_specialist, can_run_today
)

class MathSpecialistContracts(unittest.TestCase):
    def sample_rows(self):
        rows=[]
        for i in range(8):
            for lang in ("de","en"):
                for j in (0,1):
                    a=i*2+j
                    b=20
                    key=(a,b)
                    parity=(a+b)%2
                    rows.append((f"{lang}-{a}-{b}-t{j}",lang,parity,key,{"seq":j}))
                    rows.append((f"{lang}-{a}-{b}-v{j}",lang,parity,key,{"seq":j+3}))
        return rows

    def test_unique_pair_first_two_views_and_strata_balance(self):
        rows=self.sample_rows()
        a=choose_unique_math_pairs(rows,2)
        self.assertEqual(a,choose_unique_math_pairs(list(reversed(rows)),2))
        self.assertEqual(len(a),16)
        self.assertEqual(len(set(x[3] for x in a)),8)
        self.assertEqual(len(set(x[0] for x in a)),16)
        from collections import Counter
        self.assertEqual(Counter((a[1],a[2]) for a in a),
                         {("de",0):4,("de",1):4,("en",0):4,("en",1):4})
        with self.assertRaises(ValueError):
            choose_unique_math_pairs(rows,10)
        with self.assertRaises(ValueError):
            choose_unique_math_pairs(rows+[rows[0]],2)

    def test_high_accuracy_required_before_isolated_specialist_save(self):
        good=dict(new_before=0,new_after=12,prior_before=5,prior_after=10,
                  eos_count=32,total=32,repetition=0.)
        self.assertTrue(accept_specialist(**good))
        for partial in ({"new_after":11},{"prior_after":9},
                        {"eos_count":29},{"repetition":.03},{"total":33}):
            self.assertFalse(accept_specialist(**{**good,**partial}))

    def test_user_date_and_nvidia_admission(self):
        self.assertTrue(can_run_today(True,9300,8900,60,"2026-10-28T19:30:00+01:00",1200))
        self.assertFalse(can_run_today(True,8400,8900,60,"2026-10-28T19:30:00+01:00",1200))
        self.assertFalse(can_run_today(True,9300,8900,80,"2026-10-28T19:30:00+01:00",1200))
        self.assertFalse(can_run_today(True,9300,8900,60,"2026-10-28T20:26:00+01:00",1200))
        self.assertFalse(can_run_today(True,9300,8900,60,"2026-10-29T00:00:00+01:00",1200))
        self.assertFalse(can_run_today(False,9300,8900,60,"2026-10-28T19:30:00+01:00",1200))
        with self.assertRaises(ValueError):
            can_run_today(True,9300,8900,float("nan"),"2026-10-28T19:30:00+01:00",1200)

if __name__=="__main__":
    unittest.main()
