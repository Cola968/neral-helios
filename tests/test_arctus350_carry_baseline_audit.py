"""RED: fresh v3 arithmetic baseline must be heldout, balanced, read-only."""
import unittest
from collections import Counter
from frontier350.arctus350_carry_baseline_audit import (
    select_val_cases, score_observations, recognize_verified_case, admit_readonly
)
from frontier350.arctus350_carry_curriculum_v3 import _row, make_splits

class FreshCarryReadOnlyContracts(unittest.TestCase):
    def test_unseen_v3_val_selector_no_pair_leakage_and_carry_balance(self):
        splits=make_splits()
        selected=select_val_cases(splits["val"],4)
        self.assertEqual(selected,select_val_cases(list(reversed(splits["val"])),4))
        self.assertEqual(len(selected),32)
        seen=Counter()
        pairs=set()
        for row in selected:
            lang,kind,a,b,carry=recognize_verified_case(row)
            seen[(lang,kind,carry)]+=1
            self.assertNotIn((a,b),pairs)
            pairs.add((a,b))
        self.assertEqual(len(seen),8)
        self.assertTrue(all(n==4 for n in seen.values()))
        train_pairs={recognize_verified_case(row)[2:4] for row in splits["train"]}
        test_pairs={recognize_verified_case(row)[2:4] for row in splits["test"]}
        self.assertFalse(pairs & train_pairs)
        self.assertFalse(pairs & test_pairs)

    def test_reject_corrupted_answers_and_missing_cohorts(self):
        row=_row(57,68,"de","trace")
        self.assertEqual(recognize_verified_case(row),("de","trace",57,68,1))
        bad=dict(row)
        bad["messages"]=[dict(x) for x in row["messages"]]
        bad["messages"][1]["content"]="Ergebnis: 999"
        with self.assertRaises(ValueError):recognize_verified_case(bad)
        with self.assertRaises(ValueError):select_val_cases([row],4)

    def test_misleading_numeric_mention_does_not_count_as_correct(self):
        a=_row(57,68,"de","trace")
        b=_row(52,63,"en","direct")
        observations=[
            {"row":a,"text":"Falsch, aber 125 wird erwähnt.","eos":True,"tokens":[1,2,3]},
            {"row":b,"text":"115","eos":True,"tokens":[2,3,4]},
        ]
        result=score_observations(observations)
        self.assertEqual(result["overall"]["strict_exact"],1)
        self.assertEqual(result["overall"]["final_value_correct"],1)
        self.assertEqual(result["overall"]["eos_count"],2)
        self.assertNotIn("Falsch",str(result))
        self.assertNotIn("115",str(result))
        self.assertNotIn("row_id",str(result))
        self.assertEqual(result["cohorts"]["de/trace/carry"]["total"],1)
        self.assertEqual(result["cohorts"]["en/direct/no_carry"]["strict_exact"],1)

    def test_today_only_no_gpu_training(self):
        self.assertTrue(admit_readonly(True,9200,9200,60,"2026-10-28T20:00:00+01:00",600))
        self.assertFalse(admit_readonly(True,9200,9200,60,"2026-10-28T20:26:00+01:00",600))
        self.assertFalse(admit_readonly(True,7600,9200,60,"2026-10-28T20:00:00+01:00",600))
        self.assertFalse(admit_readonly(True,9200,9200,80,"2026-10-28T20:00:00+01:00",600))
        self.assertFalse(admit_readonly(True,9200,9200,60,"2026-10-29T00:01:00+01:00",600))
        with self.assertRaises(ValueError):
            admit_readonly(True,9200,9200,float("nan"),"2026-10-28T20:00:00+01:00",600)

if __name__=="__main__":unittest.main()
