"""Known-task fast-path GPU smoke is NEVER a new blind benchmark."""
import unittest
from scripts.neral_verified_fast_smoke import (
    SELECTED_DEV_IDS, select_unchanged_cases, compare_known_cases,
)

class FastSmokeContracts(unittest.TestCase):
    def test_only_known_frozen_cases_eligible(self):
        rows=select_unchanged_cases()
        self.assertEqual(len(rows),6)
        self.assertEqual([x["id"] for x in rows],list(SELECTED_DEV_IDS))
        self.assertEqual(len(set(SELECTED_DEV_IDS)),6)
        self.assertTrue(all(x["id"].startswith("blind100-") for x in rows))
    def test_summary_distinguishes_latency_from_model_intelligence(self):
        observations=[{"id":n,"full_latency_ms":10000+i*500,
            "fast_latency_ms":3000+i*100,"full_correct":i!=0,
            "fast_correct":i!=0,"full_verified":i in (3,4),
            "fast_verified":i in (3,4),"full_model_loads":2,
            "fast_model_loads":1 if i not in (3,4) else 0}
            for i,n in enumerate(SELECTED_DEV_IDS)]
        result=compare_known_cases(observations)
        self.assertEqual(result["count"],6)
        self.assertFalse(result["blind_test"])
        self.assertEqual(result["paired_answer_disagreements"],0)
        self.assertLess(result["fast_p50_ms"],result["full_p50_ms"])
        self.assertEqual(result["baseline_model_only_responses"],4)
        with self.assertRaises(ValueError):
            compare_known_cases(observations[:5])

if __name__=="__main__":unittest.main()
