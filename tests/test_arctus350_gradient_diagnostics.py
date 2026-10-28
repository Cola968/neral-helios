"""CPU tests for safe, read-only last-FFN gradient interference diagnostics."""
import math
import unittest
from frontier350.arctus350_gradient_diagnostics import (
    cosine_from_stats, choose_bilingual_probe, permit_readonly_probe
)

class GradientDiagnosticTests(unittest.TestCase):
    def test_signed_cosine_statistics_and_invalid_bounds(self):
        self.assertAlmostEqual(cosine_from_stats(-6,9,4),-1.0)
        self.assertAlmostEqual(cosine_from_stats(0,1,1),0)
        self.assertAlmostEqual(cosine_from_stats(6,9,4),1.0)
        with self.assertRaises(ValueError):
            cosine_from_stats(1,0,1)
        with self.assertRaises(ValueError):
            cosine_from_stats(float("nan"),9,4)
        with self.assertRaises(ValueError):
            cosine_from_stats(1000,9,4)

    def test_exact_two_per_language_only_train_and_deterministic(self):
        records=[(f"de-{i}","de",{"token":i}) for i in range(10)]+[
            (f"en-{i}","en",{"token":i}) for i in range(10)]
        selected=choose_bilingual_probe(records,2)
        self.assertEqual(selected,choose_bilingual_probe(list(reversed(records)),2))
        self.assertEqual(len(selected),4)
        self.assertEqual(sum(r[1]=="de" for r in selected),2)
        self.assertEqual(sum(r[1]=="en" for r in selected),2)
        with self.assertRaises(ValueError):
            choose_bilingual_probe(records,20)
        with self.assertRaises(ValueError):
            choose_bilingual_probe(records+[records[0]],2)

    def test_today_only_gpu_safety(self):
        self.assertTrue(permit_readonly_probe(10000,65,"2026-10-28T19:30:00+01:00"))
        self.assertTrue(permit_readonly_probe(9000,65,"2026-10-28T19:30:00+01:00"))
        self.assertFalse(permit_readonly_probe(8000,65,"2026-10-28T19:30:00+01:00"))
        self.assertFalse(permit_readonly_probe(10000,80,"2026-10-28T19:30:00+01:00"))
        self.assertFalse(permit_readonly_probe(10000,65,"2026-10-29T00:30:00+01:00"))
        self.assertFalse(permit_readonly_probe(10000,65,"2026-10-28T20:40:00+01:00"))
        with self.assertRaises(ValueError):
            permit_readonly_probe(10000,float("nan"),"2026-10-28T19:30:00+01:00")

if __name__=="__main__":
    unittest.main()
