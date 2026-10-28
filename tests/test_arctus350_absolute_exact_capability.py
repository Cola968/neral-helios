"""No manual-review candidate if zero exact instructions succeeded on DEV."""
from __future__ import annotations
import unittest
from frontier350.compare_chat350 import comparison_decision


def health(exact=0.0,rep=.1):
    return {"eos_fraction":.6,"mean_repeated_trigram_fraction":rep,
            "utf8_fraction":1.0,"exact_instruction_pct":exact}


class ExactCapabilityFloorTests(unittest.TestCase):
    def test_zero_exact_never_manual_review_even_if_no_regression(self):
        result=comparison_decision(health(0.),health(0.))
        self.assertTrue(result["gate"]["health_no_regression"])
        self.assertFalse(result["absolute_exact_success_observed"])
        self.assertEqual(result["decision"],"reject_zero_exact_instruction")
        self.assertFalse(result["automatic_promotion"])

    def test_a_genuine_exact_success_still_needs_manual_review(self):
        result=comparison_decision(health(0.),health(50.))
        self.assertEqual(result["decision"],"manual_review_required")
        self.assertTrue(result["absolute_exact_success_observed"])
        self.assertFalse(result["automatic_promotion"])

    def test_regression_and_missing_best_take_precedence(self):
        self.assertEqual(
          comparison_decision(health(50.),health(25.))["decision"],
          "reject_generation_regression")
        self.assertEqual(
          comparison_decision(health(0.),health(0.),{"best_checkpoint_exists":False})["decision"],
          "reject_no_sft_improvement")


if __name__=="__main__":
    unittest.main()
