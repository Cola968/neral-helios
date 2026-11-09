"""CPU regression contracts for the development-only NERAL score ratchet."""
import unittest

from scripts.neral_dev_ratchet import assess_hard36


def summary(raw=20, tir=15):
    return {
        "status": "NERAL_TIR_HARD36_DEVELOPMENT_ONLY",
        "cases": 36,
        "independent_holdout_verified": False,
        "weight_improvement_proven": False,
        "correct": {"raw_qwen35": raw, "neral_high_router": tir, "neral_high": 5},
    }


class DevelopmentRatchetTests(unittest.TestCase):
    def test_actual_reported_15_of_36_fails_against_20_of_36(self):
        result = assess_hard36(summary())
        self.assertFalse(result["development_gate_passed"])
        self.assertEqual(result["regression_cases"], 5)
        self.assertFalse(result["promotion_allowed"])

    def test_equal_score_is_only_a_development_nonregression(self):
        result = assess_hard36(summary(tir=20))
        self.assertTrue(result["development_gate_passed"])
        self.assertFalse(result["promotion_allowed"])

    def test_score_above_baseline_still_cannot_claim_holdout_promotion(self):
        result = assess_hard36(summary(tir=23))
        self.assertTrue(result["development_gate_passed"])
        self.assertFalse(result["promotion_allowed"])

    def test_wrong_baseline_or_malformed_report_is_rejected(self):
        for item in (summary(raw=19), summary(tir=True), summary(tir=37),
                     {**summary(), "independent_holdout_verified": True},
                     {**summary(), "status": "NERAL_RELEASE_APPROVED"}):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    assess_hard36(item)


if __name__ == "__main__":
    unittest.main()
