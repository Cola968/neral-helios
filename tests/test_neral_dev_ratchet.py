"""CPU regression contracts for the development-only NERAL paired ratchet."""
import unittest

from scripts.neral_dev_ratchet import assess_hard36


def summary(raw=20, tir=15, wins=3, losses=8):
    return {
        "status": "NERAL_SAME_BASE_3ARM_DEVELOPMENT_DIAGNOSTIC",
        "collector_status": "NERAL_TIR_HARD36_DEVELOPMENT_ONLY",
        "cases": 36,
        "independent_holdout_verified": False,
        "weight_improvement_proven": False,
        "automatic_promotion_authorized": False,
        "arms": {
            "raw_qwen35": {"correct": raw},
            "neral_high_router": {"correct": tir},
            "neral_high": {"correct": 5},
        },
        "paired": {
            "neral_high_router_vs_raw_qwen35": {
                "wins": wins, "losses": losses,
            },
        },
    }


class DevelopmentRatchetTests(unittest.TestCase):
    def test_actual_reported_15_of_36_fails_against_20_of_36(self):
        result = assess_hard36(summary())
        self.assertFalse(result["development_gate_passed"])
        self.assertEqual(result["paired_losses"], 8)
        self.assertEqual(result["paired_wins"], 3)
        self.assertFalse(result["promotion_allowed"])

    def test_equal_score_is_only_a_development_nonregression(self):
        result = assess_hard36(summary(tir=20, wins=3, losses=3))
        self.assertTrue(result["development_gate_passed"])
        self.assertFalse(result["promotion_allowed"])

    def test_score_above_baseline_still_cannot_claim_holdout_promotion(self):
        result = assess_hard36(summary(tir=23, wins=6, losses=3))
        self.assertTrue(result["development_gate_passed"])
        self.assertFalse(result["promotion_allowed"])

    def test_new_random_cases_use_their_own_paired_baseline(self):
        result = assess_hard36(summary(raw=19, tir=20, wins=3, losses=2))
        self.assertTrue(result["development_gate_passed"])
        self.assertEqual(result["baseline_correct"], 19)

    def test_malformed_or_unpaired_report_is_rejected(self):
        for item in (summary(tir=True), summary(tir=37), summary(tir=15, wins=0, losses=8),
                     {**summary(), "independent_holdout_verified": True},
                     {**summary(), "collector_status": "NERAL_RELEASE_APPROVED"}):
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    assess_hard36(item)


if __name__ == "__main__":
    unittest.main()
