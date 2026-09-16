"""CPU-only contract checks; no synthetic test is evidence of model capability."""
import copy
import unittest

from admission import validate_evidence


def fixture(phase="post_training", seconds=80.0):
    base = {
        "tokenizer_sha256": "a" * 64,
        "validation_sha256": "b" * 64,
        "dev_suite_sha256": "c" * 64,
        "eval_protocol_sha256": "d" * 64,
        "checkpoint_sha256": "e" * 64,
        "validation_loss": 3.5,
        "repeated_trigram_pct": 50.0,
        "dev_categories": {"math": 30.0, "language": 50.0},
        "generation": {"protocol_sha256": "f" * 64, "cases": 24, "exact_correct": 0, "eos_correct": 0},
    }
    candidate = copy.deepcopy(base)
    candidate.update(checkpoint_sha256="1" * 64, validation_loss=3.4, repeated_trigram_pct=45.0)
    candidate["dev_categories"] = {"math": 35.0, "language": 50.0}
    candidate["generation"].update(exact_correct=1, eos_correct=1)
    return {
        "evidence_version": 1,
        "source": "self_hosted_rtx4070",
        "weights_origin": "neral_own_weights",
        "checkpoint_isolated": True,
        "production_unchanged": True,
        "same_data_order": True,
        "same_token_budget": True,
        "safety_limits_unchanged": True,
        "phase": phase,
        "stages": [
            {"accepted_tokens": 5_000_000, "wall_seconds": seconds,
             "rejected_tokens": 0, "includes_eval_checkpoint_recovery": True,
             "health_pass": True}
            for _ in range(3)
        ],
        "baseline": base,
        "candidate": candidate,
    }


class AdmissionTests(unittest.TestCase):
    def test_valid_synthetic_contract_not_real_promotion(self):
        result = validate_evidence(fixture())
        self.assertTrue(result["admitted"])
        self.assertEqual(result["target_accepted_tps"], 50000)
        self.assertAlmostEqual(result["combined_accepted_tps"], 62500)
        self.assertFalse(result["stretch_70k_verified"])
        self.assertIn("not a production promotion", result["warning"])

    def test_exact_50k_boundary_is_inclusive(self):
        result = validate_evidence(fixture(seconds=100))
        self.assertTrue(result["admitted"])
        self.assertEqual(result["combined_accepted_tps"], 50000)
        self.assertFalse(result["stretch_70k_verified"])

    def test_stretch_is_separate(self):
        self.assertTrue(validate_evidence(fixture(seconds=70))["stretch_70k_verified"])

    def test_rejects_short_or_incomplete_stages(self):
        e = fixture(); e["stages"] = e["stages"][:2]
        with self.assertRaises(ValueError): validate_evidence(e)
        e = fixture(); e["stages"][0]["accepted_tokens"] -= 1
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_rejected_stage_fails_even_if_fast(self):
        e = fixture(); e["stages"][0]["rejected_tokens"] = 100
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_wall_time_must_include_overheads(self):
        e = fixture(); e["stages"][1]["includes_eval_checkpoint_recovery"] = False
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_slow_tail_fails_despite_fast_other_stages(self):
        e = fixture(); e["stages"][0]["wall_seconds"] = 101
        result = validate_evidence(e)
        self.assertFalse(result["admitted"])
        self.assertIn("stage_0_below_50k", result["failures"])

    def test_health_failure_cannot_be_masked(self):
        e = fixture(); e["stages"][0]["health_pass"] = False
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_validation_regression_rejected(self):
        e = fixture(); e["candidate"]["validation_loss"] = 3.6
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_per_category_regression_rejected(self):
        e = fixture(); e["candidate"]["dev_categories"]["math"] = 29
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_post_training_requires_real_generation_improvement(self):
        e = fixture(); e["candidate"]["generation"]["eos_correct"] = 0
        self.assertFalse(validate_evidence(e)["admitted"])
        e = fixture(); e["candidate"]["generation"]["exact_correct"] = 0
        self.assertFalse(validate_evidence(e)["admitted"])

    def test_foundation_does_not_claim_chat_success(self):
        e = fixture(phase="foundation"); del e["candidate"]["generation"]
        self.assertTrue(validate_evidence(e)["admitted"])

    def test_mismatched_dataset_fail_closed(self):
        e = fixture(); e["candidate"]["validation_sha256"] = "3" * 64
        with self.assertRaises(ValueError): validate_evidence(e)

    def test_unsafe_production_setting_fails(self):
        e = fixture(); e["safety_limits_unchanged"] = False
        with self.assertRaises(ValueError): validate_evidence(e)

    def test_nan_fails_closed(self):
        e = fixture(); e["stages"][0]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError): validate_evidence(e)

    def test_missing_category_fails_closed(self):
        e = fixture(); del e["candidate"]["dev_categories"]["math"]
        with self.assertRaises(ValueError): validate_evidence(e)


if __name__ == "__main__":
    unittest.main()
