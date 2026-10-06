"""Pure synthetic CPU contract fixtures; these are not actual GPU measurements."""
import copy
import unittest

from compare_probes import compare, verify


def report(execution="compile-default", batch=8, tps=42000.0):
    tokens = batch * 512 * 32
    setup = 61.0
    wall = tokens / tps
    return {
        "evidence_version": 2,
        "classification": "compute_only_synthetic_unaccepted_tokens",
        "gpu": "NVIDIA GeForce RTX 4070",
        "model_sha256": "a" * 64,
        "parameters": 349825152,
        "trainable_parameters": 55158272,
        "train_blocks": 4,
        "execution": execution,
        "batch": batch, "seq": 512, "warmup_steps": 8, "measured_steps": 32,
        "timed_synthetic_tokens": tokens,
        "compute_only_tokens_per_second": tps,
        "measured_wall_seconds": wall,
        "setup_and_warmup_wall_seconds": setup,
        "setup_inclusive_synthetic_tokens_per_second": tokens / (wall + setup),
        "slowest_measured_step_tokens_per_second": tps * .95,
        "synthetic_50k_target": 50000,
        "synthetic_50k_timed_target_met": tps >= 50000,
        "peak_reserved_gib": 5.0,
        "peak_temperature_c": 70,
        "eligible_for_production": False,
    }


class MatchedProbeTests(unittest.TestCase):
    def test_matched_candidate_below_50k_stays_unpromoted(self):
        result = compare(report(), report("compile-overhead", tps=49000))
        self.assertTrue(result["candidate_faster"])
        self.assertFalse(result["candidate_50k_synthetic_timed"])
        self.assertFalse(result["eligible_for_production"])
        self.assertFalse(result["numerical_parity_verified"])
        self.assertFalse(result["end_to_end_accepted_training_verified"])

    def test_synthetic_50k_never_passes_real_training(self):
        result = compare(report(), report("compile-overhead", tps=55000))
        self.assertTrue(result["candidate_50k_synthetic_timed"])
        self.assertFalse(result["candidate_50k_setup_inclusive"])
        self.assertFalse(result["end_to_end_accepted_training_verified"])

    def test_mismatched_model_or_shape_rejected(self):
        for field, value in (("batch", 16), ("model_sha256", "b" * 64),
                             ("trainable_parameters", 1234), ("measured_steps", 33)):
            with self.subTest(field=field):
                candidate = report("compile-overhead")
                candidate[field] = value
                with self.assertRaises(ValueError):
                    compare(report(), candidate)

    def test_rate_and_target_tampering_rejected(self):
        for field, value in (("measured_wall_seconds", 1),
                             ("synthetic_50k_timed_target_met", True),
                             ("setup_inclusive_synthetic_tokens_per_second", 999999),
                             ("slowest_measured_step_tokens_per_second", 100000)):
            with self.subTest(field=field):
                candidate = report("compile-overhead")
                candidate[field] = value
                with self.assertRaises(ValueError):
                    compare(report(), candidate)

    def test_safety_or_production_claim_rejected(self):
        for field, value in (("peak_temperature_c", 83), ("peak_reserved_gib", 10),
                             ("eligible_for_production", True), ("classification", "accepted")):
            with self.subTest(field=field):
                candidate = report("compile-overhead")
                candidate[field] = value
                with self.assertRaises(ValueError):
                    compare(report(), candidate)

    def test_bad_number_and_short_run_rejected(self):
        candidate = report("compile-overhead")
        candidate["compute_only_tokens_per_second"] = float("nan")
        with self.assertRaises(ValueError):
            verify(candidate, "compile-overhead")
        candidate = report("compile-overhead")
        candidate["measured_steps"] = 2
        with self.assertRaises(ValueError):
            verify(candidate, "compile-overhead")


if __name__ == "__main__":
    unittest.main()
