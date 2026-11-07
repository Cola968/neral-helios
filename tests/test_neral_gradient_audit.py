"""CPU-only tests for logged-gradient diagnostics, never GPU or model inference."""
import unittest

from scripts.neral_gradient_audit import summarize_gradient_logs, summarize_private_report


class GradientAuditTests(unittest.TestCase):
    def test_counts_only_observed_preclip_samples_not_all_optimizer_steps(self):
        rows = [
            {"step": 5, "grad_norm": 4.2948, "loss": 1.7757},
            {"step": 10, "grad_norm": 2.5255, "loss": 1.3959},
            {"step": 15, "grad_norm": 0.8, "loss": 1.2868},
            {"step": 20, "eval_loss": 0.9858},
        ]
        result = summarize_gradient_logs(rows, max_grad_norm=1.0, expected_steps=120)
        self.assertEqual(result["observed_gradient_samples"], 3)
        self.assertEqual(result["logged_above_clip_threshold"], 2)
        self.assertEqual(result["observed_fraction_above_threshold"], 2/3)
        self.assertEqual(result["optimizer_steps_in_run"], 120)
        self.assertFalse(result["all_optimizer_steps_measured"])
        self.assertEqual(result["first_observed_step"], 5)
        self.assertEqual(result["last_observed_step"], 15)
        self.assertEqual(result["norm_position"], "before_clipping")
        self.assertFalse(result["generalization_proven"])

    def test_complete_observation_does_not_claim_model_improved(self):
        r = summarize_gradient_logs([{"step": 1, "grad_norm": 1.2},
                                     {"step": 2, "grad_norm": 1.4}],
                                    max_grad_norm=1.0, expected_steps=2)
        self.assertTrue(r["all_optimizer_steps_measured"])
        self.assertEqual(r["observed_fraction_above_threshold"], 1.0)
        self.assertFalse(r["generalization_proven"])

    def test_equal_to_threshold_is_not_counted_above(self):
        r = summarize_gradient_logs([{"step": 1, "grad_norm": 1.0}], max_grad_norm=1.0)
        self.assertEqual(r["logged_above_clip_threshold"], 0)

    def test_missing_gradient_samples_fails_closed(self):
        with self.assertRaises(ValueError):
            summarize_gradient_logs([{"step": 5, "loss": 1.2}], max_grad_norm=1.0)

    def test_reject_invalid_norms_steps_and_limits(self):
        invalid = [
            ([{"step": 1, "grad_norm": float("nan")}], 1.0, 2),
            ([{"step": 1, "grad_norm": float("inf")}], 1.0, 2),
            ([{"step": 1, "grad_norm": -1.0}], 1.0, 2),
            ([{"step": True, "grad_norm": 1.2}], 1.0, 2),
            ([{"step": 2, "grad_norm": 1.2}, {"step": 1, "grad_norm": 1.5}], 1.0, 2),
            ([{"step": 1, "grad_norm": 1.2}], 0, 2),
            ([{"step": 3, "grad_norm": 1.2}], 1.0, 2),
            ([{"step": 1, "grad_norm": 1.2}], 1.0, 0),
        ]
        for logs, threshold, steps in invalid:
            with self.subTest(logs=logs, threshold=threshold, steps=steps):
                with self.assertRaises(ValueError):
                    summarize_gradient_logs(logs, max_grad_norm=threshold, expected_steps=steps)

    def test_private_scalar_report_uses_only_existing_recent_logs(self):
        report = {"steps_completed": 120,
                  "recent_train_logs": [{"step": 115, "grad_norm": 1.3},
                                        {"step": 120, "grad_norm": 1.4}],
                  "raw_model_answer": "PRIVATE CONTENT MUST NOT PRINT"}
        result = summarize_private_report(report, max_grad_norm=1.0)
        self.assertEqual(result["observed_gradient_samples"], 2)
        self.assertEqual(result["last_observed_step"], 120)
        self.assertNotIn("PRIVATE", str(result))


if __name__ == "__main__":
    unittest.main()
