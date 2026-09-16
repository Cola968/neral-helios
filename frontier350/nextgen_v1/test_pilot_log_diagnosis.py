"""Pure CPU tests; no GPU or training is executed."""
import json
import unittest

from pilot_log_diagnosis import diagnose


def metrics():
    out = []
    for n in range(1, 9):
        out.append(dict(event="train", step=n, new_tokens=n * 24576,
                        processed_tokens=1010003584 + n * 24576,
                        tokens_per_update=24576, batch_chain=f"{n:064x}",
                        loss=3.0 + n / 100.0, grad_norm=0.1 + n / 10000.0,
                        lr=2e-5, seconds=1.0, tokens_per_second=24576.0))
    return out


def encode(rows):
    return ("\n".join(json.dumps(x) for x in rows) + "\n").encode()


class LogDiagnosisTests(unittest.TestCase):
    def test_matching_metrics_never_prove_resume_or_quality(self):
        a = metrics()
        report = diagnose(encode(a), encode(a))
        self.assertTrue(report["numeric_prefix_equal_before_interruption"])
        self.assertFalse(report["exact_resume_verified"])
        self.assertFalse(report["eligible_for_production"])
        self.assertEqual(report["classification"], "recorded_metrics_equal_state_equality_unverified")

    def test_pre_boundary_gradient_drift_is_not_a_resume_failure_proof(self):
        a, b = metrics(), metrics()
        b[0]["grad_norm"] += 1e-8
        b[1]["loss"] += 1e-4
        report = diagnose(encode(a), encode(b))
        self.assertEqual(report["classification"], "pre_resume_numeric_divergence")
        self.assertEqual(report["first_numeric_difference"], {"step": 1, "field": "grad_norm"})
        self.assertEqual(report["first_loss_difference_step"], 2)
        self.assertTrue(report["all_batch_hashes_equal"])
        self.assertFalse(report["numeric_prefix_equal_before_interruption"])

    def test_later_drift_only_marks_possible_resume_difference(self):
        a, b = metrics(), metrics()
        b[4]["loss"] += 0.1
        report = diagnose(encode(a), encode(b))
        self.assertTrue(report["numeric_prefix_equal_before_interruption"])
        self.assertEqual(report["classification"], "post_resume_numeric_divergence_possible")
        self.assertFalse(report["exact_resume_verified"])

    def test_changed_batch_chain_is_detected(self):
        a, b = metrics(), metrics()
        b[1]["batch_chain"] = "f" * 64
        report = diagnose(encode(a), encode(b))
        self.assertEqual(report["first_batch_hash_mismatch_step"], 2)
        self.assertFalse(report["numeric_prefix_equal_before_interruption"])
        self.assertEqual(report["classification"], "input_chain_diverged")

    def test_tampered_counters_rejected(self):
        a, b = metrics(), metrics()
        b[2]["new_tokens"] += 1
        with self.assertRaises(ValueError):
            diagnose(encode(a), encode(b))

    def test_duplicate_fields_rejected(self):
        a = encode(metrics()).replace(b'"event": "train",', b'"event": "train", "event": "train",', 1)
        with self.assertRaises(ValueError):
            diagnose(a, encode(metrics()))

    def test_nan_rejected(self):
        a, b = metrics(), metrics()
        a[0]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            diagnose(encode(a), encode(b))


if __name__ == "__main__":
    unittest.main()
