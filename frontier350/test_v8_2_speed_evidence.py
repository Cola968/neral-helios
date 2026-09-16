from __future__ import annotations

import unittest

from apply_v8_2_speed_evidence import patch_controller
from verify_40k_quality import evaluate


class PassivePatcherTests(unittest.TestCase):
    def test_idempotent_and_training_logic_unchanged(self):
        source = (
            "        round_no += 1\n        before_tokens = train_tokens\n"
            'history.append({"round": round_no, "status": "STAGE_REJECTED_NO_REWIND",\n'
            'history.append({"round": round_no, "status": "CONTINUITY_COMMIT",\n'
        )
        changed = patch_controller(source)
        self.assertEqual(changed, patch_controller(changed))
        self.assertEqual(changed.count("stage_start_elapsed_seconds"), 2)
        self.assertEqual(changed.count("stage_end_elapsed_seconds"), 2)
        self.assertIn('"status": "CONTINUITY_COMMIT"', changed)

    def test_missing_anchor_fails_closed(self):
        with self.assertRaises(RuntimeError):
            patch_controller("unrelated source")


class AcceptanceGateTests(unittest.TestCase):
    def make_row(self, i, duration=100, status="CONTINUITY_COMMIT", healthy=True, accepted=5_000_000):
        start = (i - 1) * duration
        before = (i - 1) * 5_000_000
        if status == "STAGE_REJECTED_NO_REWIND":
            accepted = 0
        return {"round": i, "status": status,
                "stage_start_elapsed_seconds": start, "stage_end_elapsed_seconds": start + duration,
                "tokens_before": before, "tokens_after": before + accepted,
                "stage_tokens": accepted, "health": {"healthy": healthy}}

    def test_three_fast_full_healthy_stages_pass_continuity_not_full_quality(self):
        report = evaluate([self.make_row(i) for i in range(1, 4)])
        self.assertTrue(report["passed"])
        self.assertEqual(report["measured_end_to_end_accepted_tps"], 50_000)
        self.assertIn("full NIS2", report["quality_scope"])

    def test_peak_does_not_override_slow_wall_time(self):
        report = evaluate([self.make_row(i, duration=150) for i in range(1, 4)])
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["measured_end_to_end_accepted_tps"], 5_000_000 / 150)

    def test_prepatch_missing_evidence_fails(self):
        self.assertEqual(evaluate([{"status": "CONTINUITY_COMMIT"}])["status"], "INSUFFICIENT_EVIDENCE")

    def test_bad_health_fails(self):
        rows = [self.make_row(i) for i in range(1, 4)]
        rows[1]["health"]["healthy"] = False
        self.assertFalse(evaluate(rows)["passed"])

    def test_rejection_between_stages_fails_even_with_favorable_peak(self):
        rows = [self.make_row(1), self.make_row(2)]
        rows.append({"round": 3, "status": "STAGE_REJECTED_NO_REWIND",
                     "stage_start_elapsed_seconds": 200, "stage_end_elapsed_seconds": 210,
                     "continuity_tokens": 10_000_000})
        self.assertFalse(evaluate(rows)["passed"])

    def test_partial_salvage_fails(self):
        rows = [self.make_row(1), self.make_row(2)]
        rows.append(self.make_row(3, accepted=3_000_000))
        self.assertFalse(evaluate(rows)["passed"])

    def test_bad_audit_fails(self):
        rows = [self.make_row(i) for i in range(1, 4)]
        rows[2]["audit"] = {"status": "AUDIT_NOT_SAFE_CONTINUE"}
        self.assertFalse(evaluate(rows)["passed"])

    def test_disconnected_token_accounting_rejected(self):
        rows = [self.make_row(i) for i in range(1, 4)]
        rows[2]["tokens_before"] = 500_000
        with self.assertRaises(ValueError):
            evaluate(rows)

    def test_nonmonotonic_time_rejected(self):
        rows = [self.make_row(i) for i in range(1, 4)]
        rows[2]["stage_start_elapsed_seconds"] = 100
        with self.assertRaises(ValueError):
            evaluate(rows)


if __name__ == "__main__":
    unittest.main()
