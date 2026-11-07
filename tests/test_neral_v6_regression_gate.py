"""NERAL candidate release veto: RED first, CPU-only."""
from __future__ import annotations

import copy
import unittest

from scripts.neral_v6_fresh_dev_eval import (
    BASE_SHA, V5_SHA, V6_SHA, build_cases, suite_fingerprint,
)
from scripts.neral_v6_regression_gate import (
    RegressionRejected, audit_candidate_report, require_screen_pass,
)


def original_failed_v6():
    return {
        "schema": 1,
        "status": "NERAL_V6_NEW_STRUCTURAL_DEV_72_THREE_WAY",
        "total": 72,
        "task_families": 12,
        "suite_sha256": suite_fingerprint(build_cases()),
        "base_revision": BASE_SHA,
        "v5_adapter_sha256": V5_SHA,
        "v6_adapter_sha256": V6_SHA,
        "source_isolation": "exact_and_10gram_checked_against_v6_train_validation_only",
        "release_validated": False,
        "weights_promoted": False,
        "independent_benchmark_proven": False,
        "scores": {
            "qwen_fewshot": {"correct": 29, "total": 72, "incomplete": 0,
                             "ungraded_completed": 0,
                             "per_domain": {"math": 9, "reasoning": 15, "code_trace": 5}},
            "neral_v5": {"correct": 24, "total": 72, "incomplete": 0,
                         "ungraded_completed": 7,
                         "per_domain": {"math": 7, "reasoning": 14, "code_trace": 3}},
            "neral_v6": {"correct": 21, "total": 72, "incomplete": 0,
                         "ungraded_completed": 19,
                         "per_domain": {"math": 4, "reasoning": 14, "code_trace": 3}},
        },
        "paired": {
            "v5_vs_qwen": {"improvements": 3, "regressions": 8},
            "v6_vs_qwen": {"improvements": 2, "regressions": 10},
            "v6_vs_v5": {"improvements": 0, "regressions": 3},
        },
    }


def fresh_strong_candidate():
    data = original_failed_v6()
    data["v6_adapter_sha256"] = "a" * 64
    data["scores"]["neral_v6"] = {
        "correct": 31, "total": 72, "incomplete": 0,
        "ungraded_completed": 0,
        "per_domain": {"math": 10, "reasoning": 15, "code_trace": 6},
    }
    data["paired"]["v6_vs_qwen"] = {"improvements": 10, "regressions": 8}
    data["paired"]["v6_vs_v5"] = {"improvements": 7, "regressions": 0}
    return data


class RegressionGateContracts(unittest.TestCase):
    def test_real_v6_regression_is_quarantined_and_raises_on_promotion_attempt(self):
        status = audit_candidate_report(original_failed_v6())
        self.assertEqual(status["status"], "DEVELOPMENT_REGRESSION_BLOCKED")
        self.assertFalse(status["development_screen_passed"])
        self.assertFalse(status["release_authorized"])
        self.assertTrue(status["candidate_quarantined"])
        self.assertIn("known_regressed_adapter_sha", status["reasons"])
        self.assertIn("total_below_qwen_fewshot", status["reasons"])
        self.assertIn("total_below_neral_v5", status["reasons"])
        self.assertIn("ungraded_worse_than_neral_v5", status["reasons"])
        self.assertIn("domain_regression_math_vs_qwen_fewshot", status["reasons"])
        with self.assertRaises(RegressionRejected):
            require_screen_pass(original_failed_v6())

    def test_even_strong_new_candidate_has_no_release_authority(self):
        status = audit_candidate_report(fresh_strong_candidate())
        self.assertTrue(status["development_screen_passed"])
        self.assertFalse(status["candidate_quarantined"])
        self.assertFalse(status["release_authorized"])
        self.assertFalse(status["weights_promoted"])
        self.assertFalse(status["independent_benchmark_proven"])
        self.assertIsNone(require_screen_pass(fresh_strong_candidate()))

    def test_known_bad_adapter_remains_blocked_even_if_counts_are_forged(self):
        report = fresh_strong_candidate()
        report["v6_adapter_sha256"] = V6_SHA
        self.assertIn("known_regressed_adapter_sha",
                      audit_candidate_report(report)["reasons"])
        with self.assertRaises(RegressionRejected):
            require_screen_pass(report)

    def test_missing_report_or_baselines_fail_closed(self):
        for report in (None, {}, {"scores": {}}, {"release_validated": True}):
            with self.subTest(report=report):
                with self.assertRaises(RegressionRejected):
                    audit_candidate_report(report)

    def test_false_pair_arithmetic_or_case_counts_fail_closed(self):
        for field, value in (("total", 71), ("task_families", 11)):
            bad = fresh_strong_candidate()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(RegressionRejected):
                audit_candidate_report(bad)
        bad = fresh_strong_candidate()
        bad["paired"]["v6_vs_v5"]["regressions"] = 1
        with self.assertRaises(RegressionRejected):
            audit_candidate_report(bad)
        bad = fresh_strong_candidate()
        bad["scores"]["neral_v6"]["per_domain"]["math"] = 9
        with self.assertRaises(RegressionRejected):
            audit_candidate_report(bad)

    def test_no_win_or_per_domain_regression_blocks_even_when_total_grows(self):
        report = fresh_strong_candidate()
        report["scores"]["neral_v6"]["per_domain"] = {
            "math": 8, "reasoning": 17, "code_trace": 6,
        }
        status = audit_candidate_report(report)
        self.assertFalse(status["development_screen_passed"])
        self.assertIn("domain_regression_math_vs_qwen_fewshot", status["reasons"])

    def test_non_numeric_tainted_data_rejected_including_bool(self):
        for mutant in (-1, 73, True, 3.0, "21"):
            bad = original_failed_v6()
            bad["scores"]["neral_v6"]["correct"] = mutant
            with self.subTest(mutant=mutant), self.assertRaises(RegressionRejected):
                audit_candidate_report(bad)
        bad = original_failed_v6()
        bad["suite_sha256"] = "b" * 64
        with self.assertRaises(RegressionRejected):
            audit_candidate_report(bad)


if __name__ == "__main__":
    unittest.main()
