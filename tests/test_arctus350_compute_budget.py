"""Read-only model-training feasibility math, not model performance."""
import json
import subprocess
import sys
import unittest

from frontier350.arctus350_compute_budget import (
    budget_report, NATIVE_PARAMS, TRAINED_TOKENS,
)


class ArctusBudgetTests(unittest.TestCase):
    def test_native_counts_are_source_locked_and_units_are_explicit(self):
        self.assertEqual(NATIVE_PARAMS,349_825_152)
        self.assertEqual(TRAINED_TOKENS,2_073_161_216)
        report=budget_report(TRAINED_TOKENS)
        self.assertEqual(report["status"],"THEORETICAL_ESTIMATE_ONLY")
        self.assertEqual(report["native_parameter_count"],NATIVE_PARAMS)
        self.assertEqual(report["historical_tokens"],TRAINED_TOKENS)
        self.assertEqual(report["planned_tokens"],TRAINED_TOKENS)
        self.assertEqual(report["approx_training_flops"],6*NATIVE_PARAMS*TRAINED_TOKENS)
        self.assertEqual(report["tokens_per_parameter"],"5.9263")
        self.assertEqual(report["historical_work_multiplier"],"1.0000")
        self.assertIs(report["started_training"],False)
        self.assertIs(report["measured_model_quality"],False)

    def test_4t_comparison_is_not_presented_as_a_requirement(self):
        report=budget_report(4_000_000_000_000)
        self.assertEqual(report["approx_training_flops"],6*NATIVE_PARAMS*4_000_000_000_000)
        self.assertEqual(report["historical_work_multiplier"],"1929.4207")
        self.assertEqual(report["sft_or_pretraining_recommendation"],"NONE_NO_PERFORMANCE_INFERENCE")
        self.assertIn("not a required training target",report["limitations"])
        self.assertIn("attention",report["limitations"])
        self.assertIn("runtime",report["limitations"])

    def test_invalid_types_or_out_of_bounds_fail_closed(self):
        for invalid in (0,-1,True,False,2.5,"4000000000000",None,10**20):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                budget_report(invalid)

    def test_equal_token_larger_model_is_not_arctus_identity(self):
        with self.assertRaises(ValueError):
            budget_report(2_073_161_216,model_parameters=175_000_000_000)

    def test_target_projection_is_pure_and_unchanged_for_identical_inputs(self):
        a=budget_report(10_000_000_000)
        b=budget_report(10_000_000_000)
        self.assertEqual(a,b)
        self.assertEqual(a["native_parameter_count"],NATIVE_PARAMS)
        self.assertNotIn("gpu_hours",a)
        self.assertNotIn("accuracy",a)

    def test_cli_produces_only_valid_json_with_no_side_effects(self):
        p=subprocess.run(
            [sys.executable,"-m","frontier350.arctus350_compute_budget",
             "--planned-tokens","2073161216"],
            capture_output=True,text=True,check=True)
        data=json.loads(p.stdout)
        self.assertEqual(data,budget_report(TRAINED_TOKENS))
        self.assertEqual(p.stderr,"")

    def test_cli_rejects_bad_budget_and_has_no_gpu_or_cloud_args(self):
        p=subprocess.run(
            [sys.executable,"-m","frontier350.arctus350_compute_budget",
             "--planned-tokens","0"],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(p.stdout,"")
        p=subprocess.run(
            [sys.executable,"-m","frontier350.arctus350_compute_budget",
             "--resume"],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)


if __name__=="__main__":
    unittest.main()
