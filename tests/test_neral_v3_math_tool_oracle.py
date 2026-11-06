"""Regressions for the eight fixed equation-development cases."""
import unittest
from scripts.neral_v3_math_tool_oracle import score_equations


class MathToolOracle(unittest.TestCase):
    def test_frozen_eight_exactly_correct(self):
        r=score_equations()
        self.assertEqual(r["equations"],8)
        self.assertEqual(r["tool_oracle_correct"],8)
        self.assertEqual(r["old_v2_without_tool_correct"],2)
        self.assertTrue(all(x["passed"] for x in r["cases"]))

    def test_oracle_must_not_be_reported_as_agent_success(self):
        r=score_equations()
        self.assertFalse(r["model_selected_tool_actions_measured"])
        self.assertFalse(r["v3_weights_trained"])
        self.assertFalse(r["independent_release_holdout"])
        self.assertEqual(r["status"],"DETERMINISTIC_TOOL_ORACLE_DEV_ONLY")
