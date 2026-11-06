"""CPU contracts for the reproducible v3 data candidate."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.neral_v3_combined import (
    V2_ROWS, CORE_ALREADY_IN_V2, EQUATIONS, TRAJECTORIES, TOTAL,
    equation_rows, trajectory_rows, check_overlap,
)
from neral_agent_math import equation_final_answer
from scripts.neral_math_v2_eval import cases as math_cases
from scripts.neral_v2_generalization_probe import checked_cases


class CombinedV3Tests(unittest.TestCase):
    def test_counts_are_not_double_counted(self):
        self.assertEqual((V2_ROWS, CORE_ALREADY_IN_V2, EQUATIONS, TRAJECTORIES, TOTAL),
                         (744, 86, 48, 50, 842))

    def test_equations_have_exact_results_and_short_final(self):
        rows = equation_rows()
        self.assertEqual(rows, equation_rows())
        self.assertEqual(len(rows), 48)
        for row in rows:
            messages = json.loads(row)["messages"]
            self.assertEqual([m["role"] for m in messages],
                             ["system", "user", "assistant"])
            self.assertIsNotNone(equation_final_answer(messages[-1]["content"]))
            self.assertEqual(messages[-1]["content"].count("FINAL_ANSWER:"), 1)
            self.assertNotIn("<think>", messages[-1]["content"])

    def test_tool_choices_are_actual_valid_json(self):
        rows = trajectory_rows()
        self.assertEqual(len(rows), 50)
        self.assertEqual(rows, trajectory_rows())
        choices = {json.loads(json.loads(row)["messages"][-1]["content"])["action"]
                   for row in rows}
        self.assertEqual(choices, {
            "inventory", "read", "search", "python", "propose_edit",
            "syntax_check", "run_isolated_unit_test", "finish",
            "solve_math_symbolic", "eval_python_expr",
        })

    def test_new_rows_have_no_exact_eval_overlap(self):
        with patch("scripts.neral_v3_combined.load_holdout_prompts", return_value=set()):
            details = check_overlap(
                equation_rows() + trajectory_rows(),
                Path("dummy-holdout"),
                Path("benchmarks/neral_agent_coding_suite.json"),
            )
        self.assertEqual(details["exact_overlap"], 0)
        self.assertEqual(details["verbatim_coding_overlap"], 0)
        self.assertFalse(details["semantic_leakage_excluded"])

    def test_existing_eval_prompt_is_rejected(self):
        bad = json.dumps({"messages": [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": math_cases()[0]["prompt"]},
            {"role": "assistant", "content": "No"},
        ]}, ensure_ascii=False)
        with patch("scripts.neral_v3_combined.load_holdout_prompts", return_value=set()):
            with self.assertRaisesRegex(ValueError, "leakage"):
                check_overlap([bad], Path("dummy"), Path("benchmarks/neral_agent_coding_suite.json"))

    def test_generalization_cases_are_not_hidden(self):
        self.assertEqual(len(math_cases()), 80)
        self.assertEqual(len(checked_cases()), 36)

    def test_new_data_does_not_claim_training(self):
        source = Path("scripts/neral_v3_combined.py").read_text("utf-8")
        self.assertIn('"trained": False', source)
        self.assertIn('"release_ready": False', source)
        self.assertNotIn("model.save_pretrained", source)
