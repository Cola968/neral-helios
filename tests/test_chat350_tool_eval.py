from __future__ import annotations

import json
import unittest
from pathlib import Path

from frontier350.eval_chat350_tools import evaluate_tool_layer, last_user_message
from frontier350.build_chat350_stress_cases import generate_cases


class ToolEvaluationTests(unittest.TestCase):
    def test_math_stress_generalization(self):
        # Fixed generator seed but novel values; never used as training data.
        cases = [x for x in generate_cases(3502026) if x["category"] == "math"]
        report = evaluate_tool_layer(cases)
        self.assertEqual(report["mode"], "deterministic-tool-only")
        self.assertTrue(report["not_model_native"])
        self.assertEqual(report["cases"], 16)
        self.assertEqual(report["covered"], 16)
        self.assertEqual(report["correct"], 16)

    def test_frozen_math_contracts(self):
        cases = json.loads(
            (Path(__file__).resolve().parents[1] /
             "frontier350/chat350_capability_cases.json").read_text(encoding="utf8")
        )
        maths = [x for x in cases if x["category"] == "math"]
        result = evaluate_tool_layer(maths)
        self.assertEqual(result["cases"], 8)
        self.assertEqual(result["covered"], 8)
        self.assertEqual(result["correct"], 8)

    def test_abstains_on_non_tool_tasks(self):
        cases = [
            {"id": "no-fact", "category": "science",
             "prefix": "User: Explain why water freezes.\nAssistant:",
             "scoring": {"type": "contains_all", "values": ["ice"]}},
            {"id": "no-code", "category": "code",
             "prefix": "User: Write a secure web server.\nAssistant:",
             "scoring": {"type": "contains_all", "values": ["server"]}},
        ]
        report = evaluate_tool_layer(cases)
        self.assertEqual(report["covered"], 0)
        self.assertEqual(report["correct"], 0)

    def test_last_user_turn_only(self):
        prefix = "User: Berechne 10 + 10.\nAssistant: 20\nUser: Hello!\nAssistant:"
        self.assertEqual(last_user_message(prefix), "Hello!")
        with self.assertRaises(ValueError):
            last_user_message("not a chat prefix")


if __name__ == "__main__":
    unittest.main()
