"""Independent CPU oracle and fail-closed reporting for PR533 fresh logic transfer."""
import unittest

from scripts.neral_pr533_logic_transfer import CASES, count_matches, evaluated_packet


class LogicTransferGateCPU(unittest.TestCase):
    def test_four_distinct_questions_and_independently_enumerated_counts(self):
        self.assertEqual(len(CASES), 4)
        self.assertEqual(len({row["question"] for row in CASES}), 4)
        self.assertEqual([count_matches(row) for row in CASES], [4, 4, 10, 2])
        self.assertTrue(all(row["question"].endswith("Return one integer.")
                            for row in CASES))

    def test_only_model_generated_formal_solver_result_is_eligible(self):
        packet = {
            "text": "FINAL_ANSWER: 4", "verified": False,
            "verification_status": "model_extracted_schema_unverified",
            "extraction_status": "repaired",
            "tool_used": "neral_logic_exact",
            "tool_calls_succeeded": 1,
            "first_turn_telemetry": {
                "solver_repair_status": "repaired",
                "solver_repair_input_format": "plain_json",
            },
        }
        expected = 4
        self.assertTrue(evaluated_packet(packet, 200, expected)["passed"])
        for key, value in (
            ("text", "FINAL_ANSWER: 5"),
            ("verified", True),
            ("verification_status", "base_fallback_unverified"),
            ("extraction_status", "fallback"),
            ("tool_used", "neral_python_sandbox"),
            ("tool_calls_succeeded", 0),
        ):
            with self.subTest(key=key):
                self.assertFalse(evaluated_packet({**packet, key: value},
                                                   200, expected)["passed"])
        self.assertFalse(evaluated_packet(packet, 503, expected)["passed"])
        self.assertFalse(evaluated_packet({}, 503, expected)["passed"])


if __name__ == "__main__":
    unittest.main()
