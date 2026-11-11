"""CPU-only gate contracts for the isolated PR533 GPU diagnostic."""
import unittest
from scripts.neral_pr533_two_case_gpu import evaluate_two_case_gate, safe_review_reason


def a(**overrides):
    row = {
        "answer_correct": True, "verified": True,
        "review_method": "exact_user_ast_match",
        "verification_scope": "exact_user_python_execution",
        "tool_used": "neral_python_sandbox",
        "tool_calls_succeeded": 1,
    }
    row.update(overrides)
    return row


def b(**overrides):
    row = {
        "answer_correct": True, "verified": True, "tool_calls_succeeded": 1,
        "tool_used": "neral_math_exact",
        "first_turn_telemetry": {"extraction_rejection_reason": "accepted"},
    }
    row.update(overrides)
    return row


class TwoTaskGate(unittest.TestCase):
    def test_both_strict_gates_pass(self):
        self.assertEqual(evaluate_two_case_gate([a(), b()]), (True, True))

    def test_a_accepts_only_proven_integer_ast_semantics(self):
        self.assertEqual(evaluate_two_case_gate([
            a(review_method="semantic_execution_match",
              verification_scope="proven_integer_ast_equivalence"),
            b()]), (True, True))
        for variation in (
            {"review_method": "semantic_execution_match",
             "verification_scope": "execution_only"},
            {"review_method": "semantic_execution_match",
             "verification_scope": "exact_user_python_execution"},
            {"review_method": "semantic_execution_match",
             "verification_scope": "proven_integer_ast_equivalence",
             "verified": False},
        ):
            with self.subTest(variation=variation):
                self.assertEqual(evaluate_two_case_gate(
                    [a(**variation), b()]), (False, True))

    def test_a_requires_actual_verified_exact_ast(self):
        for variation in (
            {"verified": False},
            {"review_method": None},
            {"review_method": "model_reviewer"},
            {"verification_scope": "execution_only"},
            {"answer_correct": False},
            {"tool_calls_succeeded": 0},
            {"verified": False},
            {"verified": None},
            {"tool_used": None},
        ):
            with self.subTest(variation=variation):
                self.assertEqual(evaluate_two_case_gate([a(**variation), b()]), (False, True))

    def test_b_requires_correct_answer_and_one_successful_tool_call(self):
        for variation in (
            {"answer_correct": False},
            {"tool_calls_succeeded": 0},
        ):
            with self.subTest(variation=variation):
                self.assertEqual(evaluate_two_case_gate([a(), b(**variation)]), (True, False))

    def test_b_parser_rejection_does_not_count_as_pass(self):
        self.assertEqual(
            evaluate_two_case_gate([a(), b(answer_correct=False,
                tool_calls_succeeded=0,
                first_turn_telemetry={"extraction_rejection_reason": "solver_type_error"})]),
            (True, False))

    def test_gpu_workflow_gates_tir12_behind_both_case_success(self):
        from pathlib import Path
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/neral-pr533-gpu-two-case.yml").read_text("utf-8")
        self.assertIn("Execute EXACTLY one code and one logic case", workflow)
        self.assertIn("if: steps.two_case_gate.outcome == 'success'", workflow)
        self.assertIn("scripts.neral_tir12_manual", workflow)
        self.assertIn(
            "ref: 'ca06dc0801d67063829f0436b83ff5b144dd4d54'", workflow)
        self.assertIn("NERAL_PR533_APPROVE_GPU_GATE", workflow)

    def test_b_requires_verification_not_only_solver_execution(self):
        for value in (False, None):
            with self.subTest(value=value):
                self.assertEqual(evaluate_two_case_gate(
                    [a(), b(answer_correct=True, verified=value)]), (True, False))

    def test_review_reason_is_whitelisted_scalar_only(self):
        self.assertEqual(safe_review_reason("ast_structure_mismatch"),
                         "ast_structure_mismatch")
        for category in ("extra_trailing_statements",
                         "leading_setup_statements",
                         "body_statement_modified", "function_wrapped"):
            with self.subTest(category=category):
                self.assertEqual(safe_review_reason(category), category)
        for bad in ("CUSTOM_PRIVATE_USER_CODE", {"secret": "payload"}, ["bad"], 12):
            self.assertIsNone(safe_review_reason(bad))


if __name__ == "__main__":
    unittest.main()
