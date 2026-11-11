"""Regression of PR533 live GPU rejection categories (CPU only)."""
import json
import unittest

from neral_formal_constraints import (
    PREFIX, InvalidFormalConstraints, try_solve_formal_constraints,
)
from neral_turn0_extractor import parse_extracted_constraints
from neral_tir_experimental import _exact_source_python_answer


class BooleanCardinalityCPU(unittest.TestCase):
    @staticmethod
    def solve(expr):
        payload = {
            "variables": {name: [False, True] for name in "ABC"},
            "rules": [], "bool_rules": [expr], "query": "count",
        }
        return try_solve_formal_constraints(PREFIX + json.dumps(payload))

    def test_at_least_accepts_literal_and_nested_boolean_expression(self):
        expr = {"op": "at_least", "min": 2,
                "args": ["A", True, {"op": "and", "args": ["B", "C"]}]}
        self.assertEqual(self.solve(expr), 5)

    def test_exactly_accepts_nary_terms_with_boolean_literal(self):
        expr = {"op": "exactly", "count": 2,
                "args": ["A", "B", "C", False]}
        self.assertEqual(self.solve(expr), 3)

    def test_nary_and_or_and_comparison_to_literal(self):
        expr = {"op": "and", "args": [
            {"op": "or", "args": ["A", "B", "C"]},
            {"op": "eq", "args": ["A", True]},
            {"op": "not", "args": [False]},
        ]}
        self.assertEqual(self.solve(expr), 4)

    def test_cardinality_bounds_and_duplicate_variables_fail_closed(self):
        for expression in [
            {"op": "at_least", "min": -1, "args": ["A", "B"]},
            {"op": "at_least", "min": 3, "args": ["A", "B"]},
            {"op": "at_least", "min": True, "args": ["A", "B"]},
            {"op": "at_least", "min": 1, "args": ["A", "A"]},
            {"op": "exactly", "count": 4, "args": ["A", "B"]},
        ]:
            with self.subTest(expression=expression):
                with self.assertRaises(InvalidFormalConstraints):
                    self.solve(expression)

    def test_extractor_literal_is_not_counted_as_variable(self):
        question = ("How many Boolean assignments to A,B satisfy ALL: "
                    "At least two of A, B, True are True? Return one integer.")
        proposal = {"kind": "formal", "complete": True, "schema": {
            "variables": {"A": [False, True], "B": [False, True]},
            "rules": [], "bool_rules": [{
                "op": "at_least", "min": 2, "args": ["A", "B", "true"],
            }], "query": "count"},
            "evidence": ["How many Boolean assignments to A,B",
                         "At least two of A, B, True are True",
                         "Return one integer"]}
        reasons, details, ast_tags = [], [], []
        answer = parse_extracted_constraints(question, json.dumps(proposal),
            rejection_reason_sink=reasons,
            solver_rejection_detail_sink=details,
            ast_check_code_sink=ast_tags)
        self.assertEqual(answer, 3)
        self.assertEqual(reasons, ["accepted"])
        self.assertEqual(details, [])
        self.assertEqual(ast_tags, ["boolean_literal_token"])


class ExactUserAstOutputCPU(unittest.TestCase):
    def review(self, source, executed, stdout="5"):
        reasons = []
        result = _exact_source_python_answer(
            "What integer does this Python program output:\n" + source,
            "python_sandbox", executed, stdout, "",
            strict_integer=True, diagnostic=reasons)
        return result, reasons

    def test_fresh_typed_alias_of_final_assignment_is_verifiable(self):
        self.assertEqual(self.review(
            "value = 2 + 3",
            "value = 2+3\nanswer: int = value\nprint(answer)"),
            ("FINAL_ANSWER: 5", []))

    def test_fresh_alias_of_final_expression_is_verifiable(self):
        self.assertEqual(self.review(
            "x=2\nx+3",
            "x=2\nx+3\nanswer=x+3\nprint(answer)"),
            ("FINAL_ANSWER: 5", []))

    def test_unrelated_helper_or_mutation_does_not_verify(self):
        for code in (
            "value=2+3\nanswer=999\nprint(answer)",
            "value=2+3\nvalue=6\nprint(value)",
            "value=2+4\nanswer=value\nprint(answer)",
            "value=2+3\nanswer: int = 999\nprint(answer)",
        ):
            with self.subTest(code=code):
                result, reasons = self.review("value=2+3", code)
                self.assertIsNone(result)
                self.assertEqual(reasons, ["ast_structure_mismatch"])

    def test_extra_prints_cannot_change_original_program_output(self):
        result, reasons = self.review(
            "print(2+3)", "print(2+3)\nprint(2+3)", "5\n5")
        self.assertIsNone(result)
        self.assertEqual(reasons, ["python_no_eligible_execution"])


if __name__ == "__main__":
    unittest.main()
