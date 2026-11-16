"""Bounded structural diagnostics without silently verifying altered user tasks."""
import json
import unittest

from neral_turn0_extractor import parse_extracted_constraints
from neral_tir_experimental import _exact_source_python_answer, run_experimental_tir
from scripts.neral_tir_hard_live import safe_first_turn_event


def inspect_logic(question, bool_rules):
    evidence = ["How many Boolean assignments to A,B", "Return one integer"]
    clues = question.split("ALL: ", 1)[-1].split("? Return", 1)[0]
    evidence.extend(part.strip() for part in clues.split(";"))
    payload = json.dumps({
        "kind": "formal", "complete": True,
        "schema": {
            "variables": {"A": [False, True], "B": [False, True]},
            "rules": [], "bool_rules": bool_rules, "query": "count",
        },
        "evidence": evidence[:12],
    })
    events = []
    number = parse_extracted_constraints(
        question, payload, structural_telemetry_sink=events)
    return number, events[0]


class SingletonUnsatStructureCPU(unittest.TestCase):
    def test_singleton_op_literal_and_remaining_count_for_count_task(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; B is False? Return one integer.")
        contradiction = {"op": "eq", "args": [
            "A", {"op": "not", "args": ["A"]},
        ]}
        count, event = inspect_logic(q, [
            contradiction,
            {"op": "eq", "args": ["A", True]},
            {"op": "eq", "args": ["B", False]},
        ])
        self.assertEqual(count, 0)
        self.assertEqual(event["unsat_core_size"], 1)
        self.assertEqual(event["unsat_core_index"], 0)
        self.assertEqual(event["unsat_clause_op"], "eq")
        self.assertIs(event["unsat_clause_has_literal"], False)
        self.assertEqual(event["post_relaxation_solution_count"], 1)
        self.assertEqual(event["solver_relaxation_status"], "single_candidate")

    def test_two_or_more_solutions_count_as_diagnostic_candidate(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A can be either value; B can be either value? Return one integer.")
        contradiction = {"op": "eq", "args": [
            "A", {"op": "not", "args": ["A"]},
        ]}
        count, event = inspect_logic(q, [contradiction])
        self.assertEqual(count, 0)
        self.assertEqual(event["unsat_core_index"], 0)
        self.assertEqual(event["unsat_clause_op"], "eq")
        self.assertEqual(event["post_relaxation_solution_count"], 4)
        self.assertEqual(event["solver_relaxation_status"], "single_candidate")
        self.assertEqual(event["solver_relaxation_count_bucket"], "multiple_solutions")
        self.assertEqual(event["relaxed_constraint_index"], 0)

    def test_protected_source_contradiction_still_reports_remaining_count(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A equals NOT A; B is False? Return one integer.")
        contradiction = {"op": "eq", "args": [
            "A", {"op": "not", "args": ["A"]},
        ]}
        count, event = inspect_logic(q, [
            contradiction, {"op": "eq", "args": ["B", False]},
        ])
        self.assertEqual(count, 0)
        self.assertEqual(event["unsat_core_size"], 1)
        self.assertEqual(event["post_relaxation_solution_count"], 2)
        self.assertEqual(event["solver_relaxation_status"], "source_clause_protected")
        self.assertNotIn("relaxed_constraint_index", event)

    def test_literal_in_intrinsically_false_clause_is_reported(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; B is False? Return one integer.")
        count, event = inspect_logic(q, [
            {"op": "eq", "args": [True, False]},
            {"op": "eq", "args": ["A", True]},
            {"op": "eq", "args": ["B", False]},
        ])
        self.assertEqual(count, 0)
        self.assertEqual(event["unsat_clause_op"], "eq")
        self.assertIs(event["unsat_clause_has_literal"], True)
        self.assertEqual(event["post_relaxation_solution_count"], 1)
        # No referenced variable -> no positive evidence that this clause
        # is an extraction artifact; fail closed.
        self.assertNotIn("relaxed_constraint_index", event)

    def test_scalar_telemetry_validates_enum_counts_and_bool_strictly(self):
        event = {
            "decision": "extraction_fallback", "done_reason": "stop",
            "unsat_core_size": 1, "unsat_core_index": 5,
            "unsat_clause_op": "at_least",
            "unsat_clause_has_literal": True,
            "post_relaxation_solution_count": 128,
            "solver_relaxation_status": "single_candidate",
            "solver_relaxation_count_bucket": "multiple_solutions",
        }
        safe = safe_first_turn_event(event)
        self.assertEqual(safe["unsat_clause_op"], "at_least")
        self.assertIs(safe["unsat_clause_has_literal"], True)
        self.assertEqual(safe["post_relaxation_solution_count"], 128)
        for key, bad in (
            ("unsat_clause_op", "raw_private_clause"),
            ("unsat_clause_has_literal", 1),
            ("post_relaxation_solution_count", True),
            ("post_relaxation_solution_count", 99999),
            ("post_relaxation_solution_count", "4"),
        ):
            with self.subTest(key=key, value=bad):
                self.assertIsNone(safe_first_turn_event({**event, key: bad}))


class ExprMismatchStructureCPU(unittest.TestCase):
    def review(self, source, executed):
        details = []
        reasons = []
        reviewed = _exact_source_python_answer(
            "What integer does this Python program print:\n" + source,
            "python_sandbox", executed, "5", "",
            strict_integer=True, diagnostic=reasons,
            ast_diff_details_sink=details)
        return reviewed, reasons, details

    def test_changed_terminal_call_is_classified_with_index_and_counts(self):
        reviewed, reasons, details = self.review(
            "x=2\nprint(x)", "x=2\nprint(x+1)")
        self.assertIsNone(reviewed)
        self.assertEqual(reasons, ["body_statement_modified"])
        self.assertEqual(details, [{
            "ast_diff_expr_type": "Call_vs_Call",
            "ast_diff_statement_index": 1,
            "user_stmt_count": 2, "exec_stmt_count": 2,
        }])

    def test_changed_mid_program_expr_reports_subtype_and_offset(self):
        reviewed, reasons, details = self.review(
            "x=2\nx\nprint(x)", "x=2\nx+1\nprint(x)")
        self.assertIsNone(reviewed)
        self.assertEqual(details[0]["ast_diff_expr_type"], "Name_vs_BinOp")
        self.assertEqual(details[0]["ast_diff_statement_index"], 1)
        self.assertEqual(details[0]["user_stmt_count"], 3)
        self.assertEqual(details[0]["exec_stmt_count"], 3)

    def test_inserted_expression_reports_length_mismatch_without_copying_text(self):
        reviewed, reasons, details = self.review(
            "x=2\nprint(x)", "x=2\nx\nprint(x)")
        self.assertIsNone(reviewed)
        self.assertEqual(details[0]["ast_diff_statement_index"], 1)
        self.assertEqual(details[0]["ast_diff_expr_type"], "Call_vs_Name")
        self.assertEqual(details[0]["user_stmt_count"], 2)
        self.assertEqual(details[0]["exec_stmt_count"], 3)

    def test_equivalent_output_alias_does_not_emit_diff(self):
        reviewed, reasons, details = self.review(
            "v=2\nprint(v+3)", "v=2\n_result=v+3\nprint(_result)")
        self.assertEqual(reviewed, "FINAL_ANSWER: 5")
        self.assertEqual(reasons, [])
        self.assertEqual(details, [])

    def test_end_to_end_unverified_answer_emits_only_bounded_structures(self):
        calls = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"x=2\\nprint(x+1)"}',
            "NERAL_FINAL_TEXT: 5",
        ])
        result = run_experimental_tir(
            [{"role": "system", "content": "NERAL"},
             {"role": "user", "content": "Earlier discussion."},
             {"role": "assistant", "content": "Prior reply."},
             {"role": "user", "content":
              "What integer does this Python program print:\nx=2\nprint(x)"}],
            lambda _: next(calls),
            tool_runner=lambda *_: {
                "status": "PASSED", "output": "5", "stderr": "",
                "verification_scope": "execution_only",
            })
        self.assertFalse(result["verified"])
        self.assertEqual(result["review_match_reason"], "body_statement_modified")
        self.assertEqual(result["ast_diff_expr_type"], "Call_vs_Call")
        self.assertEqual(result["ast_diff_statement_index"], 1)
        self.assertEqual(result["user_stmt_count"], 2)
        self.assertEqual(result["exec_stmt_count"], 2)


if __name__ == "__main__":
    unittest.main()
