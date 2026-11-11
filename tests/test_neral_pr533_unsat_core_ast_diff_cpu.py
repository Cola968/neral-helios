"""PR533 CPU regression: first AST mismatch and bounded subset-minimal UNSAT core."""
import json
import unittest

from neral_turn0_extractor import parse_extracted_constraints
from neral_tir_experimental import _exact_source_python_answer, run_experimental_tir
from scripts.neral_tir_hard_live import safe_first_turn_event


def schema_packet(question, clauses, *, evidence=None, variables=None):
    if evidence is None:
        evidence = [
            "How many Boolean assignments to A,B",
            "Return one integer",
        ]
    return json.dumps({
        "kind": "formal", "complete": True,
        "schema": {
            "variables": variables or {"A": [False, True], "B": [False, True]},
            "rules": [], "bool_rules": clauses, "query": "count"},
        "evidence": evidence,
    })


def analyze(question, clauses, *, evidence=None, variables=None):
    telemetry = []
    result = parse_extracted_constraints(
        question, schema_packet(question, clauses, evidence=evidence,
                                variables=variables),
        structural_telemetry_sink=telemetry)
    return result, telemetry[0]


class AstFirstMismatchNodeCPU(unittest.TestCase):
    def compare(self, source, executed):
        reasons, nodes = [], []
        result = _exact_source_python_answer(
            "What integer does this Python program print:\n" + source,
            "python_sandbox", executed, "5", "",
            strict_integer=True, diagnostic=reasons, ast_diff_node_sink=nodes)
        return result, reasons, nodes

    def test_reports_first_changed_statement_class_not_source_text(self):
        for before, after, expected in (
            ("a=3\nprint(a)", "a=4\nprint(a)", "Assign"),
            ("a=3\nprint(a)", "a=3\nprint(a+2)", "Call"),
            ("a=3\na+=2\nprint(a)", "a=3\na-=2\nprint(a)", "AugAssign"),
            ("for i in range(2):\n    pass\nprint(5)",
             "for i in range(3):\n    pass\nprint(5)", "For"),
            ("def f():\n    return 2\nprint(f())",
             "def f():\n    return 3\nprint(f())", "FunctionDef"),
        ):
            with self.subTest(kind=expected):
                result, reasons, nodes = self.compare(before, after)
                self.assertIsNone(result)
                self.assertEqual(reasons, ["body_statement_modified"])
                self.assertEqual(nodes, [expected])

    def test_fresh_alias_of_original_print_expression_is_output_only(self):
        source = "value=2\nprint(value+3)"
        executed = "value=2\n_result=value+3\nprint(_result)"
        answer, reasons, nodes = self.compare(source, executed)
        self.assertEqual(answer, "FINAL_ANSWER: 5")
        self.assertEqual(reasons, [])
        self.assertEqual(nodes, [])

    def test_changed_print_value_cannot_be_verified_as_read_only(self):
        source = "value=2\nprint(value+3)"
        for executed in (
            "value=2\n_result=value+4\nprint(_result)",
            "value=2\nprint(value+4)",
            "value=2\nprint(5)",
            "value=2\nprint(value+3)\nprint(value+3)",
        ):
            with self.subTest(executed=executed):
                answer, reasons, _ = self.compare(source, executed)
                self.assertIsNone(answer)
                self.assertTrue(reasons)

    def test_matched_ast_and_proven_constant_fold_do_not_emit_diff(self):
        for src, changed in (
            ("v=2+3\nprint(v)", "v=5\nprint(v)"),
            ("v=5\nprint(v)", "v=5\nprint(v)"),
        ):
            answer, reasons, nodes = self.compare(src, changed)
            self.assertEqual(answer, "FINAL_ANSWER: 5")
            self.assertEqual(reasons, [])
            self.assertEqual(nodes, [])

    def test_end_to_end_rejection_has_only_safe_node_name(self):
        calls = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"a=4\\nprint(a)"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 4",
        ])
        question = "What integer does this Python program print:\na=3\nprint(a)"
        result = run_experimental_tir(
            [{"role": "system", "content": "NERAL"},
             {"role": "user", "content": question}],
            lambda _: next(calls),
            tool_runner=lambda *_: {
                "status": "PASSED", "output": "4", "stderr": "",
                "verification_scope": "execution_only"})
        self.assertFalse(result["verified"])
        self.assertEqual(result["review_match_reason"], "body_statement_modified")
        self.assertEqual(result["ast_diff_node"], "Assign")


class MinimalUnsatCoreCPU(unittest.TestCase):
    def test_core_drops_irrelevant_clause_and_keeps_contradiction(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; A is False? Return one integer.")
        clauses = [
            {"op": "eq", "args": ["A", True]},
            {"op": "eq", "args": ["A", False]},
            {"op": "eq", "args": ["B", True]},
        ]
        result, t = analyze(q, clauses)
        self.assertEqual(result, 0)
        self.assertEqual(t["unsat_core_size"], 2)
        self.assertNotIn("relaxed_constraint_index", t)

    def test_nongrounded_connected_clause_is_single_drop_diagnostic(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is False; B is False? Return one integer.")
        clauses = [
            {"op": "or", "args": ["A", "B"]},
            {"op": "eq", "args": ["A", False]},
            {"op": "eq", "args": ["B", False]},
        ]
        result, t = analyze(q, clauses, evidence=[
            "How many Boolean assignments to A,B",
            "A is False", "B is False", "Return one integer"])
        self.assertEqual(result, 0)
        self.assertEqual(t["unsat_core_size"], 3)
        self.assertEqual(t["relaxed_constraint_index"], 0)
        self.assertEqual(t["solver_relaxation_status"], "single_candidate")
        self.assertEqual(t["solver_relaxation_count_bucket"], "single_solution")

    def test_nested_and_core_uses_stable_flattened_index(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is False; B is False? Return one integer.")
        clauses = [{"op": "and", "args": [
            {"op": "or", "args": ["A", "B"]},
            {"op": "eq", "args": ["A", False]},
            {"op": "eq", "args": ["B", False]}]}]
        result, t = analyze(q, clauses)
        self.assertEqual(result, 0)
        self.assertEqual(t["unsat_core_size"], 3)
        self.assertEqual(t["relaxed_constraint_index"], 0)

    def test_contradictory_source_clues_never_autorelax_even_if_one_solution(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; A is False? Return one integer.")
        clauses = [
            {"op": "eq", "args": ["A", True]},
            {"op": "eq", "args": ["A", False]},
            {"op": "eq", "args": ["B", True]}]
        _, t = analyze(q, clauses)
        self.assertEqual(t["unsat_core_size"], 2)
        self.assertNotIn("relaxed_constraint_index", t)

    def test_singleton_self_contradiction_is_indexed_and_recounted_unverified(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; B is False? Return one integer.")
        clauses = [
            {"op":"eq","args":["A",{"op":"not","args":["A"]}]},
            {"op":"eq","args":["A",True]},
            {"op":"eq","args":["B",False]},
        ]
        result, event = analyze(q, clauses, evidence=[
            "How many Boolean assignments to A,B",
            "A is True", "B is False", "Return one integer"])
        self.assertEqual(result, 0)
        self.assertEqual(event["unsat_core_size"], 1)
        self.assertEqual(event["unsat_core_index"], 0)
        self.assertEqual(event["relaxed_constraint_index"], 0)
        self.assertEqual(event["solver_relaxation_count_bucket"], "single_solution")
        self.assertEqual(event["solver_relaxation_status"], "single_candidate")

    def test_directly_stated_self_contradiction_is_not_discarded(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A equals NOT A; B is False? Return one integer.")
        clauses = [
            {"op":"eq","args":["A",{"op":"not","args":["A"]}]},
            {"op":"eq","args":["B",False]},
        ]
        result, event = analyze(q, clauses, evidence=[
            "How many Boolean assignments to A,B",
            "A equals NOT A", "B is False", "Return one integer"])
        self.assertEqual(result, 0)
        self.assertEqual(event["unsat_core_size"], 1)
        self.assertEqual(event["unsat_core_index"], 0)
        self.assertNotIn("relaxed_constraint_index", event)

    def test_singleton_unsat_multiple_remaining_solutions_is_diagnostic_not_proof(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A can be true or false; B can be true or false? Return one integer.")
        clauses = [
            {"op":"eq","args":["A",{"op":"not","args":["A"]}]},
        ]
        result,event=analyze(q,clauses)
        self.assertEqual(result,0)
        self.assertEqual(event["unsat_core_index"],0)
        self.assertEqual(event["post_relaxation_solution_count"],4)
        self.assertEqual(event["relaxed_constraint_index"],0)
        self.assertEqual(event["solver_relaxation_status"],"single_candidate")
        self.assertEqual(event["solver_relaxation_count_bucket"],"multiple_solutions")

    def test_large_assignment_space_is_not_exhaustively_recomputed(self):
        q = ("How many integer assignments to A,B,C,D,E,F,G,H satisfy ALL: "
             "A equals zero; A equals one? Return one integer.")
        variables = {name: list(range(4)) for name in "ABCDEFGH"}
        obj = {
            "kind": "formal", "complete": True,
            "schema": {"variables": variables,
                       "rules": [{"left": "A", "op": "==", "value": 0},
                                 {"left": "A", "op": "==", "value": 1}],
                       "bool_rules": [], "query": "count"},
            "evidence": ["How many integer assignments to A,B,C,D,E,F,G,H",
                         "A equals zero", "A equals one", "Return one integer"]}
        events = []
        self.assertEqual(parse_extracted_constraints(q, json.dumps(obj),
                              structural_telemetry_sink=events), 0)
        self.assertNotIn("unsat_core_size", events[0])

    def test_telemetry_rejects_excessive_or_malformed_core_metadata(self):
        valid = {"decision": "extraction_fallback", "done_reason": "stop",
                 "unsat_core_size": 1, "relaxed_constraint_index": 0}
        valid["unsat_core_index"] = 0
        self.assertEqual(safe_first_turn_event(valid)["unsat_core_index"], 0)
        self.assertEqual(safe_first_turn_event(valid)["unsat_core_size"], 1)
        self.assertEqual(safe_first_turn_event(valid)["relaxed_constraint_index"], 0)
        for value in (-1, True, 50, "private text"):
            with self.subTest(value=value):
                invalid = {**valid, "unsat_core_size": value}
                self.assertIsNone(safe_first_turn_event(invalid))
        invalid = {**valid, "relaxed_constraint_index": "secret"}
        self.assertIsNone(safe_first_turn_event(invalid))


if __name__ == "__main__":
    unittest.main()
