"""CPU-only UNSAT single-clause diagnostics and proven integer AST rewrites."""
import json
import unittest

from neral_turn0_extractor import parse_extracted_constraints
from neral_tir_experimental import _exact_source_python_answer, run_experimental_tir

SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."
Q = ("How many Boolean assignments to A,B satisfy ALL: "
     "A OR B is True; B is False? Return one integer.")


def propose(*clauses):
    return json.dumps({
        "kind": "formal", "complete": True,
        "schema": {
            "variables": {"A": [False, True], "B": [False, True]},
            "rules": [], "bool_rules": list(clauses), "query": "count",
        },
        "evidence": [
            "How many Boolean assignments to A,B",
            "A OR B is True", "B is False", "Return one integer"
        ],
    })


OR = {"op": "or", "args": ["A", "B"]}
B_FALSE = {"op": "eq", "args": ["B", "false"]}
A_FALSE = {"op": "eq", "args": ["A", "false"]}
LITERAL = {"op": "at_least", "min": 1, "args": ["A", "B", "true"]}


class OneStepRelaxationCPU(unittest.TestCase):
    def test_unsupported_literal_assignment_is_retried_once_without_automatic_proof(self):
        events, reason = [], []
        n = parse_extracted_constraints(
            Q, propose(OR, B_FALSE, A_FALSE, LITERAL),
            rejection_reason_sink=reason, structural_telemetry_sink=events)
        self.assertEqual(n, 0)
        self.assertEqual(reason, ["accepted"])
        self.assertEqual(events[0]["solver_solution_count_bucket"], "zero_unsat")
        self.assertEqual(events[0]["extracted_constraint_count"], 4)
        self.assertEqual(events[0]["expected_sentence_clues"], 2)
        self.assertEqual(events[0]["boolean_literal_context"], "multiple")
        self.assertEqual(events[0]["solver_relaxation_status"], "single_candidate")
        self.assertEqual(events[0]["solver_relaxation_count_bucket"], "single_solution")

    def test_no_relaxation_if_candidate_is_directly_quoted(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; A is False? Return one integer.")
        data = json.dumps({
            "kind": "formal", "complete": True,
            "schema": {"variables": {"A":[False,True], "B":[False,True]},
                       "rules": [], "bool_rules": [
                           {"op":"eq","args":["A",True]},
                           {"op":"eq","args":["A",False]},
                           {"op":"at_least","min":1,"args":["A","B",True]}
                       ],"query":"count"},
            "evidence":["How many Boolean assignments to A,B",
                        "A is True", "A is False", "Return one integer"]})
        events = []
        self.assertEqual(parse_extracted_constraints(
            q, data, structural_telemetry_sink=events), 0)
        self.assertEqual(events[0]["solver_relaxation_status"],
                         "no_ungrounded_literal")

    def test_raw_baseline_fallback_remains_unverified_even_with_candidate(self):
        responses=[]
        result=run_experimental_tir(
            [{"role":"system","content":"NERAL"},
             {"role":"user","content":Q + SUFFIX}],
            lambda turns: (responses.append(turns) or "FINAL_ANSWER: 0"),
            extract_constraints=lambda _:propose(OR,B_FALSE,A_FALSE,LITERAL))
        self.assertFalse(result["verified"])
        self.assertEqual(result["verification_status"],"base_fallback_unverified")
        self.assertEqual(result["first_turn_telemetry"]["solver_relaxation_status"],
                         "single_candidate")
        self.assertEqual(len(responses),1)


class StrictSemanticIntegerProofCPU(unittest.TestCase):
    def check(self, source, executed, stdout="5"):
        reasons, methods=[], []
        answer = _exact_source_python_answer(
            "What integer does this Python program print:\n" + source,
            "python_sandbox", executed, stdout, "",
            strict_integer=True, diagnostic=reasons,
            review_method_sink=methods)
        return answer, reasons, methods

    def test_constant_fold_preserves_pure_integer_program(self):
        answer, reasons, methods = self.check(
            "value=2+3\nprint(value)",
            "value=5\nprint(value)")
        self.assertEqual(answer,"FINAL_ANSWER: 5")
        self.assertEqual(reasons,[])
        self.assertEqual(methods,["semantic_execution_match"])

    def test_constant_fold_and_terminal_print_instrumentation(self):
        answer,reasons,methods=self.check(
            "value=2*3-1",
            "value=5\nprint(value)")
        self.assertEqual(answer,"FINAL_ANSWER: 5")
        self.assertEqual(methods,["semantic_execution_match"])

    def test_unproven_mutations_never_receive_semantic_verification(self):
        for changed in (
            "value=2+4\nprint(value)",
            "value=999\nprint(value)",
            "value=5\nprint(999)",
            "print(5)",
            "value=2+3\nprint(5)",
        ):
            with self.subTest(changed=changed):
                answer,reasons,methods=self.check(
                    "value=2+3\nprint(value)",changed)
                self.assertIsNone(answer)
                self.assertEqual(methods, [])
                self.assertEqual(reasons,["body_statement_modified"])

    def test_full_tir_api_reports_proven_semantic_method(self):
        responses = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox",'
            '"code":"value=5\\nprint(value)"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 5",
        ])
        question = ("What integer does this Python program print:" +
                    chr(10) + "value=2+3" + chr(10) + "print(value)")
        result = run_experimental_tir(
            [{"role":"system","content":"NERAL"},
             {"role":"user","content":question + SUFFIX}],
            lambda _: next(responses),
            tool_runner=lambda *_: {"status":"PASSED", "output":"5",
                                     "stderr":"", "verification_scope":"execution_only"})
        self.assertTrue(result["verified"])
        self.assertEqual(result["review_method"], "semantic_execution_match")
        self.assertEqual(result["verification_scope"],
                         "proven_integer_ast_equivalence")
        self.assertEqual(result["text"], "FINAL_ANSWER: 5")

    def test_no_unjustified_verification_for_overloaded_or_dynamic_arithmetic(self):
        for before, after in (
            ("value=x+3\nprint(value)","value=5\nprint(value)"),
            ("value=3**9\nprint(value)","value=19683\nprint(value)"),
            ("value=2/1\nprint(value)","value=2\nprint(value)"),
        ):
            with self.subTest(before=before):
                result, _, methods = self.check(before,after)
                self.assertIsNone(result)
                self.assertEqual(methods,[])


if __name__ == "__main__":
    unittest.main()
