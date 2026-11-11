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

    def test_nested_conjunction_is_only_flattened_for_single_drop(self):
        events = []
        candidate = propose(
            {"op":"and","args":[OR,B_FALSE,A_FALSE]}, LITERAL)
        self.assertEqual(parse_extracted_constraints(
            Q, candidate, structural_telemetry_sink=events), 0)
        self.assertEqual(events[0]["solver_relaxation_status"],
                         "single_candidate")

    def test_two_unsupported_conflicting_literals_are_ambiguous(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A OR B is True; B can be True or False? Return one integer.")
        data = json.dumps({
            "kind":"formal","complete":True,
            "schema":{"variables":{"A":[False,True],"B":[False,True]},
                      "rules":[],"bool_rules":[
                          OR,
                          {"op":"eq","args":["A", True]},
                          {"op":"eq","args":["A", False]},
                          LITERAL],
                      "query":"count"},
            "evidence":["How many Boolean assignments to A,B",
                        "A OR B is True", "B can be True or False",
                        "Return one integer"]})
        events = []
        self.assertEqual(parse_extracted_constraints(
            q, data, structural_telemetry_sink=events), 0)
        self.assertEqual(events[0]["solver_relaxation_status"],
                         "ambiguous_candidates")
        self.assertNotIn("solver_relaxation_count_bucket", events[0])

    def test_two_drops_required_never_relax_as_one_step(self):
        events = []
        self.assertEqual(parse_extracted_constraints(
            Q, propose(OR,B_FALSE,A_FALSE,A_FALSE,LITERAL),
            structural_telemetry_sink=events), 0)
        self.assertEqual(events[0]["solver_relaxation_status"],
                         "no_positive_single_drop")

    def test_safe_telemetry_exports_only_fixed_relaxation_categories(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        event = {"decision":"extraction_fallback",
                 "done_reason":"stop", "eval_count":302,
                 "prompt_eval_count":864,
                 "solver_solution_count_bucket":"zero_unsat",
                 "solver_relaxation_status":"single_candidate",
                 "solver_relaxation_count_bucket":"single_solution"}
        self.assertEqual(safe_first_turn_event(event)["solver_relaxation_status"],
                         "single_candidate")
        event["solver_relaxation_status"] = "private source leak"
        self.assertIsNone(safe_first_turn_event(event))

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

    def test_full_tir_api_prefers_exact_user_code_over_model_rewrite(self):
        question = ("What integer does this Python program print:" +
                    chr(10) + "value=2+3" + chr(10) + "print(value)")
        seen = []
        def runner(tool, code):
            seen.append((tool, code))
            return {"status": "PASSED", "output": "5", "stderr": "",
                    "verification_scope": "execution_only"}
        result = run_experimental_tir(
            [{"role":"system","content":"NERAL"},
             {"role":"user","content":question + SUFFIX}],
            lambda _: self.fail("Unchanged source must bypass model rewrite"),
            tool_runner=runner)
        self.assertEqual(seen, [("python_sandbox", "value=2+3\\nprint(value)")])
        self.assertTrue(result["verified"])
        self.assertEqual(result["review_method"], "exact_user_ast_match")
        self.assertEqual(result["verification_scope"],
                         "exact_user_python_execution")
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
