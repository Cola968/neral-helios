"""CPU-only red/green diagnostics for model-authored logic and Python AST."""
import json
import unittest

from neral_turn0_extractor import parse_extracted_constraints
from neral_tir_experimental import run_experimental_tir, _exact_source_python_answer
from scripts.neral_tir_hard_live import safe_first_turn_event

SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."


def packet(question, nodes, *, vars=("A", "B")):
    evidence = [question.split(" satisfy ALL: ")[0],
                question.split("satisfy ALL: ")[1].split("?")[0],
                "Return one integer"]
    return json.dumps({
        "kind": "formal", "complete": True,
        "schema": {
            "variables": {x: [False, True] for x in vars},
            "rules": [], "bool_rules": nodes, "query": "count",
        }, "evidence": evidence,
    })


class SolverScalarDiagnostics(unittest.TestCase):
    def test_zero_model_extracted_count_falls_back_not_final_answer_zero(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A is True; A is False? Return one integer.")
        proposed = packet(q, [
            {"op": "eq", "args": ["A", True]},
            {"op": "eq", "args": ["A", False]}])
        calls = []
        def respond(turns):
            calls.append(turns)
            return "Fallback was used"
        result = run_experimental_tir(
            [{"role": "system", "content": "NERAL"},
             {"role": "user", "content": q + SUFFIX}],
            respond, extract_constraints=lambda _: proposed)
        self.assertEqual(result["text"], "Fallback was used")
        self.assertFalse(result["verified"])
        self.assertEqual(len(calls), 1)
        event = result["first_turn_telemetry"]
        self.assertEqual(event["extraction_rejection_reason"], "solver_unsat_fallback")
        self.assertEqual(event["solver_solution_count_bucket"], "zero_unsat")
        self.assertEqual(event["extracted_constraint_count"], 2)
        self.assertEqual(event["expected_sentence_clues"], 2)
        self.assertEqual(safe_first_turn_event(event)["solver_solution_count_bucket"],
                         "zero_unsat")

    def test_positive_count_and_semicolon_clues_are_observations_not_proof(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "A OR B is True; A is True? Return one integer.")
        proposed = packet(q, [{"op": "eq", "args": ["A", True]}])
        event = []
        result = parse_extracted_constraints(q, proposed,
                structural_telemetry_sink=event)
        self.assertEqual(result, 2)
        self.assertEqual(event[0]["solver_solution_count_bucket"], "multiple_solutions")
        self.assertEqual(event[0]["extracted_constraint_count"], 1)
        self.assertEqual(event[0]["expected_sentence_clues"], 2)

    def test_literal_context_distinguishes_cardinality_and_comparison(self):
        q = ("How many Boolean assignments to A,B satisfy ALL: "
             "At least two of A B True hold; B is True? Return one integer.")
        tree = [
            {"op": "at_least", "min": 2, "args": ["A", "B", "true"]},
            {"op": "eq", "args": ["B", "true"]}]
        events, tags = [], []
        result = parse_extracted_constraints(q, packet(q, tree),
            structural_telemetry_sink=events, ast_check_code_sink=tags)
        self.assertEqual(result, 1)
        self.assertEqual(tags, ["boolean_literal_token"])
        self.assertEqual(events[0]["boolean_literal_context"], "multiple")
        self.assertEqual(events[0]["extracted_constraint_count"], 2)

    def test_zero_is_valid_for_explicit_trusted_schema(self):
        from neral_formal_constraints import PREFIX
        schema = {"variables": {"A": [False, True]}, "rules": [],
                  "bool_rules": [{"op": "eq", "args": ["A", True]},
                                 {"op": "eq", "args": ["A", False]}],
                  "query": "count"}
        answer = run_experimental_tir(
            [{"role": "system", "content": "NERAL"},
             {"role": "user", "content": PREFIX + json.dumps(schema)}],
            lambda _: self.fail("No model fallback for explicit schema"))
        self.assertEqual(answer["text"], "0")
        self.assertTrue(answer["verified"])


class AstStructuralFingerprints(unittest.TestCase):
    @staticmethod
    def reason(source, executed):
        codes = []
        result = _exact_source_python_answer(
            "What integer does this Python program print:\n" + source,
            "python_sandbox", executed, "5", "", strict_integer=True,
            diagnostic=codes)
        return result, codes

    def test_structure_categories_do_not_claim_verification(self):
        cases = [
            ("x=2\nprint(x+3)", "y=0\nx=2\nprint(x+3)", "leading_setup_statements"),
            ("x=2\nprint(x+3)", "x=2\nprint(x+3)\nprint(x+3)",
             "extra_trailing_statements"),
            ("x=2\nprint(x+3)", "x=4\nprint(x+3)", "body_statement_modified"),
            ("x=2\nprint(x+3)", "def runner():\n    x=2\n    print(x+3)\nrunner()",
             "function_wrapped"),
        ]
        for source, executed, expected in cases:
            with self.subTest(expected=expected):
                result, reasons = self.reason(source, executed)
                self.assertIsNone(result)
                self.assertEqual(reasons, [expected])

    def test_exact_source_prefix_and_safe_fresh_alias_still_verify(self):
        result, reasons = self.reason(
            "value=2+3", "value=2+3\nanswer: int = value\nprint(answer)")
        self.assertEqual(result, "FINAL_ANSWER: 5")
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
