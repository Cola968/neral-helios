"""RED contracts for constrained v2 sampling and advisory NL source coverage.

All fake model calls; no GPU, Ollama daemon, training data, or new100 access.
"""
import copy
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from neral_nl_schema_bridge import (
    EXTRACT_QLORA_PROMPT, EXTRACT_BASE_PROMPT, coverage_gate,
)
from neral_nl_schema_format import get_v2_ollama_format
from scripts.neral_nl_schema_probe import make_local_extractor
from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics
from neral_typed_constraints_v2 import PREFIX, solve_typed_constraints


def d(values, kind="int"):
    return {"type": kind, "domain": list(values)}

def v(name):
    return {"var": name}

def c(value):
    return {"const": value}

def cmp(left, op, right):
    return {"compare": {"left": left, "op": op, "right": right}}


def schema(clauses=None):
    return {"version": 2, "variables": {"X": d(range(4))},
            "constraints": clauses or [cmp(v("X"), "<=", c(2))],
            "query": "count"}


class StructuredOutputContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = get_v2_ollama_format()
        Draft202012Validator.check_schema(cls.output)
        cls.validator = Draft202012Validator(cls.output)

    def test_format_is_schema_object_not_json_mode(self):
        output = self.output
        self.assertEqual(output["type"], "object")
        self.assertEqual(set(output["required"]),
                         {"version", "variables", "constraints", "query"})
        self.assertFalse(output["additionalProperties"])
        self.assertEqual(output["properties"]["version"].get("const"), 2)
        self.assertEqual(output["properties"]["query"].get("enum"),
                         ["count", "satisfiable"])
        self.assertIn("$defs", output)

    def test_valid_v2_grammar_covers_all_five_families(self):
        graph = {"version": 2,
            "variables": {"C0": d(range(3)), "C1": d(range(3))},
            "constraints": [cmp(v("C0"), "==", c(0)),
                            cmp(v("C0"), "!=", v("C1"))], "query": "count"}
        binary = {"version": 2,
            "variables": {"B0": d([False, True], "bool"),
                          "B1": d([False, True], "bool")},
            "constraints": [{"exactly": {"vars": ["B0", "B1"],
                "count": 1, "value": True}}], "query": "count"}
        pin = {"version": 2,
            "variables": {"P1": d(range(5)), "P2": d(range(5))},
            "constraints": [{"all_different": ["P1", "P2"]},
                cmp({"sum": [v("P1"), v("P2")]}, "==", c(3)),
                cmp({"mod": {"term": v("P2"), "by": 2}}, "==", c(0))],
                "query": "count"}
        order = {"version": 2, "variables": {"A": d([1, 2]),
            "B": d([1, 2])}, "constraints": [{"all_different": ["A", "B"]},
            cmp(v("A"), "<", v("B"))], "query": "count"}
        logic = schema([{"iff": [cmp(v("X"), "<", c(2)),
                               {"not": cmp(v("X"), "==", c(3))}]},
                        {"implies": [cmp(v("X"), ">", c(0)),
                                    cmp(v("X"), ">=", c(1))]}])
        for case in (graph, binary, pin, order, logic):
            with self.subTest(case=case):
                self.validator.validate(case)
                self.assertIsInstance(solve_typed_constraints(
                    PREFIX + json.dumps(case)), int)

    def test_structured_schema_rejects_observed_failed_variants(self):
        wrong_version = {**schema(), "version": "NERAL constraints v2"}
        variables_list = {**schema(), "variables": [
            {"name": "X", "type": "int", "domain": [0, 1]}]}
        query_obj = {**schema(), "query": {"type": "count"}}
        free_logic = {**schema(), "constraints": [{"logic": "X is even"}]}
        color_domain = {**schema(), "variables": {
            "X": d(["red", "green", "blue"])}}
        for case in (wrong_version, variables_list, query_obj,
                     free_logic, color_domain):
            with self.subTest(case=case):
                self.assertFalse(self.validator.is_valid(case))
        self.assertFalse(self.validator.is_valid({
            **schema(), "logic": "non-AST source"}))

    def test_duplicate_json_members_rejected_even_if_sampling_is_constrained(self):
        from neral_nl_schema_bridge import _parse_candidate
        bad = '{"version":2,"version":2,"variables":{"X":{"type":"int","domain":[0]}},"constraints":[{"compare":{"left":{"var":"X"},"op":"==","right":{"const":0}}}],"query":"count"}'
        with self.assertRaises(ValueError):
            _parse_candidate(bad)

    def test_format_immutable_across_calls(self):
        output = get_v2_ollama_format()
        output["properties"]["version"]["const"] = 37
        self.assertEqual(get_v2_ollama_format()["properties"]["version"]["const"], 2)

    def test_local_ollama_payload_uses_schema_and_1024_repeat_penalty(self):
        seen = []
        def send(payload):
            seen.append(payload)
            return {"message": {"content": json.dumps(schema())},
                    "done_reason": "stop"}
        ext = make_local_extractor("qwen3.5:9b", transport=send)
        result = ext([{"role": "system", "content": EXTRACT_BASE_PROMPT},
                      {"role": "user", "content": "Count X in 0..3 with X<=2"}])
        self.assertEqual(json.loads(result), schema())
        p = seen[0]
        self.assertIsInstance(p["format"], dict)
        self.assertEqual(p["format"]["properties"]["version"]["const"], 2)
        self.assertEqual(p["options"]["num_predict"], 1024)
        self.assertEqual(p["options"]["repeat_penalty"], 1.05)
        self.assertEqual(p["options"]["temperature"], 0)
        self.assertFalse(p["think"])

    def test_both_prompts_are_concise_and_include_real_ast_examples(self):
        for prompt in (EXTRACT_QLORA_PROMPT, EXTRACT_BASE_PROMPT):
            self.assertLess(len(prompt), 1800)
            for token in ('"version":2', '"variables"', '"query":"count"',
                          '"compare"', '"all_different"', '"exactly"',
                          '"sum"', '"mod"'):
                self.assertIn(token, prompt)
            self.assertIn("nicht", prompt.lower())
            self.assertIn("integer", prompt.lower())

    def test_full_bridge_unverified_even_with_structured_sampling(self):
        payload = json.dumps(schema())
        def fake_transport(p):
            return {"message": {"content": payload}, "done_reason": "stop"}
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   side_effect=fake_transport):
            result = collect_live_diagnostics("21453", "adapter:test", "base:test")
        self.assertFalse(result["verified"])
        self.assertFalse(result["safe_to_promote"])


class MoreStructuralCoverageTests(unittest.TestCase):
    def test_graph_edges_each_need_a_distinct_inequality_witness(self):
        question = ("Color vertices 0..2 with red green blue. Vertex 0 is red, "
                    "and endpoints of edges [(0, 1), (0, 2), (1, 2)] must differ.")
        variables = {f"C{i}": d(range(3)) for i in range(3)}
        good = {"version": 2, "variables": variables,
            "constraints": [cmp(v("C0"), "==", c(0))] + [
                cmp(v(f"C{a}"), "!=", v(f"C{b}"))
                for a, b in [(0, 1), (0, 2), (1, 2)]], "query": "count"}
        self.assertTrue(coverage_gate(question, good)["passed"])
        bad = copy.deepcopy(good)
        bad["constraints"].pop()
        self.assertFalse(coverage_gate(question, bad)["passed"])

    def test_binary_exact_count_and_forbidden_distance_two_need_witnesses(self):
        question = ("Count binary strings of length 5 with exactly 2 ones, "
                    "first two bits opposite and no two ones exactly two positions apart.")
        variables = {f"B{i}": d([False, True], "bool") for i in range(5)}
        rules = [{"exactly": {"vars": list(variables), "value": True, "count": 2}},
                 cmp(v("B0"), "!=", v("B1"))] + [
            {"not": {"and": [cmp(v(f"B{i}"), "==", c(True)),
                              cmp(v(f"B{i+2}"), "==", c(True))]}}
            for i in range(3)]
        good = {"version": 2, "variables": variables,
                "constraints": rules, "query": "count"}
        self.assertTrue(coverage_gate(question, good)["passed"])
        no_cardinality = {**good, "constraints": rules[1:]}
        self.assertFalse(coverage_gate(question, no_cardinality)["passed"])
        no_distance = {**good, "constraints": rules[:2]}
        self.assertFalse(coverage_gate(question, no_distance)["passed"])

    def test_core_coverage_never_promotes_schemas_as_verified(self):
        from neral_nl_schema_bridge import assess_dual_extraction
        case = schema()
        result = assess_dual_extraction(
            "Count X in 0..3 with X<=2",
            lambda _: json.dumps(case), lambda _: json.dumps(case))
        self.assertFalse(result["verified"])
        self.assertFalse(result["safe_to_promote"])


if __name__ == "__main__":
    unittest.main()
