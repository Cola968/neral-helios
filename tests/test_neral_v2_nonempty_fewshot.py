"""RED contracts: no empty variable maps, full non-leaking few-shots, and
source-derived explicit required variable names for supported problem families.

CPU-only synthetic paraphrases. Never imports or reads prospective-100.
"""
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from neral_nl_schema_format import (
    expected_variable_declarations, get_v2_ollama_format,
)
from neral_nl_schema_bridge import (
    EXTRACT_QLORA_PROMPT, EXTRACT_BASE_PROMPT, _parse_candidate,
)
from scripts.neral_nl_schema_probe import make_local_extractor
from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics
from neral_typed_constraints_v2 import PREFIX, InvalidTypedConstraints, solve_typed_constraints


def cmp(a, op, b):
    return {"compare": {"left": {"var": a}, "op": op, "right": {"const": b}}}


class NonEmptySourceBoundContractTests(unittest.TestCase):
    def test_empty_variables_rejected_by_schema_solver_and_candidate(self):
        raw = {"version": 2, "variables": {},
               "constraints": [cmp("X0", "==", 0)], "query": "count"}
        self.assertFalse(Draft202012Validator(get_v2_ollama_format()).is_valid(raw))
        with self.assertRaises(InvalidTypedConstraints):
            solve_typed_constraints(PREFIX + json.dumps(raw))
        with self.assertRaises(ValueError):
            _parse_candidate(json.dumps(raw))

    def test_diagnostic_distinguishes_missing_variables_from_general_invalid(self):
        bad = json.dumps({"version": 2, "variables": {},
                          "constraints": [cmp("X0", "==", 0)], "query": "count"})
        with patch("scripts.neral_nl_schema_diagnostics._local_transport",
                   return_value={"message": {"content": bad}, "done_reason": "stop"}):
            result = collect_live_diagnostics("logic-02", "adapter:test", "base:test")
        self.assertEqual(result["models"]["qlora"]["status"], "empty_variables")
        self.assertEqual(result["models"]["base"]["status"], "empty_variables")
        self.assertFalse(result["verified"])
        self.assertFalse(result["safe_to_promote"])

    def test_graph_template_extends_generically_to_new_size(self):
        for max_index in (2, 4, 6):
            q = ("Color vertices 0.." + str(max_index) +
                 " with red, green or blue. Vertex 0 is red, "
                 "and endpoints of edges [(0, 1)] must differ.")
            decl = expected_variable_declarations(q)
            self.assertEqual(set(decl), {f"C{i}" for i in range(max_index+1)})
            self.assertEqual(decl["C0"], {"type": "int", "domain": [0, 1, 2]})
            props = get_v2_ollama_format(q)["properties"]["variables"]
            self.assertEqual(set(props["properties"]), set(decl))
            self.assertEqual(set(props["required"]), set(decl))
            self.assertFalse(props["additionalProperties"])
            Draft202012Validator.check_schema(get_v2_ollama_format(q))
            candidate = {"version":2, "variables":{}, "constraints":[cmp("C0","==",0)],"query":"count"}
            self.assertFalse(Draft202012Validator(get_v2_ollama_format(q)).is_valid(candidate))

    def test_binary_template_from_source_length_not_case_id(self):
        for n in (3, 5, 7):
            q = (f"Count binary strings of length {n} with exactly 1 ones, "
                 "first two bits opposite and no two ones exactly two positions apart.")
            decl = expected_variable_declarations(q)
            self.assertEqual(set(decl), {f"B{i}" for i in range(n)})
            self.assertEqual(decl["B0"], {"type": "bool", "domain": [False, True]})
            self.assertEqual(set(get_v2_ollama_format(q)["properties"]["variables"]["required"]),
                             set(decl))

    def test_german_code_template_uses_digits_from_sentence(self):
        for count, word in ((3, "dreistelliger"), (4, "vierstelliger"),
                            (5, "fünfstelliger")):
            q = ("Ein " + word + " Code verwendet " +
                 ", ".join(str(i) for i in range(1, count+1)) +
                 " genau einmal. Die 2 steht vor der 3.")
            decl = expected_variable_declarations(q)
            self.assertEqual(set(decl), {f"P{i}" for i in range(1, count+1)})
            self.assertEqual(decl["P1"], {"type": "int",
                                         "domain": list(range(count))})
            self.assertEqual(set(get_v2_ollama_format(q)["properties"]["variables"]["required"]),
                             set(decl))

    def test_unsupported_phrasing_never_invents_required_names(self):
        for question in ("What is the meaning of life?", "Color anything.",
                         "Count binary strings of length 1000.",
                         "Ein fünfstelliger Code verwendet 1, 2, 3 genau einmal."):
            self.assertIsNone(expected_variable_declarations(question))
            self.assertNotIn("required", get_v2_ollama_format(question)["properties"]["variables"])

    def test_few_shots_are_complete_valid_non_reference_schemas(self):
        from neral_nl_schema_bridge import V2_FEW_SHOT_EXAMPLES
        self.assertEqual(len(V2_FEW_SHOT_EXAMPLES), 3)
        validator = Draft202012Validator(get_v2_ollama_format())
        for example in V2_FEW_SHOT_EXAMPLES:
            task, s = example["task"], example["schema"]
            self.assertGreater(len(task), 35)
            validator.validate(s)
            self.assertIsInstance(solve_typed_constraints(PREFIX+json.dumps(s)), int)
            self.assertTrue(s["variables"])
            self.assertEqual(len(s["variables"]), len(set(s["variables"])))
        for prompt in (EXTRACT_BASE_PROMPT, EXTRACT_QLORA_PROMPT):
            self.assertLess(len(prompt), 5200)
            self.assertEqual(prompt.count("BEISPIEL"), 3)
            for example in V2_FEW_SHOT_EXAMPLES:
                self.assertIn(json.dumps(example["schema"], ensure_ascii=False, separators=(",", ":")),
                              prompt)
            self.assertIn("i=0..n-3", prompt)
            self.assertIn('"variables"', prompt)
            self.assertIn('nicht leer', prompt.lower())

    def test_ollama_structured_schema_uses_source_required_names(self):
        q="Count binary strings of length 4 with exactly 2 ones, first two bits opposite."
        seen=[]
        def transport(payload):
            seen.append(payload)
            return {"message":{"content":"{}"},"done_reason":"stop"}
        model=make_local_extractor("base:qwen",transport=transport)
        model([{"role":"system","content":EXTRACT_BASE_PROMPT},
               {"role":"user","content":q}])
        spec=seen[0]["format"]["properties"]["variables"]
        self.assertEqual(spec["required"], ["B0","B1","B2","B3"])
        self.assertFalse(spec["additionalProperties"])
        self.assertEqual(seen[0]["options"]["num_predict"],1024)


if __name__ == "__main__":
    unittest.main()
