"""Fail-closed JSON-to-Z3 compiler contracts; no model, GPU or Docker required."""
import ast
import json
import unittest

from neral_z3_json_compiler import SchemaError, compile_z3_json


def sample():
    return {
        "variables": [
            {"name": "ada", "min": 0, "max": 2},
            {"name": "berta", "min": 0, "max": 2},
            {"name": "cora", "min": 0, "max": 2},
        ],
        "constraints": [
            {"rule_id": "R1", "description": "Alle Schichten eindeutig", "expr": "ada != berta"},
            {"rule_id": "R2", "description": "Alle Schichten eindeutig", "expr": "ada != cora"},
            {"rule_id": "R3", "description": "Alle Schichten eindeutig", "expr": "berta != cora"},
            {"rule_id": "R4", "description": "Ada nicht morgens", "expr": "ada != 0"},
            {"rule_id": "R5", "description": "Berta nicht nachts", "expr": "berta != 2"},
            {"rule_id": "R6", "description": "Cora nicht abends", "expr": "cora != 1"},
        ],
        "objective": "count_all",
    }


class Compiler(unittest.TestCase):
    def test_real_full_count_builder_checks_each_rule_and_blocks_all_vars(self):
        code = compile_z3_json(json.dumps(sample(), ensure_ascii=False))
        ast.parse(code)
        self.assertIn("while solver.check() == z3.sat:", code)
        self.assertIn("solver.add(z3.Or([v != m[v] for v in all_vars]))", code)
        self.assertEqual(code.count("assert z3.is_true("), 1)
        self.assertIn("for rid,rule in zip(rule_ids,rules):",code)
        self.assertIn("R6",code)
        self.assertIn("cora != 1",code)
        self.assertIn("for rid,rule in zip(rule_ids,rules):",code)
        self.assertIn("for v,lo,hi in zip(all_vars,bounds_lo,bounds_hi):",code)
        self.assertIn("solver.add(v>=lo,v<=hi)",code)
        self.assertIn("print(count)", code)
        self.assertIn("bounds_lo=[0,0,0]", code)
        self.assertIn("bounds_hi=[2,2,2]", code)

    def test_json_without_markdown_only(self):
        raw = chr(96) * 3 + "json\n" + json.dumps(sample()) + "\n" + chr(96) * 3
        self.assertIn("solver=z3.Solver()",compile_z3_json(raw))
        with self.assertRaises(SchemaError):
            compile_z3_json(chr(96)*3 + "python\n" + json.dumps(sample()) + "\n" + chr(96)*3)
        # A harmless presentation suffix is discarded, never interpreted.
        self.assertIn("solver=z3.Solver()",
                      compile_z3_json(json.dumps(sample()) + "\nFertig."))

    def test_trailing_unstructured_text_is_ignored_but_second_payload_is_rejected(self):
        raw=json.dumps(sample())
        code=compile_z3_json(raw+"\nAlle Regeln sind erfasst.")
        self.assertIn("solver=z3.Solver()",code)
        with self.assertRaises(SchemaError):
            compile_z3_json(raw+"\n"+raw)
        with self.assertRaises(SchemaError):
            compile_z3_json(raw+"\n{\"objective\":\"find_one\"}")

    def test_unknown_schema_metadata_is_dropped_only_with_complete_required_fields(self):
        s=sample()
        s["schema"]={"type":"object","example":"untrusted metadata"}
        code=compile_z3_json(json.dumps(s))
        self.assertIn("solver=z3.Solver()",code)
        del s["constraints"]
        with self.assertRaises(SchemaError):
            compile_z3_json(json.dumps(s))

    def test_unambiguous_json_in_single_response_wrapper_is_safely_unwrapped(self):
        raw=json.dumps(sample())
        self.assertIn("solver=z3.Solver()",compile_z3_json("JSON:\n"+raw))
        self.assertIn("solver=z3.Solver()",
                      compile_z3_json("JSON:\n"+raw+"\nErklärung folgt"))
        with self.assertRaises(SchemaError):
            compile_z3_json(raw+"\n"+raw)

    def test_mismatched_json_keys_have_concrete_diagnostic(self):
        s=sample()
        del s["objective"]
        s["answer"]=2
        with self.assertRaises(SchemaError) as failure:
            compile_z3_json(json.dumps(s))
        self.assertIn("objective",str(failure.exception))
        self.assertIn("answer",str(failure.exception))

    def test_reject_duplicate_keys_fields_and_variable_names(self):
        raw = json.dumps(sample()).replace('"objective": "count_all"', '"objective": "count_all", "objective": "find_one"')
        with self.assertRaises(SchemaError):
            compile_z3_json(raw)
        s = sample()
        s["variables"].append({"name": "ada", "min": 0, "max": 2})
        with self.assertRaises(SchemaError):
            compile_z3_json(json.dumps(s))
        s = sample()
        s["unexpected"] = True
        with self.assertRaises(SchemaError):
            compile_z3_json(json.dumps(s))

    def test_reject_unsafe_ast_and_non_boolean_constraint(self):
        for expr in ("__import__('os').system('whoami')", "ada.__class__",
                     "ada[0] == 1", "ada if 1 else berta",
                     "(lambda: 1)()", "ada and berta", "ada + berta",
                     "z3.And(ada == 1)", "Ada == 1"):
            with self.subTest(expr=expr):
                s = sample()
                s["constraints"][0]["expr"] = expr
                with self.assertRaises(SchemaError):
                    compile_z3_json(json.dumps(s))

    def test_sixteen_rules_and_twelve_variables_fit_fixed_sandbox(self):
        vars_=[{"name":f"x{i}","min":0,"max":2} for i in range(12)]
        rules_=[
            {"rule_id":f"R{i+1}","description":f"Regel Nummer {i+1}",
             "expr":f"x{i%12} != {i%3}"}
            for i in range(16)
        ]
        code=compile_z3_json(json.dumps({
            "variables":vars_, "constraints":rules_, "objective":"count_all"
        }))
        self.assertLessEqual(len(code.encode("utf-8")),4096)
        self.assertLessEqual(sum(1 for _ in ast.walk(ast.parse(code))),384)
        self.assertIn("for rid,rule in zip(rule_ids,rules):",code)
        self.assertIn("solver.add(z3.Or([v != m[v] for v in all_vars]))",code)

    def test_allow_only_approved_functions_and_arithmetic(self):
        s = sample()
        s["constraints"][0]["expr"] = "Implies(ada == 0, And(berta != 2, Or(cora == 0, Not(cora == 1))))"
        self.assertIn("z3.Implies(", compile_z3_json(json.dumps(s)))
        s["constraints"][0]["expr"] = "ada + berta * 2 - cora >= -1"
        self.assertIn("ada + berta * 2 - cora >= -1", compile_z3_json(json.dumps(s)))

    def test_boolean_and_chained_comparison_syntax_normalizes_to_allowed_z3_ast(self):
        s=sample()
        s["constraints"][0]["expr"] = "ada != berta and berta != cora"
        code=compile_z3_json(json.dumps(s))
        self.assertIn("z3.And(ada != berta, berta != cora)",code)
        s["constraints"][0]["expr"] = "0 <= ada <= 2 and not (ada == berta)"
        code=compile_z3_json(json.dumps(s))
        self.assertIn("z3.And(z3.And(0 <= ada, ada <= 2), z3.Not(ada == berta))",code)
        s["constraints"][0]["expr"] = "ada != berta or cora == 1"
        self.assertIn("z3.Or(ada != berta, cora == 1)",compile_z3_json(json.dumps(s)))

    def test_find_one_builder_and_bounds_are_fail_closed(self):
        s = sample()
        s["objective"] = "find_one"
        code = compile_z3_json(json.dumps(s))
        self.assertIn("if solver.check() == z3.sat:", code)
        self.assertNotIn("while solver.check() == z3.sat:", code)
        s["variables"][0]["min"] = True
        with self.assertRaises(SchemaError):
            compile_z3_json(json.dumps(s))
        s = sample()
        s["constraints"][0]["rule_id"] = "R2"
        with self.assertRaises(SchemaError):
            compile_z3_json(json.dumps(s))

    def test_no_precomputed_model_count_in_generated_program(self):
        code = compile_z3_json(json.dumps(sample()))
        self.assertNotIn("print(2)", code)
        self.assertNotIn("eval(", code.replace("m.eval(", ""))
        self.assertLessEqual(len(code.encode("utf-8")), 4096)
        self.assertLessEqual(sum(1 for _ in ast.walk(ast.parse(code))), 384)


if __name__ == "__main__":
    unittest.main()
