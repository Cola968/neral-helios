"""RED CPU contracts: opt-in bounded formal solving and private first-turn telemetry."""
from __future__ import annotations
import json
import unittest

from neral_formal_constraints import InvalidFormalConstraints, try_solve_formal_constraints
from neral_tir_experimental import run_experimental_tir

PREFIX = "NERAL_CONSTRAINTS_JSON: "

def formal(**kw):
    return PREFIX + json.dumps(kw, separators=(",", ":"))

def messages(question):
    return [{"role": "system", "content": "Du bist NERAL."},
            {"role": "user", "content": question}]


class FormalConstraintContracts(unittest.TestCase):
    def test_bounded_integer_order_count(self):
        q = formal(variables={"A":[1,2,3], "B":[1,2,3]},
                   rules=[{"left":"A", "op":"<", "right":"B"}], query="count")
        self.assertEqual(try_solve_formal_constraints(q), 3)

    def test_boolean_exclusion_and_satisfiability(self):
        base = {"variables":{"A":[False,True],"B":[False,True]},
                "rules":[{"left":"A","op":"!=","right":"B"}]}
        self.assertEqual(try_solve_formal_constraints(formal(**base,query="count")), 2)
        self.assertEqual(try_solve_formal_constraints(formal(**base,query="satisfiable")), 1)
        base["rules"].append({"left":"A","op":"==","right":"B"})
        self.assertEqual(try_solve_formal_constraints(formal(**base,query="satisfiable")), 0)

    def test_fixed_value_and_all_different(self):
        q = formal(variables={"A":[1,2,3],"B":[1,2,3],"C":[1,2,3]},
                   rules=[{"left":"A","op":"==","value":2}],
                   all_different=[["A","B","C"]],query="count")
        self.assertEqual(try_solve_formal_constraints(q), 2)

    def test_nonformal_requests_are_never_classified_by_words(self):
        self.assertIsNone(try_solve_formal_constraints(
            "There are three people A, B, C. Count valid permutations."))
        self.assertIsNone(try_solve_formal_constraints("Explain a galaxy."))

    def test_rejects_unsafe_schema_and_excessive_search(self):
        base = {"variables":{"A":[1,2,3],"B":[1,2,3]},
                "rules":[{"left":"A","op":"<","right":"B"}],"query":"count"}
        variants=[
            {**base,"extra":"ignore me"},
            {**base,"rules":[{"left":"A","op":"__import__","right":"B"}]},
            {**base,"rules":[{"left":"A","op":"<","right":"DOES_NOT_EXIST"}]},
            {**base,"rules":[{"left":"A","op":"<","right":"B","value":0}]},
            {**base,"variables":{"A":["__import__('os').system('echo X')"]}},
            {**base,"variables":{"A":list(range(101)), "B":list(range(101)),
                                 "C":list(range(101))}},
            {**base,"variables":{"A":[1,1],"B":[1,2]}},
        ]
        for variant in variants:
            with self.subTest(variant=str(variant)[:90]):
                with self.assertRaises(InvalidFormalConstraints):
                    try_solve_formal_constraints(formal(**variant))
        with self.assertRaises(InvalidFormalConstraints):
            try_solve_formal_constraints(PREFIX + '{"variables":')

    def test_formal_route_does_not_invoke_model_or_generated_python(self):
        q = formal(variables={"A":[1,2,3],"B":[1,2,3]},
                   rules=[{"left":"A","op":"<","right":"B"}],query="count")
        output = run_experimental_tir(
            messages(q+"\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."),
            lambda _: self.fail("formal solver must bypass model"),
            tool_runner=lambda *_: self.fail("must never execute generated code"))
        self.assertEqual(output["text"],"FINAL_ANSWER: 3")
        self.assertEqual(output["calls_used"],0)
        self.assertTrue(output["verified"])
        self.assertFalse(output["model_generated"])
        self.assertEqual(output["tool_used"],"neral_logic_exact")

    def test_invalid_formal_schema_fails_closed(self):
        out = run_experimental_tir(
            messages(PREFIX+"{bad JSON"),
            lambda _: self.fail("malformed structured request must not prompt-inject"))
        self.assertEqual(out["verification_status"],"formal_schema_rejected")
        self.assertFalse(out["verified"])
        self.assertEqual(out["calls_used"],0)


class FirstTurnTelemetryContracts(unittest.TestCase):
    def test_only_scalar_first_turn_evidence_is_emitted(self):
        seen=[]
        replies=iter([
            'NERAL_EXECUTE_JSON: {"tool":"solve","code":"6*7"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 42",
        ])
        def respond(turns):
            seen.append([dict(x) for x in turns])
            return next(replies)
        result = run_experimental_tir(
            messages("Compute 6*7."),respond,
            tool_runner=lambda *_:{"status":"PASSED","output":"42"},
            first_turn_diagnostic=lambda:{
                "eval_count":17, "prompt_eval_count":65,"done_reason":"stop",
                "content":"SECRET_MUST_NOT_LEAK","thinking":"SECRET_MUST_NOT_LEAK"})
        self.assertEqual(result["first_turn_telemetry"],
                         {"eval_count":17,"prompt_eval_count":65,
                          "done_reason":"stop","decision":"tool"})
        self.assertNotIn("SECRET_MUST_NOT_LEAK",str(result))

    def test_free_text_fallback_classified_as_invalid(self):
        answers=iter(["lengthy reasoning without any packet", "Fallback answer"])
        result=run_experimental_tir(messages("Count all arrangements"),
                                    lambda _:next(answers),
                                    first_turn_diagnostic=lambda:{
                                        "eval_count":2048,"done_reason":"stop"})
        self.assertEqual(result["first_turn_telemetry"]["decision"],"unstructured")
        self.assertEqual(result["first_turn_telemetry"]["eval_count"],2048)
        self.assertEqual(result["verification_status"],"base_fallback_unverified")


class BenchmarkTelemetrySafety(unittest.TestCase):
    def test_scalar_filter_never_leaks_arbitrary_payload(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        event = {"eval_count":2048,"prompt_eval_count":110,
                 "done_reason":"stop","decision":"unstructured",
                 "prompt":"TOP_SECRET","content":"TOP_SECRET"}
        filtered=safe_first_turn_event(event)
        self.assertEqual(filtered, {"eval_count":2048,"prompt_eval_count":110,
                         "done_reason":"stop","decision":"unstructured"})
        self.assertNotIn("TOP_SECRET",str(filtered))
        self.assertIsNone(safe_first_turn_event(
            {"eval_count":True,"prompt_eval_count":0,"done_reason":"stop",
             "decision":"tool"}))
        self.assertIsNone(safe_first_turn_event(
            {"eval_count":100,"prompt_eval_count":0,"done_reason":"unexpected",
             "decision":"tool"}))

if __name__=="__main__":
    unittest.main()
