"""V4.3 CPU-only integration, including real restricted Python tests."""
import json
import sys
import time
import unittest
from unittest.mock import Mock, patch
from neral_v43_bounded import bounded_answer, Verdict
from neral_v43_routes import verify_answer, source_plan
from scripts.neral_v43_windows_worker import evaluate_function
from scripts.neral_v43_obsidian_codex import run_test_repair_loop


class RouteCPU(unittest.TestCase):
    def run_route(self,question,answers,**kw):
        gen=Mock(return_value={"text":answers[0]})
        repair=Mock(return_value={"text":answers[-1]})
        result=bounded_answer(gen,lambda draft:verify_answer(
            question,draft,deadline=time.monotonic()+5,**kw),
            repair=repair,remaining=lambda:30)
        return result,repair

    def test_everyday_has_one_round(self):
        r,rep=self.run_route("Warum ist der Himmel blau?",["Lichtstreuung."])
        self.assertEqual(r["text"],"Lichtstreuung.")
        self.assertEqual(r["generation_rounds"],1)
        rep.assert_not_called()

    def test_knowledge_timeout_preserves_draft(self):
        web=Mock(side_effect=TimeoutError())
        r,rep=self.run_route("In welchem Jahr landete Apollo 11?",["1969"],
                             web_query="Apollo 11 Mondlandung",search=web)
        self.assertEqual(r["text"],"1969")
        self.assertEqual(r["verification_status"],"verification_incomplete")
        rep.assert_not_called()

    def test_math_repair_independent_arithmetic(self):
        r,rep=self.run_route("Berechne 6*7.",["41","42"])
        self.assertEqual(r["text"],"42")
        self.assertTrue(r["verified"])
        self.assertEqual(r["generation_rounds"],2)
        rep.assert_called_once()

    def test_math_timeout_keeps_draft(self):
        r=verify_answer("Berechne 6*7.",{"text":"42"},deadline=0)
        self.assertEqual(r.status,"timeout")

    def test_math_logic_z3_and_boundary(self):
        spec={"variables":[{"name":"x","kind":"int","lower":0,"upper":3}],
              "premises":[],"claim":{"op":"ge","args":[{"var":"x"},{"const":0}]}}
        q="Prüfe die Aussage.\nNERAL_MATH_SPEC: "+json.dumps(spec)
        r,rep=self.run_route(q,["Nein","Ja"])
        self.assertTrue(r["verified"])
        self.assertEqual(r["text"],"Ja")

    def test_coding_repair_via_windows_runner(self):
        q="Schreibe f.\nNERAL_TEST_SPEC: "+json.dumps({"tests":[{"function":"f","args":[3],"expected":4}],"properties":[{"function":"f","kind":"monotonic_int","min":0,"max":5}]})
        r,rep=self.run_route(q,["def f(x): return x-1","def f(x): return x+1"],backend="restricted")
        self.assertEqual(r["text"],"def f(x): return x+1")
        self.assertTrue(r["verified"])
        rep.assert_called_once()

    def test_coding_timeout_preserves_draft(self):
        q='Schreibe f.\nNERAL_TEST_SPEC: {"tests":[{"function":"f","args":[3],"expected":4}],"properties":[]}'
        with patch("neral_v43_routes.run_test_repair_loop",return_value={"status":"timeout"}):
            r,rep=self.run_route(q,["def f(x): return x+1"])
        self.assertEqual(r["text"],"def f(x): return x+1")
        rep.assert_not_called()

    def test_syntax_repair_is_bounded(self):
        q='Schreibe f.\nNERAL_TEST_SPEC: {"tests":[{"function":"f","args":[3],"expected":4}],"properties":[]}'
        r,rep=self.run_route(q,["def f(x) return x+1","def f(x): return x+1"],backend="restricted")
        self.assertTrue(r["verified"])
        rep.assert_called_once()

    def test_repair_requires_twelve_seconds(self):
        rep=Mock()
        r=bounded_answer(lambda:{"text":"41"},lambda _:Verdict("counterexample","42"),
                         repair=rep,remaining=lambda:11)
        rep.assert_not_called()
        self.assertEqual(r["text"],"41")

    def test_restricted_worker_never_executes_imports_or_attributes(self):
        for code in ("import os\ndef f(x): return x","def f(x): return x.__class__",
                     "def f(x): return open('file')","def f(x): return 2**100000"):
            with self.assertRaises(ValueError):
                evaluate_function(code,"f",[1])

    def test_real_restricted_worker_pass_and_fail(self):
        tests=[{"function":"f","args":[3],"expected":4}]
        for code,expected in [("def f(x): return x+1","passed"),
                              ("def f(x): return x-1","tests_failed")]:
            r=run_test_repair_loop(code,tests=tests,properties=[],backend="restricted",max_repairs=0)
            self.assertEqual(r["status"],expected)

    def test_no_invented_test_or_math_spec(self):
        self.assertIsNone(source_plan("Write a general Python app."))
        self.assertIsNone(source_plan("Tell me about Berlin."))


if __name__=="__main__":
    unittest.main()
