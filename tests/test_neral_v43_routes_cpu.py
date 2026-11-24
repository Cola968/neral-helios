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

    def test_restricted_worker_interprets_bounded_for_and_selected_imports(self):
        examples=(
            ("def f(n):\n total=0\n for x in range(n):\n  total+=x\n return total", [5], 10),
            ("def f(xs):\n out=[]\n for x in xs:\n  out=out+[max(x,0)]\n return out", [[-2,3]], [0,3]),
            ("from statistics import median as mid\ndef f(xs):\n return mid(sorted(set(list(xs))))", [[1,7,3,5,7]], 4),
            ("def f(n):\n total=0\n for x in range(n):\n  if x==3: break\n  total+=x\n return total", [6], 3),
            ("def f(x):\n import math as m\n return m.ceil(x/4)", [5], 2),
        )
        for code,args,expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(evaluate_function(code,"f",args),expected)

    def test_restricted_worker_denies_unbounded_and_unapproved_constructs(self):
        examples=(
            "def f(x):\n for i in range(257):\n  x+=i\n return x",
            "def f(x):\n for i in range(256):\n  for j in range(256):\n   x+=j\n return x",
            "from math import *\ndef f(x): return x",
            "import math\ndef f(x): return math.__dict__",
            "def f(x):\n if False:\n  import os\n return x",
            "def f(x):\n if False: break\n return x",
        )
        for code in examples:
            with self.subTest(code=code), self.assertRaises(ValueError):
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


class HttpRouteCPU(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from threading import Lock
        from neral_three_tier_runtime import ThreeTierRuntime, LUNA, RAW_9B, register_three_tier_api
        self.model=Mock(model=RAW_9B,loaded=True,adapter_sha256=None)
        self.runtime=ThreeTierRuntime(Mock(model=LUNA,adapter_sha256=None),self.model,
            execute_python=Mock(),v43_profile=True,evidence_agent=False)
        self.runtime._verified_source=Mock(return_value=None)
        self.search=Mock()
        self.app=Flask(__name__)
        self.app.secret_key=b"a"*32
        self.busy=Lock()
        register_three_tier_api(self.app,self.runtime,busy=self.busy,search_service=self.search)
        self.client=self.app.test_client()

    def ask(self,prompt,answers,**opts):
        self.model.respond_chat.side_effect=answers
        response=self.client.post("/api/neral/chat/three-tier",json={
            "tier":"Obsidian","prompt":prompt,"benchmark_isolated":True,**opts})
        self.assertEqual(response.status_code,200,response.get_json())
        self.assertFalse(self.busy.locked())
        result=response.get_json()
        self.assertFalse(result["adapter_loaded"])
        self.assertIsNone(result["error_code"])
        self.assertTrue(result["v43_bounded"])
        return result

    def test_http_everyday_raw_profile(self):
        r=self.ask("Warum ist der Himmel blau?",["Lichtstreuung."])
        self.assertEqual(r["text"],"Lichtstreuung.")
        self.assertEqual(r["generation_rounds"],1)

    def test_http_web_timeout_without_legacy_agent(self):
        self.search.search.side_effect=TimeoutError()
        r=self.ask("In welchem Jahr landete Apollo 11?",["1969"],web_query="Apollo 11 Mondlandung")
        self.assertEqual(r["text"],"1969")
        self.assertEqual(r["verification_status"],"verification_incomplete")
        self.assertEqual(r["generation_rounds"],1)

    def test_http_math_repair(self):
        r=self.ask("Berechne 6*7.",["41","42"])
        self.assertEqual(r["text"],"42")
        self.assertTrue(r["verified"])
        self.assertEqual(r["generation_rounds"],2)

    def test_http_math_timeout(self):
        with patch("neral_v43_routes.exact_algebra",side_effect=TimeoutError()):
            r=self.ask("Berechne 6*7.",["42"])
        self.assertEqual(r["text"],"42")
        self.assertEqual(r["verification_status"],"verification_incomplete")

    def test_http_codex_repair(self):
        from neral_v43_routes import run_test_repair_loop as real_run
        q='Schreibe f.\nNERAL_TEST_SPEC: {"tests":[{"function":"f","args":[3],"expected":4}],"properties":[]}'
        def restricted(*a,**kw):
            kw["backend"]="restricted"
            return real_run(*a,**kw)
        with patch("neral_v43_routes.run_test_repair_loop",side_effect=restricted):
            r=self.ask(q,["def f(x): return x-1","def f(x): return x+1"],force_sandbox=True)
        self.assertEqual(r["text"],"def f(x): return x+1")
        self.assertEqual(r["generation_rounds"],2)

    def test_http_codex_timeout(self):
        q='Schreibe f.\nNERAL_TEST_SPEC: {"tests":[{"function":"f","args":[3],"expected":4}],"properties":[]}'
        with patch("neral_v43_routes.run_test_repair_loop",return_value={"status":"timeout"}):
            r=self.ask(q,["def f(x): return x+1"],force_sandbox=True)
        self.assertEqual(r["text"],"def f(x): return x+1")
        self.assertEqual(r["generation_rounds"],1)
        self.assertTrue(r["verification_warning"])


if __name__=="__main__":
    unittest.main()
