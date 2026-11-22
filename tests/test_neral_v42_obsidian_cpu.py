"""Obsidian opt-in contracts: CPU only, no live network or GPU."""
import json
import unittest
from unittest.mock import Mock, patch
from neral_evidence_agent import run_evidence_agent
from neral_web_search import WebSearchError
from neral_obsidian_deep import ProgressRegistry, constrained_query

QUESTION = "Ein Paket wiegt 7 kg, ein anderes 12 kg. Bestimme das Gesamtgewicht."
PLAN = {"tool": "python_sandbox", "parts": ["Beide Gewichte in kg addieren"], "search_query": ""}
BAD = {"locale": "de", "calculations": [{"label":"Gewicht", "expr":"u1 * u2", "unit":"kg"}], "missing":[]}
GOOD = {"locale": "de", "calculations": [{"label":"Gewicht", "expr":"u1 + u2", "unit":"kg"}], "missing":[]}
OK = {"ok":True,"issue":""}
DIRECT = {"tool":"direct_reasoning","parts":["Beantworten und belegen"],"search_query":""}

class DeepTests(unittest.TestCase):
    def run_deep(self, question, packets, execute, **kwargs):
        calls = []
        events = []
        def respond(system, payload, budget, schema):
            calls.append(json.loads(payload))
            return json.dumps(packets.pop(0))
        # No shortcut may hide the tool retry under test.
        with patch("neral_evidence_agent.quantity_plan", return_value=None), \
             patch("neral_v32_verifier.verified_math_question", return_value=None), \
             patch("neral_v32_verifier.verified_python_question", return_value=None), \
             patch("neral_v32_verifier.verified_shift_count", return_value=None), \
             patch("neral_knowledge_verifier.verified_element_question", return_value=None):
            result = run_evidence_agent(question, respond, execute, deep=True,
                                        progress=events.append, **kwargs)
        return result, calls, events

    def test_python_repair_second_attempt(self):
        execute = Mock(side_effect=[{"status":"FAILED","error_code":"runtime_error"},
                                    {"status":"PASSED","output":"19"}])
        result,calls,events = self.run_deep(QUESTION,[PLAN,BAD,PLAN,GOOD,OK],execute)
        self.assertEqual(result["agent_attempts"],2)
        self.assertIn("19 kg",result["text"])
        self.assertIn("runtime_error",calls[2]["correction"])
        self.assertIn("previous_proposal",calls[3])
        self.assertIn("+Decimal",execute.call_args.args[0])
        self.assertEqual([(e["attempt"],e["tool"]) for e in events if e["stage"]=="tool"],
                         [(1,"python_sandbox"),(2,"python_sandbox")])

    def test_critic_repair_third_attempt(self):
        execute = Mock(side_effect=[{"status":"FAILED","error_code":"timeout"},
                                    {"status":"PASSED","output":"84"},
                                    {"status":"PASSED","output":"19"}])
        result,calls,events = self.run_deep(QUESTION,[PLAN,BAD,PLAN,BAD,
            {"ok":False,"issue":"Addition statt Multiplikation, beide Einheiten kg"},PLAN,GOOD,OK],execute)
        self.assertEqual(result["agent_attempts"],3)
        self.assertIn("Addition",calls[-3]["correction"])
        self.assertIn("19 kg",result["text"])
        self.assertTrue(result["agent_checked"])
        self.assertFalse(result["verified"])
        self.assertEqual(execute.call_count,3)
        self.assertEqual(events[-1]["stage"],"critic")

    def test_all_attempts_fail_closed(self):
        execute=Mock(return_value={"status":"FAILED","error_code":"timeout"})
        result,_,_=self.run_deep(QUESTION,[PLAN,BAD]*3,execute)
        self.assertFalse(result["agent_checked"])
        self.assertFalse(result["tool_verified"])
        self.assertEqual(execute.call_count,3)
        self.assertEqual(result["agent_attempts"],3)

    def test_z3_repairs_after_tool_failure(self):
        plan={"tool":"z3_solver","parts":["Zuordnung und Ausschluss"],"search_query":""}
        def worker(expr):
            return {"schema":{"variables":[{"name":"ada","min":1,"max":3}],
                "constraints":[{"rule_id":"R1","description":"Ada Montag","expr":expr}],
                "objective":"find_one"},"legend":"Montag=1"}
        execute=Mock(side_effect=[{"status":"FAILED","error_code":"runtime_error"},
                                  {"status":"PASSED","output":"ada=1"}])
        with patch("neral_evidence_agent.compile_z3_json",side_effect=["bad","repaired"]):
            result,calls,_=self.run_deep("Ordne Ada dem Montag zu.",[plan,worker("ada == 2"),
                plan,worker("ada == 1"),OK],execute)
        self.assertEqual(result["agent_attempts"],2)
        self.assertIn("ada=1",result["text"])
        self.assertEqual(execute.call_args.args[0],"repaired")

    def test_web_empty_then_exception_then_repaired(self):
        search=Mock()
        search.search.side_effect=[{"results":[]},WebSearchError("unavailable"),{"results":[{"url":"https://example.com"}]}]
        pages=Mock(side_effect=[[],[{"url":"https://example.com","page_read":True,
               "page_text":"Elektrostatik untersucht ruhende elektrische Ladungen."}]])
        plans=[dict(DIRECT,search_query=q) for q in ("Elektrostatik Definition Physik",
               "Elektrostatik Definition","Elektrostatik Physik")]
        result,_,events=self.run_deep("Was ist Elektrostatik?",plans+[
            {"answer":"Elektrostatik untersucht ruhende elektrische Ladungen.","citations":["p1"]},OK],
            Mock(),web_query="Elektrostatik Definition Physik",search=search,page_reader=pages)
        self.assertEqual(result["agent_attempts"],3)
        self.assertEqual([c.args[0] for c in search.search.call_args_list],
                         ["Elektrostatik Definition Physik","Elektrostatik Definition","Elektrostatik Physik"])
        self.assertTrue(result["agent_checked"])
        self.assertEqual([e["attempt"] for e in events if e["stage"]=="web"],[1,2,3])

    def test_wiki_miss_falls_back_to_authorized_web(self):
        search=Mock()
        search.search.return_value={"results":[{"url":"https://example.com"}]}
        pages=Mock(return_value=[{"url":"https://example.com","page_read":True,
               "page_text":"Elektrostatik untersucht ruhende elektrische Ladungen."}])
        result,_,events=self.run_deep("Was ist Elektrostatik?",[DIRECT,
            {"answer":"Elektrostatik untersucht ruhende elektrische Ladungen.","citations":["p1"]},OK],
            Mock(),local_lookup=Mock(return_value=[]),fallback_query="Elektrostatik",
            search=search,page_reader=pages)
        self.assertIn("Elektrostatik",result["text"])
        search.search.assert_called_once_with("Elektrostatik")
        self.assertEqual([e["stage"] for e in events][:4],["plan","wiki","web","tool"])

    def test_wiki_hit_is_not_proof(self):
        local=Mock(return_value=[{"title":"Elektrostatik","context":"Elektrostatik untersucht ruhende elektrische Ladungen.",
                                  "url":"http://127.0.0.1:9691/content/wiki/Elektrostatik","book":"wiki"}])
        search=Mock()
        result,_,_=self.run_deep("Was ist Elektrostatik?",[DIRECT,
            {"answer":"Elektrostatik untersucht ruhende elektrische Ladungen.","citations":["p1"]},OK],
            Mock(),local_lookup=local,fallback_query="Elektrostatik",search=search)
        search.search.assert_not_called()
        self.assertTrue(result["agent_checked"])
        self.assertFalse(result["tool_verified"])
        self.assertEqual(result["knowledge_route"],"offline_zim")
        self.assertIn("ZIM",result["text"])

    def test_query_cannot_exfiltrate_new_words(self):
        self.assertEqual(constrained_query("Elektrostatik Physik","Elektrostatik Geheimnis"),"Elektrostatik Physik")
        self.assertEqual(constrained_query("Elektrostatik Physik","Physik"),"Physik")

    def test_progress_scoped_bounded_and_finished(self):
        r=ProgressRegistry()
        r.begin("session-a","a"*32)
        r.emit("session-a","a"*32,{"stage":"plan","attempt":1,"tool":None,"secret":"no"})
        self.assertIsNone(r.read("session-b","a"*32))
        self.assertNotIn("secret",r.read("session-a","a"*32)["events"][0])
        r.finish("session-a","a"*32)
        self.assertTrue(r.read("session-a","a"*32)["done"])

    def test_source_only_element_proof_keeps_model_unloaded(self):
        from neral_three_tier_runtime import ThreeTierRuntime
        from test_neral_three_tier_api import make_runtime
        runtime, low, high, _ = make_runtime()
        runtime.evidence_agent=True
        proof={"text":"Helium","verified":True,"safe_to_promote":True,
               "verification_scope":"one_exact_atomic_number_1_to_118",
               "proof_backend":"periodic_table_index"}
        with patch.dict("os.environ",{"NERAL_V42_OBSIDIAN_DEEP":"1"}), \
             patch("neral_knowledge_verifier.verified_element_question",return_value=proof):
            result=runtime.answer_with_evidence([{"role":"system","content":"test"},
                {"role":"user","content":"Welches Element hat die Ordnungszahl 2?"}],"Obsidian")
        self.assertEqual(result["tier"],"Obsidian")
        self.assertTrue(result["verified"])
        self.assertFalse(result["model_generated"])
        self.assertFalse(result["adapter_loaded"])
        self.assertFalse(high.calls)

    def test_progress_http_session_and_benchmark_isolation(self):
        import threading
        from flask import Flask
        from test_neral_three_tier_api import make_runtime
        from neral_three_tier_runtime import register_three_tier_api
        runtime, _, _, _ = make_runtime()
        runtime.evidence_agent=True
        app=Flask(__name__)
        app.secret_key="test-secret"
        register_three_tier_api(app,runtime,busy=threading.Lock())
        client=app.test_client()
        with client.session_transaction() as session:
            session["csrf"]="a"*32
        request_id="f"*32
        path="/api/neral/chat/three-tier/progress/"+request_id
        def fake_agent(*args,**kw):
            self.assertTrue(kw.get("deep"))
            kw["progress"]({"stage":"tool","attempt":2,"tool":"z3_solver"})
            state=client.get(path,headers={"X-Neral-CSRF":"a"*32}).get_json()
            self.assertEqual(state["events"][0]["attempt"],2)
            return {"text":"Test","verified":False,"safe_to_promote":False,
                    "proof_backend":None,"error_code":None}
        payload={"tier":"Obsidian","prompt":"Komplexe Aufgabe","history":[],"progress_id":request_id}
        with patch.dict("os.environ",{"NERAL_V42_OBSIDIAN_DEEP":"1"}), \
             patch("neral_evidence_agent.run_evidence_agent",side_effect=fake_agent):
            response=client.post("/api/neral/chat/three-tier",json=payload)
        self.assertEqual(response.status_code,200)
        self.assertEqual(client.get(path).status_code,403)
        self.assertTrue(client.get(path,headers={"X-Neral-CSRF":"a"*32}).get_json()["done"])
        with patch.dict("os.environ",{"NERAL_V42_OBSIDIAN_DEEP":"1"}), \
             patch("neral_evidence_agent.run_evidence_agent",return_value={"text":"Test","verified":False}) as agent:
            response=client.post("/api/neral/chat/three-tier",json={**payload,"benchmark_isolated":True})
            self.assertEqual(response.status_code,200)
            self.assertNotIn("deep",agent.call_args.kwargs)

if __name__=="__main__":
    unittest.main()

