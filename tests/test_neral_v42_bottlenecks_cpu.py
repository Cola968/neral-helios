"""V4.2 telemetry bottleneck regressions; exclusively CPU and fake inference."""
import threading
import unittest
from unittest.mock import patch, Mock
from flask import Flask

from neral_obsidian_deep import is_simple_everyday_question
from neral_v42_quality import final_marked_value, exact_executed_numeric_tail
from neral_v42_telemetry import begin_request, end_request, timed_search_service
from neral_three_tier_runtime import register_three_tier_api
from test_neral_three_tier_api import make_runtime
from scripts.neral_v41_blind200_live import _exact_text


class ObsidianFastPathCPU(unittest.TestCase):
    def test_only_plain_explanations_bypass_three_loop_agent(self):
        self.assertTrue(is_simple_everyday_question(
            "Warum beschlägt ein Badezimmerspiegel nach dem Duschen? Erkläre es in zwei verständlichen Sätzen."))
        for prompt in (
            "Berechne zuerst 17 mal 23 und ziehe 46 ab.",
            "Schreibe eine Python-Funktion für Primzahlen.",
            "Plane vier Schichten mit Z3 und Bedingungen.",
            "Vergleiche mehrere aktuelle Quellen zu Apollo 11.",
            "In welchem Jahr landete Apollo 11? Belege mit einer gelesenen Quelle.",
            "Was ist heute der aktuelle Stand der BER-Flüge?",
        ):
            with self.subTest(prompt=prompt):
                self.assertFalse(is_simple_everyday_question(prompt))

    def test_obsidian_real_http_plain_explanation_max_two_generations(self):
        runtime, low, high, legacy = make_runtime()
        runtime.evidence_agent = True
        app = Flask("v42-fastpath")
        register_three_tier_api(app, runtime, busy=threading.Lock(), search_service=Mock())
        with patch.object(runtime, "answer_with_evidence",
                          side_effect=AssertionError("Slow evidence pipeline forbidden")), \
             patch("neral_verified_notes.exact_note_lookup", return_value=None):
            response = app.test_client().post("/api/neral/chat/three-tier", json={
                "tier":"Obsidian",
                "prompt":"Warum beschlägt ein Spiegel nach einer warmen Dusche? Erkläre es kurz.",
                "history":[],
            })
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(response.json.get("obsidian_fast_path"))
        self.assertLessEqual(len(high.calls), 2)
        self.assertFalse(response.json["verified"])


class ExactExecutedFinalCPU(unittest.TestCase):
    def test_markdown_and_punctuation_are_unambiguous(self):
        for text in ("Rechenweg\\nENDERGEBNIS: **345**",
                     "Rechenweg\\n  ENDERGEBNIS: 345.  ",
                     "Rechenweg\\nENDERGEBNIS: `345`",
                     "Rechenweg\\nENDERGEBNIS: __345__"):
            with self.subTest(text=text):
                self.assertEqual(final_marked_value(text), "345")
                self.assertEqual(_exact_text(text, "integer"), "345")
        for bad in ("ENDERGEBNIS: 345 oder 346",
                    "ENDERGEBNIS: 345\\nENDERGEBNIS: 346",
                    "ENDERGEBNIS: 345\\nweitere Erklärungen",
                    "ENDERGEBNIS: **345** und 346"):
            self.assertIsNone(_exact_text(bad, "integer"))

    def test_canonical_tail_only_from_successfully_executed_single_integer(self):
        self.assertEqual(exact_executed_numeric_tail(
            "345", succeeded=True, prompt="Berechne den Rest. ENDERGEBNIS: <Wert>"), "ENDERGEBNIS: 345")
        for stdout, good, prompt in (
            ("345", False, "Berechne 345"),
            ("345\\n346", True, "Berechne den Rest"),
            ("345 oder 346", True, "Berechne den Rest"),
            ("345", True, "Schreibe Python-Code für einen Zähler."),
            ("Keine Ausgabe", True, "Berechne den Rest"),
        ):
            self.assertIsNone(exact_executed_numeric_tail(stdout, succeeded=good, prompt=prompt))

    def test_sandbox_stdout_becomes_real_http_numeric_terminal_line(self):
        runtime, low, high, old = make_runtime()
        generated=[]
        def propose(turns, **kwargs):
            high.loaded=True
            generated.append(turns)
            return "```python\\nprint(17 * 23 - 46)\\n```"
        high.respond_chat=propose
        tool=Mock(return_value={"status":"PASSED","output":"345"})
        runtime.execute_python=tool
        answer=runtime.answer([
            {"role":"system","content":"Löse nur die aktuelle Frage."},
            {"role":"user","content":"Berechne die Restmenge aus 17 Kisten mit 23 Stück abzüglich 46. ENDERGEBNIS: <Wert>"},
        ], "Quartz")
        self.assertTrue(answer["sandbox_executed"])
        self.assertEqual(_exact_text(answer["text"], "integer"), "345")
        self.assertIn("nicht unabhängig geprüft",answer["text"])
        self.assertEqual(len(generated),1,"Do not generate a conflicting second model completion")
        tool.assert_called_once()



class MandatoryResearchCPU(unittest.TestCase):
    def test_explicit_source_question_attempts_kiwix_then_authorized_web(self):
        from neral_local_wiki import source_required
        question="In welchem Jahr landete Apollo 11? Belege mit einem wörtlichen Zitat aus einer tatsächlich gelesenen Quelle."
        self.assertTrue(source_required(question))
        self.assertFalse(source_required("Warum beschlägt ein Spiegel?"))
        self.assertFalse(source_required("Mein Passwort lautet geheim; suche eine Quelle."))

        runtime, low, high, old = make_runtime()
        runtime.evidence_agent = True
        app=Flask("source-routing")
        service=Mock()
        register_three_tier_api(app,runtime,busy=threading.Lock(),search_service=service)
        verified_response={"tier":"Quartz","mode":"QUARTZ","text":"Keine Quelle vorhanden.",
                           "verified":False,"safe_to_promote":False,
                           "agent_checked":False,"tool_verified":False,
                           "error_code":"source_evidence_missing"}
        def simulated_agent(question, respond, execute, **kwargs):
            self.assertIsNone(kwargs["web_query"])
            self.assertEqual(kwargs["fallback_query"],
                             "Apollo 11 erste bemannte Mondlandung Jahr NASA")
            self.assertEqual(kwargs["local_lookup"](question), [])
            return verified_response
        with patch("neral_local_wiki.local_wiki_search",return_value=[]) as wiki, \
             patch("neral_evidence_agent.run_evidence_agent",
                   side_effect=simulated_agent) as agent:
            response=app.test_client().post("/api/neral/chat/three-tier",json={
                "tier":"Quartz","prompt":question,"history":[],
                "web_query":"Apollo 11 erste bemannte Mondlandung Jahr NASA",
            })
        self.assertEqual(response.status_code,200,response.get_json())
        wiki.assert_called_once()
        self.assertEqual(agent.call_count,1)

    def test_real_read_web_page_can_supply_literal_source_without_false_veto(self):
        import json
        from neral_evidence_agent import run_evidence_agent
        from neral_v42_telemetry import timed_search_service
        question="In welchem Jahr landete Apollo 11? Belege mit einem wörtlichen Zitat aus einer gelesenen Quelle."
        quote="Die Mission Apollo 11 landete 1969 auf dem Mond und brachte Menschen auf die Mondoberfläche."
        class Source:
            calls=0
            def search(self, query):
                self.calls+=1
                return {"results":[{"url":"https://example.org/apollo11"}]}
        service=Source()
        prompts=[
            {"tool":"direct_reasoning","parts":["Beleg lesen"],"search_query":""},
            {"answer":quote,"citations":["p1"]},
            {"ok":True,"issue":""},
        ]
        def answer(system, data, budget, schema):
            return json.dumps(prompts.pop(0))
        def pages(hits, query):
            return [{"url":"https://example.org/apollo11","page_read":True,
                     "page_text":quote}]
        result=run_evidence_agent(question,answer,Mock(),web_query=None,
                                  local_lookup=lambda q:[],
                                  fallback_query="Apollo 11 NASA Mondlandung Jahr",
                                  search=timed_search_service(service),
                                  page_reader=pages,max_attempts=1)
        self.assertEqual(service.calls,1)
        self.assertTrue(result["agent_checked"],result)
        self.assertIn("[web1]",result["text"])
        self.assertIsNone(result["error_code"])

    def test_search_service_method_instrumented_not_only_callable_object(self):
        class Source:
            def search(self, query):
                return {"results":[]}
        telemetry, token=begin_request()
        try:
            routed=timed_search_service(Source())
            self.assertEqual(routed.search("Apollo 11"),{"results":[]})
            self.assertIn("web_v4",telemetry.summary()["tool_breakdown_ms"])
        finally:
            end_request(token)


class StartupWarmupCPU(unittest.TestCase):
    def test_warmup_happens_before_server_listens_and_is_idempotent(self):
        from neral_preview_server import warmup_three_tier_before_serve
        runtime, low, high, old=make_runtime()
        load_calls=[]
        def load():
            high.loaded=True
            load_calls.append(1)
        high._load=load
        app=Flask("warmup-health")
        register_three_tier_api(app,runtime,busy=threading.Lock())
        self.assertEqual(app.test_client().get("/health").status_code,503)
        served=[]
        def serve(app, **kwargs):
            self.assertEqual(app.test_client().get("/health").status_code,200)
            self.assertTrue(high.loaded)
            served.append(kwargs)
        warmup_three_tier_before_serve(runtime,app,serve,host="127.0.0.1",port=9690,threads=4)
        self.assertEqual(load_calls,[1])
        self.assertEqual(len(served),1)
        self.assertIs(runtime.active_backend,high)
        runtime.warmup_default_model()
        self.assertEqual(load_calls,[1])

    def test_failed_warmup_never_binds_health_port(self):
        from neral_preview_server import warmup_three_tier_before_serve
        runtime, low, high, old=make_runtime()
        high._load=Mock(side_effect=RuntimeError("No CUDA"))
        app=Flask("failed-warmup")
        register_three_tier_api(app,runtime,busy=threading.Lock())
        serve=Mock()
        with self.assertRaises(RuntimeError):
            warmup_three_tier_before_serve(runtime,app,serve,host="127.0.0.1",port=9690,threads=4)
        serve.assert_not_called()
        self.assertEqual(app.test_client().get("/health").status_code,503)


if __name__=="__main__":
    unittest.main()
