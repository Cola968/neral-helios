"""V4.2 follow-up regressions: CPU only, fake HF, synthetic public examples."""
import json
import threading
import unittest
from unittest.mock import Mock, patch
from flask import Flask
from neral_three_tier_runtime import register_three_tier_api
from neral_v42_quality import exact_executed_numeric_tail, year_evidence_verdict, checker_veto_class
from neral_evidence_agent import _passages, run_evidence_agent
from scripts.neral_v41_blind200_live import _exact_text
from scripts.neral_v42_telemetry_probe import DIAGNOSTIC_CASES
from test_neral_three_tier_api import make_runtime

MATH = DIAGNOSTIC_CASES[1][1]
YEAR = DIAGNOSTIC_CASES[2][1]
QUOTE = "Apollo 11 landete 1969 auf dem Mond und brachte Menschen auf die Mondoberflaeche."
SOURCE = {"url": "https://www.nasa.gov/history/apollo-11", "page_read": True,
          "page_text": "Apollo 11 wurde 1968 vorbereitet.\n" + QUOTE}


class ObsidianFastPathCPU(unittest.TestCase):
    def test_http_uses_allowed_9b_budget_and_one_generation(self):
        runtime, low, high, _ = make_runtime()
        runtime.evidence_agent = True
        budgets = []
        def strict_backend(turns, *, max_new_tokens, **kwargs):
            if max_new_tokens not in (256, 768):
                raise ValueError("Unapproved NERAL inference budget")
            budgets.append(max_new_tokens)
            high.loaded = True
            return "Wasserdampf kondensiert am kuehlen Spiegel."
        high.respond_chat = strict_backend
        app = Flask("strict-ob-fast")
        register_three_tier_api(app, runtime, busy=threading.Lock(), search_service=Mock())
        with patch("neral_verified_notes.exact_note_lookup", return_value=None), \
             patch.object(runtime, "answer_with_evidence", side_effect=AssertionError("No Deep route")):
            response = app.test_client().post("/api/neral/chat/three-tier", json={
                "tier": "Obsidian", "prompt": DIAGNOSTIC_CASES[0][1], "history": [],
                "benchmark_isolated": True})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(response.json["obsidian_fast_path"])
        self.assertEqual(budgets, [256])


class TwoStepMathCPU(unittest.TestCase):
    def test_canonical_only_for_source_matched_steps(self):
        self.assertEqual(exact_executed_numeric_tail(
            "345", succeeded=True, prompt=MATH), "ENDERGEBNIS: 345")
        self.assertEqual(exact_executed_numeric_tail(
            "391\n345", succeeded=True, prompt=MATH), "ENDERGEBNIS: 345")
        for raw in ("391\n346", "392\n345", "391\n345\n346",
                    "345\n346", "391\nENDERGEBNIS: 345\nENDERGEBNIS: 346",
                    "391\n345 oder 346"):
            with self.subTest(raw=raw):
                self.assertIsNone(exact_executed_numeric_tail(
                    raw, succeeded=True, prompt=MATH))
        self.assertIsNone(exact_executed_numeric_tail(
            "391\n345", succeeded=False, prompt=MATH))
        self.assertIsNone(exact_executed_numeric_tail(
            "391\n345", succeeded=True, prompt="Berechne 17 plus 23."))

    def test_quartz_multiline_stdout_http_numeric_final(self):
        runtime, low, high, _ = make_runtime()
        generated = []
        def propose(turns, **kwargs):
            high.loaded = True
            generated.append(turns)
            return "\x60\x60\x60python\nprint(17*23)\nprint(17*23-46)\n\x60\x60\x60"
        high.respond_chat = propose
        runtime.execute_python = Mock(return_value={"status": "PASSED", "output": "391\n345"})
        response = runtime.answer([
            {"role": "system", "content": "Berechne."},
            {"role": "user", "content": MATH}], "Quartz")
        self.assertTrue(response["sandbox_executed"])
        self.assertIn("391\n345", response["text"])
        self.assertEqual(_exact_text(response["text"], "integer"), "345")
        self.assertEqual(len(generated), 1)

    def _deep(self, stdout):
        packets = [
            {"tool":"python_sandbox","parts":["Lieferung und Rest"],"search_query":""},
            {"locale":"de","calculations":[
                {"label":"Lieferung","expr":"u1 * u2","unit":"Bauteile"},
                {"label":"Rest","expr":"u1 * u2 - u3","unit":"Bauteile"}],"missing":[]},
            {"ok":True,"issue":""},
        ]
        def model(*_):
            return json.dumps(packets.pop(0))
        execute=Mock(return_value={"status":"PASSED","output":stdout})
        with patch("neral_evidence_agent.quantity_plan", return_value=None), \
             patch("neral_v32_verifier.verified_math_question", return_value=None), \
             patch("neral_v32_verifier.verified_python_question", return_value=None), \
             patch("neral_v32_verifier.verified_shift_count", return_value=None), \
             patch("neral_knowledge_verifier.verified_element_question", return_value=None):
            result=run_evidence_agent(MATH,model,execute,deep=True,max_attempts=1)
        return result,execute

    def test_deep_two_step_outputs_end_in_canonical_value(self):
        result, tool = self._deep("391\n345")
        self.assertTrue(result["agent_checked"],result)
        self.assertIn("Lieferung: 391",result["text"])
        self.assertEqual(_exact_text(result["text"],"integer"),"345")
        tool.assert_called_once()

    def test_deep_wrong_second_step_never_gets_canonical(self):
        result,_ = self._deep("391\n346")
        self.assertIsNone(_exact_text(result.get("text",""),"integer"))


class SourceYearCPU(unittest.TestCase):
    def test_passages_prefer_landing_year_and_skip_unread(self):
        evidence={"user":YEAR,"web1":SOURCE["page_text"]}
        found=_passages(evidence,YEAR,sources=[SOURCE])
        self.assertEqual(found["p1"][1],QUOTE)
        self.assertEqual(_passages(evidence,YEAR,sources=[{**SOURCE,"page_read":False}]),{})

    def test_literal_source_year_verdicts_are_distinct(self):
        packet={"answer":"Apollo 11 landete 1969.","citations":[{"source":"web1","quote":QUOTE}]}
        evidence={"user":YEAR,"web1":QUOTE}
        self.assertEqual(year_evidence_verdict(YEAR,packet,evidence,[SOURCE],[]),"supported")
        self.assertEqual(year_evidence_verdict(YEAR,
            {**packet,"answer":"Apollo 11 landete 1968."},evidence,[SOURCE],[]),"unsupported_year")
        self.assertEqual(year_evidence_verdict(YEAR,
            {**packet,"citations":[{"source":"web1","quote":"Nur ein Snippet: 1969"}]},
            evidence,[SOURCE],[]),"invalid_quote")
        self.assertEqual(year_evidence_verdict(YEAR,packet,evidence,
            [{**SOURCE,"page_read":False}],[]),"no_read_page")
        self.assertEqual(year_evidence_verdict(YEAR,packet,evidence,
            [{**SOURCE,"page_text":"Fremder Artikel"}],[]),"invalid_quote")

    def test_web_answer_supported_year_needs_no_fallible_model_critic(self):
        class Search:
            calls=0
            def search(self,_):
                self.calls+=1
                return {"results":[{"url":SOURCE["url"],"snippet":"1969"}]}
        search=Search()
        packets=[{"tool":"direct_reasoning","parts":["Beleg pruefen"]},
                 {"answer":"Apollo 11 landete 1969.","citations":["p1"]}]
        def model(*_):
            if not packets:
                raise AssertionError("Valid literal year must not require model critic")
            return json.dumps(packets.pop(0))
        result=run_evidence_agent(YEAR,model,Mock(),web_query="Apollo 11 Mondlandung Jahr NASA",
                                  search=search,page_reader=lambda hits,q:[SOURCE],max_attempts=1)
        self.assertEqual(search.calls,1)
        self.assertTrue(result["agent_checked"],result)
        self.assertIsNone(result["error_code"])
        self.assertIn("[web1]",result["text"])
        self.assertFalse(result["verified"])

    def test_privacy_safe_error_taxonomy(self):
        self.assertEqual(checker_veto_class(
            "Keine gelesene Webseite als Quellenbeleg verfügbar."),"no_read_page")
        self.assertEqual(checker_veto_class(
            "Wörtlicher Quellenbeleg fehlt."),"invalid_quote")
        self.assertEqual(checker_veto_class(
            "Unbelegte Jahreszahl in der Antwort."),"unsupported_year")
        self.assertEqual(checker_veto_class(
            "Konkreter Widerspruch: Beleg verneint die Aussage."),"critic_contradiction")


if __name__=="__main__":
    unittest.main()
