"""CPU-only reproductions for post-PR602 Quartz math and read-page year evidence.

Only public synthetic input. No GPU, live network, sealed benchmark or disk files.
"""
import json
import unittest
from unittest.mock import Mock, patch

from neral_v42_quality import exact_executed_numeric_tail, year_evidence_verdict
from neral_evidence_agent import run_evidence_agent
from scripts.neral_v41_blind200_live import _exact_text
from scripts.neral_v42_telemetry_probe import DIAGNOSTIC_CASES
from test_neral_three_tier_api import make_runtime

MATH = DIAGNOSTIC_CASES[1][1]
YEAR = DIAGNOSTIC_CASES[2][1]
QUOTE = "Apollo 11 launched on July 16, 1969 and landed on the Moon on July 20, 1969."
URL = "https://example.org/apollo11-mission"
PAGE = {"url": URL, "page_read": True, "page_text": QUOTE}


class RealisticMathStdoutCPU(unittest.TestCase):
    def test_labeled_multistep_stdout_is_verified_from_user_numbers(self):
        for output in (
            "Gesamtlieferung: 391\nRest: 345",
            "Gesamtlieferung = 391 Bauteile\nRestmenge = 345 Bauteile",
            "391 Bauteile\n345 Bauteile",
            "Gesamtlieferung: 391; Rest: 345",
        ):
            with self.subTest(output=output):
                self.assertEqual(exact_executed_numeric_tail(
                    output, succeeded=True, prompt=MATH), "ENDERGEBNIS: 345")

    def test_wrong_duplicate_or_swapped_numbers_fail_closed(self):
        for output in (
            "Gesamtlieferung: 392\nRest: 345",
            "Gesamtlieferung: 391\nRest: 346",
            "Rest: 345\nGesamtlieferung: 391",
            "Gesamtlieferung: 391\nRest: 345\nAlternative: 346",
            "Gesamtlieferung: 391; Rest: 345; Alternative: 346",
            "391 und 345 oder 346",
        ):
            with self.subTest(output=output):
                self.assertIsNone(exact_executed_numeric_tail(
                    output, succeeded=True, prompt=MATH))

    def test_other_source_numbers_never_use_fixed_diagnostic_answer(self):
        task=("Ein Lager erhält zuerst 12 Kisten mit jeweils 9 Teilen und gibt "
              "danach 10 Teile ab. Berechne zuerst die Gesamtlieferung und danach "
              "den Rest. ENDERGEBNIS: <ganze Zahl>")
        self.assertEqual(exact_executed_numeric_tail(
            "Lieferung: 108\nRest: 98", succeeded=True, prompt=task),
            "ENDERGEBNIS: 98")
        self.assertIsNone(exact_executed_numeric_tail(
            "Lieferung: 108\nRest: 345", succeeded=True, prompt=task))

    def test_quartz_real_runtime_keeps_tool_output_and_last_line(self):
        runtime, _, high, _ = make_runtime()
        generations=[]
        def generate(*args, **kwargs):
            high.loaded = True
            generations.append(1)
            return "&#96;&#96;&#96;python\nprint('Gesamtlieferung: 391')\nprint('Rest: 345')\n&#96;&#96;&#96;".replace("&#96;", chr(96))
        high.respond_chat=generate
        runtime.execute_python=Mock(return_value={
            "status":"PASSED","output":"Gesamtlieferung: 391\nRest: 345"})
        answer=runtime.answer([
            {"role":"system","content":"Nur die aktuelle Frage."},
            {"role":"user","content":MATH},
        ],"Quartz")
        self.assertTrue(answer["sandbox_executed"])
        self.assertIn("Gesamtlieferung: 391",answer["text"])
        self.assertEqual(_exact_text(answer["text"],"integer"),"345")
        self.assertEqual(len(generations),1)


class RobustSourceYearCPU(unittest.TestCase):
    def test_same_year_twice_in_real_quote_is_not_a_contradiction(self):
        packet={"answer":"Apollo 11 landete 1969.","citations":[
            {"source":"web1","quote":QUOTE}]}
        evidence={"user":YEAR,"web1":QUOTE}
        self.assertEqual(year_evidence_verdict(
            YEAR,packet,evidence,[PAGE],[]),"supported")
        mixed=QUOTE+" Apollo 11 landed in 1970."
        packet["citations"][0]["quote"]=mixed
        self.assertEqual(year_evidence_verdict(
            YEAR,packet,{"user":YEAR,"web1":mixed},
            [{**PAGE,"page_text":mixed}],[]),"unsupported_year")

    def _agent(self, page_text, answer="Apollo 11 landete 1969.", extra_pages=None):
        packets=[{"tool":"direct_reasoning","parts":["Beleg lesen"]},
                 {"answer":answer,"citations":["p1"]}]
        def respond(*_):
            return json.dumps(packets.pop(0)) if packets else json.dumps(
                {"ok":False,"issue":"Zahl durch Quelle nicht belegt"})
        class Search:
            def search(self,_):
                return {"results":[{"url":URL,"snippet":"untrusted 1968"}]}
        pages=[{**PAGE,"page_text":page_text}]+(extra_pages or [])
        result=run_evidence_agent(
            YEAR,respond,Mock(),web_query="Apollo 11 erste Mondlandung Jahr",
            search=Search(),page_reader=lambda hits,q:pages,max_attempts=1)
        return result

    def test_literal_year_at_end_of_long_page_is_not_lost_to_clipping(self):
        page=("Navigation und Metadaten ohne Jahresbeleg.\n"*125)+QUOTE
        result=self._agent(page)
        self.assertTrue(result["agent_checked"],result)
        self.assertIsNone(result["error_code"])
        self.assertIn(QUOTE,result["text"])

    def test_wrong_model_year_is_repaired_only_from_read_page(self):
        result=self._agent(QUOTE,answer="Apollo 11 landete 1970.")
        self.assertTrue(result["agent_checked"],result)
        self.assertIsNone(result["error_code"])
        self.assertIn("1969",result["text"])
        self.assertNotIn("1970",result["text"])
        self.assertFalse(result["verified"])
        self.assertIn("[web1]",result["text"])

    def test_unread_page_or_conflicting_years_must_never_be_repaired(self):
        opposite="Apollo 11 landed on the Moon in 1970."
        result=self._agent(QUOTE,answer="Apollo 11 landete 1975.",
                           extra_pages=[{"url":"https://example.net/another",
                                         "page_read":True,"page_text":opposite}])
        self.assertFalse(result.get("agent_checked",False),result)
        self.assertEqual(result.get("error_code"),"checker_veto")

    def test_merely_a_plan_or_denial_is_not_historical_evidence(self):
        from neral_v42_quality import _year_quote_supports
        for bad in ("Apollo 11 planned to land on the Moon in 1969.",
                    "Apollo 11 did not land on the Moon in 1969."):
            self.assertFalse(_year_quote_supports(YEAR,bad,"1969"))


if __name__=="__main__":
    unittest.main()
