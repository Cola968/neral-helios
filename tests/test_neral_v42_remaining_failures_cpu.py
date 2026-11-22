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



class SurgicalMathAndWebRegressionsCPU(unittest.TestCase):
    def test_stock_stdout_variants_and_source_consistency(self):
        valid = (
            "Gesamt: 391\nErgebnis: 345",
            "Gesamt: 391 / Ergebnis: 345",
            "17 * 23 = 391\n391 - 46 = 345",
            "Gesamtlieferung: 391\nRest: 345\nENDERGEBNIS: 345",
            "345",
        )
        for stdout in valid:
            with self.subTest(stdout=stdout):
                self.assertEqual(
                    exact_executed_numeric_tail(stdout, succeeded=True, prompt=MATH),
                    "ENDERGEBNIS: 345")
        invalid = (
            "346",
            "Gesamt: 392\nErgebnis: 345",
            "Gesamt: 391\nErgebnis: 346",
            "Ergebnis: 345\nGesamt: 391",
            "17 * 24 = 391\n391 - 46 = 345",
            "17 * 23 = 391\n391 - 45 = 345",
            "Gesamt: 391\nErgebnis: 345\nENDERGEBNIS: 346",
            "Gesamt: 391\nErgebnis: 345\nENDERGEBNIS: 345\nENDERGEBNIS: 345",
        )
        for stdout in invalid:
            with self.subTest(stdout=stdout):
                self.assertIsNone(exact_executed_numeric_tail(
                    stdout, succeeded=True, prompt=MATH))

    def test_quartz_runtime_preserves_one_terminal_result_from_three_lines(self):
        from neral_v42_quality import final_marked_value
        runtime, _, high, _ = make_runtime()
        calls = []
        def generation(*_args, **_kw):
            calls.append(1)
            high.loaded = True
            return chr(96)*3 + "python\nprint(391)\nprint(345)\n" + chr(96)*3
        high.respond_chat = generation
        runtime.execute_python = Mock(return_value={"status": "PASSED",
            "output": "Gesamt: 391\nErgebnis: 345\nENDERGEBNIS: 345"})
        response = runtime.answer([
            {"role":"system","content":"Only current question"},
            {"role":"user","content": MATH}], "Quartz")
        lines = [row.strip() for row in response["text"].splitlines() if row.strip()]
        self.assertEqual(lines.count("ENDERGEBNIS: 345"), 1)
        self.assertEqual(lines[-1], "ENDERGEBNIS: 345")
        self.assertEqual(final_marked_value(response["text"]), "345")
        self.assertEqual(len(calls), 1)

    def test_repeated_year_in_answer_is_one_unique_year(self):
        packet = {"answer": "Apollo 11 landete 1969, im Jahr 1969.",
                  "citations": [{"source": "web1", "quote": QUOTE}]}
        self.assertEqual(year_evidence_verdict(
            YEAR, packet, {"user": YEAR, "web1": QUOTE}, [PAGE], []), "supported")
        packet["answer"] = "Apollo 11 landete 1969, nicht 1970."
        self.assertEqual(year_evidence_verdict(
            YEAR, packet, {"user": YEAR, "web1": QUOTE}, [PAGE], []),
            "unsupported_year")

    def test_two_adjacent_sentences_keep_year_attached_to_event(self):
        from neral_v42_quality import unique_read_year_source
        text = "Apollo 11 landed the first humans on the Moon.\nThe mission took place in July 1969."
        page = {**PAGE, "page_text": text}
        found = unique_read_year_source(YEAR, [page])
        self.assertIsNotNone(found)
        self.assertEqual(found["year"], "1969")
        self.assertIn("Apollo 11", found["quote"])
        self.assertIn("1969", found["quote"])
        self.assertIn(found["quote"], text)
        conflict = {**PAGE, "page_text": text + "\nApollo 11 landed on the Moon in 1970."}
        self.assertIsNone(unique_read_year_source(YEAR, [page, conflict]))

    def test_long_html_keeps_adjacent_year_lines(self):
        from neral_page_reader import MainText, MAX_TEXT
        filler = "".join(
            "<p>Apollo mission overview and flight chronology section %d.</p>" % i
            for i in range(180))
        html = ("<main>" + filler +
                "<p>Apollo 11 landed the first humans on the Moon.</p>"
                "<p>The mission took place in July 1969.</p></main>")
        parser = MainText()
        parser.feed(html)
        extracted = parser.text("Apollo 11 first humans landing Moon year")
        self.assertLessEqual(len(extracted), MAX_TEXT)
        self.assertIn("Apollo 11 landed the first humans on the Moon.\nThe mission took place in July 1969.", extracted)

    def test_agent_shortcuts_two_sentence_https_without_worker(self):
        sources = [{**PAGE, "page_text":
            "Apollo 11 landed the first humans on the Moon.\nThe mission took place in July 1969."}]
        calls = []
        def respond(*args):
            calls.append(1)
            return json.dumps({"tool":"direct_reasoning","parts":["Get year"],"search_query":""})
        class Search:
            def search(self, _):
                return {"results":[{"url": URL, "snippet":"untrusted 1971"}]}
        result = run_evidence_agent(YEAR, respond, Mock(),
            web_query="Apollo 11 NASA moon landing", search=Search(),
            page_reader=lambda hits, query:sources, deep=True, max_attempts=3)
        self.assertEqual(len(calls),1)
        self.assertTrue(result["agent_checked"],result)
        self.assertEqual(result["error_code"],None)
        self.assertEqual(result.get("source_year_shortcut"),True)

    def test_no_new_passage_stops_repair_loop_and_exposes_subcode(self):
        sources = [{**PAGE, "page_text":"Apollo 11 mission overview.\n1969 is an important year."}]
        packets = [
            {"tool":"direct_reasoning","parts":["Get year"],"search_query":""},
            {"answer":"Apollo 11 landed in 1969.", "citations":["p1"]},
        ]
        def respond(*_args):
            self.assertTrue(packets, "same ungrounded source must not regenerate")
            return json.dumps(packets.pop(0))
        class Search:
            def search(self,_):
                return {"results":[{"url": URL, "snippet":"1969"}]}
        result = run_evidence_agent(YEAR, respond, Mock(),
            web_query="Apollo 11 NASA moon landing", search=Search(),
            page_reader=lambda hits, q:sources,deep=True,max_attempts=3)
        self.assertEqual(result["error_code"],"checker_veto")
        self.assertEqual(result["agent_attempts"],1)
        self.assertEqual(result.get("checker_veto_detail_code"),"unsupported_year_no_passage")

if __name__=="__main__":
    unittest.main()
