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


class ThreeSentenceYearEvidenceCPU(unittest.TestCase):
    def test_three_adjacent_sentences_can_ground_one_unique_event_year(self):
        from neral_v42_quality import unique_read_year_source, year_evidence_verdict
        text = ("Apollo 11 landed the first humans on the Moon.\n"
                "The crew descended toward the lunar surface.\n"
                "The landing took place in July 1969.")
        page = {**PAGE, "page_text": text}
        candidate = unique_read_year_source(YEAR, [page])
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["year"], "1969")
        self.assertEqual(candidate["quote"], text)
        self.assertEqual(year_evidence_verdict(
            YEAR, {"answer":"1969",
                   "citations":[{"source":"web1","quote":text}]},
            {"user":YEAR,"web1":text}, [page], []), "supported")

    def test_competing_apollo_mission_cannot_supply_the_year(self):
        from neral_v42_quality import unique_read_year_source, _year_quote_supports
        text = ("Apollo 11 landed the first humans on the Moon.\n"
                "The Apollo 12 spacecraft was a separate mission.\n"
                "Apollo 12 landed in November 1969.")
        self.assertIsNone(unique_read_year_source(
            YEAR, [{**PAGE, "page_text":text}]))
        self.assertFalse(_year_quote_supports(YEAR, text, "1969"))

    def test_long_html_preserves_three_line_event_and_year(self):
        from neral_page_reader import MainText, MAX_TEXT
        filler = "".join(
            "<p>Apollo mission overview and flight chronology section %d.</p>" % i
            for i in range(180))
        text = ("Apollo 11 landed the first humans on the Moon.\n"
                "The astronauts descended toward the lunar surface.\n"
                "The landing took place in July 1969.")
        html = ("<main>" + filler +
                "<p>Apollo 11 landed the first humans on the Moon.</p>"
                "<p>The astronauts descended toward the lunar surface.</p>"
                "<p>The landing took place in July 1969.</p></main>")
        parser = MainText()
        parser.feed(html)
        actual = parser.text("Apollo 11 first humans landing Moon year")
        self.assertLessEqual(len(actual), MAX_TEXT)
        self.assertIn(text, actual)

    def test_agent_shortcuts_three_sentence_literal_without_worker(self):
        text = ("Apollo 11 landed the first humans on the Moon.\n"
                "The crew descended toward the lunar surface.\n"
                "The landing took place in July 1969.")
        sources = [{**PAGE, "page_text": text}]
        calls = []
        def respond(*args):
            calls.append(1)
            return json.dumps({"tool":"direct_reasoning","parts":["Get year"],"search_query":""})
        class Search:
            def search(self, _):
                return {"results":[{"url": URL, "snippet":"untrusted 1971"}]}
        result = run_evidence_agent(
            YEAR, respond, Mock(), web_query="Apollo 11 NASA moon landing",
            search=Search(), page_reader=lambda hits,q:sources,
            deep=True, max_attempts=3)
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["agent_checked"], result)
        self.assertEqual(result.get("source_year_shortcut"), True)


class ApolloInfoboxAndQuartzStdoutRegressionCPU(unittest.TestCase):
    """Synthetic only: no source URL fetched, no private benchmark cases."""

    def test_apollo_infobox_outside_main_survives_4600_char_window(self):
        from neral_page_reader import MainText, MAX_TEXT
        from neral_v42_quality import year_source_pipeline_code, unique_read_year_source
        date = "Apollo 11 landed on the Moon on 20 July 1969."
        filler = "".join(
            "<p>Apollo 11 NASA moon landing mission overview, section %d. "
            "General information about missions without any year.</p>" % i
            for i in range(170))
        html = (
            '<table class="infobox vevent"><tbody>'
            '<tr><th scope="row">Landing</th><td>' + date + '</td></tr>'
            '</tbody></table>'
            '<main><h1>Apollo 11</h1>' + filler + '</main>')
        parser = MainText()
        parser.feed(html)
        excerpt = parser.text("Apollo 11 NASA moon landing year")
        self.assertLessEqual(len(excerpt), MAX_TEXT)
        self.assertIn(date, excerpt)
        self.assertLess(excerpt.index(date), 800)
        pages = [{"url": URL, "page_read": True, "page_text": excerpt,
                  "year_clip_code": parser.year_clip_code}]
        self.assertEqual(parser.year_clip_code, "year_preserved_in_extract")
        self.assertEqual(year_source_pipeline_code(YEAR, pages, {"web1": excerpt}),
                         "candidate_available")
        candidate = unique_read_year_source(YEAR, pages)
        self.assertIsNotNone(candidate)
        self.assertIn(candidate["quote"], excerpt)

    def test_dated_lead_paragraph_after_long_repetitive_content_wins_ranking(self):
        from neral_page_reader import MainText, MAX_TEXT
        date = "Apollo 11 landed on the Moon on 20 July 1969."
        filler = "".join(
            "<p>NASA Apollo 11 moon landing mission section %d. "
            "NASA Apollo 11 moon landing mission chronology notes.</p>" % i
            for i in range(180))
        markup = "<main>" + filler + "<p>" + date + "</p></main>"
        parser = MainText()
        parser.feed(markup)
        excerpt = parser.text("Apollo 11 NASA moon landing year")
        self.assertLessEqual(len(excerpt), MAX_TEXT)
        self.assertIn(date, excerpt)
        self.assertLess(excerpt.index(date), 800)

    def test_undated_or_different_mission_infobox_never_proves_1969(self):
        from neral_page_reader import MainText
        from neral_v42_quality import unique_read_year_source
        markup = ('<table class="infobox"><tr><th>Landing</th>'
                  '<td>Apollo 12 landed on the Moon in November 1969.</td>'
                  '</tr></table><main><p>Apollo 11 was a lunar mission.</p></main>')
        parser = MainText()
        parser.feed(markup)
        excerpt = parser.text("Apollo 11 NASA moon landing year")
        pages = [{"url": URL, "page_read": True, "page_text": excerpt}]
        self.assertIsNone(unique_read_year_source(YEAR, pages))

    def test_two_step_stock_stdout_accepts_verified_realistic_single_line_forms(self):
        from neral_v42_quality import stock_stdout_diagnostic_code
        valid = (
            "Gesamt: 391, Rest: 345",
            "Gesamt: 391; Rest: 345",
            "17 * 23 - 46 = 345",
            "17 × 23 - 46 = 345",
            "Gesamt: 391\nNach Abgabe von 46 Bauteilen bleiben 345 übrig",
            ("Die Gesamtlieferung beträgt 391 Bauteile. "
             "Nach Abgabe von 46 Bauteilen bleiben 345 Bauteile übrig."),
        )
        for output in valid:
            with self.subTest(output=output):
                self.assertEqual(exact_executed_numeric_tail(
                    output, succeeded=True, prompt=MATH), "ENDERGEBNIS: 345")
                self.assertEqual(stock_stdout_diagnostic_code(
                    output, succeeded=True, prompt=MATH), "accepted")

    def test_new_forms_reject_wrong_inputs_contradiction_and_extra_output(self):
        bad = (
            "Gesamt: 391, Rest: 346",
            "Gesamt: 392, Rest: 345",
            "Gesamt: 391, Rest: 345, Rest: 346",
            "Gesamt: 391, Rest: 345\nRest: 346",
            "17 * 23 - 45 = 346",
            "17 * 24 - 46 = 362",
            "17 * 23 - 46 = 345; ENDERGEBNIS: 346",
            "Gesamt: 391\nNach Abgabe von 47 Bauteilen bleiben 345 übrig",
            ("Die Gesamtlieferung beträgt 391 Bauteile. "
             "Nach Abgabe von 46 Bauteilen bleiben 344 Bauteile übrig."),
            "Gesamt: 391\nAntwort: 345",
        )
        for output in bad:
            with self.subTest(output=output):
                self.assertIsNone(exact_executed_numeric_tail(
                    output, succeeded=True, prompt=MATH))

    def test_quartz_emits_single_terminal_result_without_second_generation(self):
        runtime, _, high, _ = make_runtime()
        generations = []
        def generation(*args, **kw):
            generations.append(1)
            high.loaded = True
            return chr(96)*3 + "python\nprint('Gesamt: 391, Rest: 345')\n" + chr(96)*3
        high.respond_chat = generation
        runtime.execute_python = Mock(return_value={
            "status": "PASSED", "output": "Gesamt: 391, Rest: 345"})
        response = runtime.answer([
            {"role": "system", "content": "Nur die aktuelle Aufgabe"},
            {"role": "user", "content": MATH}], "Quartz")
        self.assertTrue(response["sandbox_executed"])
        lines = [line.strip() for line in response["text"].splitlines() if line.strip()]
        self.assertEqual(lines[-1], "ENDERGEBNIS: 345")
        self.assertEqual(lines.count("ENDERGEBNIS: 345"), 1)
        self.assertEqual(response["math_stdout_diagnostic_code"], "accepted")
        self.assertEqual(len(generations), 1)


class OwnerApolloReaderStageContractCPU(unittest.TestCase):
    def test_approved_diag_stages_backs_up_and_restores_page_reader(self):
        from pathlib import Path
        owner = (Path(__file__).resolve().parents[1] /
                 "scripts" / "neral_v41_owner.ps1").read_text(encoding="utf-8")
        self.assertEqual(owner.count("$files=@("), 1)
        tracked = owner.split("$files=@(", 1)[1].split(")\n$backup=", 1)[0]
        self.assertIn("'neral_page_reader.py'", tracked)
        self.assertIn("foreach($file in $files)", owner)
        self.assertIn("Experiment rollback checksum mismatch:", owner)
        self.assertIn("V4 installed module hash mismatch:", owner)
        # Never consider this original installed dependency optional.
        optional = owner.split("elseif($file -in @(", 1)[1].split("))", 1)[0]
        self.assertNotIn("'neral_page_reader.py'", optional)


class BoundedYearPriorityAndFinalNumericCPU(unittest.TestCase):
    """New regression scenarios; never read benchmark fixtures or remote URLs."""

    def test_long_article_keeps_literal_apollo_landing_year_window(self):
        from neral_page_reader import MainText, MAX_TEXT
        from neral_v42_quality import unique_read_year_source, year_source_pipeline_code
        undated = "".join(
            "<p>Apollo 11 NASA lunar landing exploration mission detail "
            "without chronological dates, section %d. Another undated "
            "Apollo 11 mission summary.</p>" % i for i in range(180))
        literal = ("Apollo 11 landed the first humans on the Moon. "
                   "The crew reached the surface on July 20, 1969.")
        html = "<main>" + undated + "<p>" + literal + "</p></main>"
        p = MainText()
        p.feed(html)
        excerpt = p.text("Apollo 11 first humans moon landing year NASA")
        self.assertLessEqual(len(excerpt), MAX_TEXT)
        self.assertIn(literal, excerpt)
        self.assertLess(excerpt.index(literal), 900)
        self.assertEqual(p.year_clip_code, "year_preserved_in_extract")
        pages = [{"url": URL, "page_read": True, "page_text": excerpt,
                  "year_clip_code": p.year_clip_code}]
        self.assertEqual(year_source_pipeline_code(YEAR, pages,
                                                  {"web1": excerpt}),
                         "candidate_available")
        self.assertIsNotNone(unique_read_year_source(YEAR, pages))

    def test_long_single_paragraph_uses_bounded_literal_sentence_window(self):
        from neral_page_reader import MainText, MAX_TEXT
        filler = "Apollo 11 mission study without a flight date. " * 180
        date = "Apollo 11 landed on the Moon on July 20, 1969."
        html = "<main><p>" + filler + date + "</p></main>"
        p = MainText()
        p.feed(html)
        excerpt = p.text("Apollo 11 moon landing year")
        self.assertLessEqual(len(excerpt), MAX_TEXT)
        self.assertIn(date, excerpt)
        self.assertLess(excerpt.index(date), 900)

    def test_year_priority_never_uses_competing_apollo_mission(self):
        from neral_page_reader import MainText
        from neral_v42_quality import unique_read_year_source
        article = ("<main>" + "<p>Apollo 11 first landing mission overview.</p>" * 140
                   + "<p>Apollo 12 landed on the Moon in November 1969.</p></main>")
        p = MainText()
        p.feed(article)
        excerpt = p.text("Apollo 11 first landing year")
        pages = [{"url": URL, "page_read": True, "page_text": excerpt}]
        self.assertIsNone(unique_read_year_source(YEAR, pages))

    def test_quartz_accepts_free_prose_with_last_independent_end_value(self):
        from neral_v42_quality import stock_stdout_diagnostic_code
        outputs = (
            "Es wurden 391 Bauteile geliefert.\nNach Abzug von 46 bleiben 345 übrig.",
            "Zuerst gab es 17 Kisten zu je 23 Teilen.\n"
            "Die Gesamtmenge lautet 391.\nNach Abgabe von 46 sind noch 345 vorhanden.",
            "Zwischenschritt: 391\n46 Teile abgegeben.\nDer Rest beträgt 345.",
            "Es ergibt sich 391, danach gehen 46 weg und übrig bleiben 345.",
        )
        for output in outputs:
            with self.subTest(output=output):
                self.assertEqual(exact_executed_numeric_tail(
                    output, succeeded=True, prompt=MATH), "ENDERGEBNIS: 345")
                self.assertEqual(stock_stdout_diagnostic_code(
                    output, succeeded=True, prompt=MATH), "accepted")

    def test_quartz_final_numeric_fails_closed_for_contradictions(self):
        bad = (
            "Es wurden 391 Bauteile geliefert.\nNach Abzug von 46 bleiben 344 übrig.",
            "Es wurden 392 Bauteile geliefert.\nNach Abzug von 46 bleiben 345 übrig.",
            "Es wurden 391 Bauteile geliefert.\nNach Abzug von 47 bleiben 345 übrig.",
            "Rest 346. Nach Prüfung doch 345.",
            "Rest 391. Nach Korrektur 345.",
            "Es wurden 391 Bauteile geliefert.\nRest ist 345.\nWeitere Ausgabe 391.",
            "Achtung: 17 * 23 - 46 = 344, aber die Antwort lautet 345.",
            "Gesamt: 391\nAntwort: 345",
        )
        for output in bad:
            with self.subTest(output=output):
                self.assertIsNone(exact_executed_numeric_tail(
                    output, succeeded=True, prompt=MATH))
        self.assertIsNone(exact_executed_numeric_tail(
            "Es wurden 391 Bauteile geliefert.\nNach Abzug von 46 bleiben 345 übrig.",
            succeeded=False, prompt=MATH))

    def test_stock_code_prompt_requires_terminal_print_only_in_stock_mode(self):
        runtime, _, high, _ = make_runtime()
        code_prompts = []
        def model(turns, **kwargs):
            high.loaded = True
            code_prompts.append(turns[0]["content"])
            return chr(96)*3 + "python\nprint('345')\n" + chr(96)*3
        high.respond_chat = model
        runtime.execute_python = Mock(return_value={
            "status": "PASSED", "output": "345"})
        runtime.answer([
            {"role":"system","content":"Beantworte die Frage."},
            {"role":"user","content": MATH}], "Quartz")
        self.assertTrue(code_prompts)
        self.assertIn('print(f"ENDERGEBNIS: {ergebnis}")', code_prompts[0])
        self.assertIn("allerletzte", code_prompts[0])

if __name__=="__main__":
    unittest.main()
