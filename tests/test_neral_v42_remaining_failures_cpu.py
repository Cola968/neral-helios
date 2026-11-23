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


class GermanApolloYearContextCPU(unittest.TestCase):
    """Strict synthetic source excerpts, no network and no sealed test data."""

    def _candidate(self, text, **overrides):
        from neral_v42_quality import unique_read_year_source
        source = {**PAGE, "page_text": text, **overrides}
        return unique_read_year_source(YEAR, [source])

    def test_german_ordinal_stays_with_event_and_year(self):
        from neral_v42_quality import year_source_pipeline_code
        text = "Apollo 11\nDie Mission landete am 20. Juli 1969 auf dem Mond."
        found = self._candidate(text)
        self.assertIsNotNone(found)
        self.assertEqual(found["year"], "1969")
        self.assertIn("20. Juli 1969", found["quote"])
        self.assertIn(found["quote"], text)
        self.assertEqual(year_source_pipeline_code(
            YEAR, [{**PAGE, "page_text": text}], {"web1": text}),
            "candidate_available")

    def test_three_line_german_subject_pronoun_landing(self):
        text = ("Apollo 11\nDie Mission wurde von der NASA gesteuert.\n"
                "Sie landete am 20. Juli 1969 auf dem Mond.")
        candidate = self._candidate(text)
        self.assertIsNotNone(candidate)
        self.assertIn("Sie landete am 20. Juli 1969", candidate["quote"])
        self.assertIn(candidate["quote"], text)

    def test_verified_read_page_heading_can_anchor_distant_pronoun_window(self):
        from neral_v42_quality import year_evidence_verdict
        title = "Apollo 11"
        filler = ("Hintergrund der Raumfahrt ohne Datumsangabe.\n" * 12)
        date = "Die Mission landete auf dem Mond am 20. Juli 1969."
        text = title + "\n" + filler + date
        candidate = self._candidate(text, title="Irreführender Suchtreffer-Titel")
        self.assertIsNotNone(candidate)
        self.assertIn(date, candidate["quote"])
        self.assertIn(candidate["quote"], text)
        self.assertLessEqual(len(candidate["quote"]), 400)
        self.assertEqual(year_evidence_verdict(
            YEAR, {"answer": "1969", "citations": [
                {"source": "web1", "quote": candidate["quote"]}]},
            {"web1": text, "user": YEAR},
            [{**PAGE, "page_text": text,
              "title": "Irreführender Suchtreffer-Titel"}], []), "supported")

    def test_infobox_landing_fact_under_read_apollo_heading(self):
        from neral_page_reader import MainText
        from neral_v42_quality import year_source_pipeline_code
        html = ("<main><h1>Apollo 11</h1>"
                "<table class='infobox'><tr><th>Mondlandung</th>"
                "<td>20. Juli 1969</td></tr></table>"
                "<p>Weitere Informationen zur Raumfahrtmission.</p></main>")
        parser = MainText()
        parser.feed(html)
        excerpt = parser.text("Apollo 11 Mondlandung Jahr")
        candidate = self._candidate(excerpt)
        self.assertIsNotNone(candidate)
        self.assertIn("1969", candidate["quote"])
        self.assertIn(candidate["quote"], excerpt)
        self.assertEqual(year_source_pipeline_code(
            YEAR, [{**PAGE, "page_text": excerpt}], {"web1": excerpt}),
            "candidate_available")

    def test_search_result_title_alone_cannot_assign_pronoun_event(self):
        text = "Die Mission landete auf dem Mond am 20. Juli 1969."
        self.assertIsNone(self._candidate(text, title="Apollo 11"))

    def test_exact_subject_heading_with_nasa_subtitle(self):
        text = ("Apollo 11 – NASA\nMission zur Erkundung des Mondes.\n"
                "Die Mission landete am 20. Juli 1969 auf dem Mond.")
        candidate = self._candidate(text)
        self.assertIsNotNone(candidate)
        self.assertIn("20. Juli 1969", candidate["quote"])

    def test_pronoun_does_not_attach_to_adjacent_apollo_12(self):
        from neral_v42_quality import year_evidence_verdict
        quote = "Die Mission landete am 20. Juli 1969 auf dem Mond."
        text = ("Apollo 11\nApollo 12 wurde hier ebenfalls beschrieben.\n" +
                quote)
        self.assertIsNone(self._candidate(text))
        self.assertEqual(year_evidence_verdict(
            YEAR, {"answer": "1969", "citations": [
                {"source": "web1", "quote": quote}]},
            {"web1": text, "user": YEAR},
            [{**PAGE, "page_text": text}], []), "unsupported_year")


    def test_wrong_mission_or_negated_event_cannot_become_year_source(self):
        for text in (
            "Apollo 11\nApollo 12 landete am 20. Juli 1969 auf dem Mond.",
            "Apollo 11\nDie Mission landete nicht am 20. Juli 1969.",
            "Apollo 11\nDie Mission sollte am 20. Juli 1969 landen.",
            "Apollo 11\nDie Mission startete am 16. Juli 1969.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self._candidate(text))

    def test_conflicting_landing_years_on_read_page_are_rejected(self):
        text = ("Apollo 11\nMondlandung: 20. Juli 1969.\n"
                "Mondlandung: 20. Juli 1970.")
        self.assertIsNone(self._candidate(text))



class SearchTopicAndWikipediaFallbackCPU(unittest.TestCase):
    """Synthetic public Apollo search; no network, no model, no Blind200 data."""

    APOLLO_QUERY = "Apollo 11 erste bemannte Mondlandung Jahr NASA"
    BAD_HITS = [
        {"title": "Dein Optiker & Akustiker | Apollo",
         "url": "https://www.apollo.de/", "snippet": "Brillen und Kontaktlinsen"},
        {"title": "The AI GTM System | Apollo",
         "url": "https://www.apollo.io/", "snippet": "Sales intelligence"},
        {"title": "Apollo News",
         "url": "https://apollo-news.net/", "snippet": "Aktuelle Nachrichten 2026"},
    ]
    GOOD_HIT = {"title": "Apollo 11 – NASA Mondlandung",
                "url": "https://de.wikipedia.org/wiki/Apollo_11",
                "snippet": "Die NASA-Mission Apollo 11 brachte Menschen auf den Mond."}
    GOOD_TEXT = "Apollo 11\nDie Mission landete auf dem Mond am 20. Juli 1969."

    def test_query_rewriter_retains_numbered_entity_and_final_anchors(self):
        from neral_web_search import rewrite_search_query
        query = rewrite_search_query(self.APOLLO_QUERY)
        self.assertIn("Apollo 11", query)
        for word in ("Mondlandung", "Jahr", "NASA"):
            self.assertIn(word, query)
        self.assertLessEqual(len(query.split()), 8)
        self.assertNotIn("1969", query)

    def test_mixed_bing_aliases_do_not_consume_three_read_slots(self):
        from neral_page_reader import read_top_pages
        from neral_v42_quality import unique_read_year_source
        opened = []
        def fake_fetch(url, query):
            opened.append(url)
            return {"url": url, "text": self.GOOD_TEXT,
                    "year_clip_code": "year_preserved_in_extract"}
        rows = read_top_pages(self.BAD_HITS + [self.GOOD_HIT], self.APOLLO_QUERY,
                              fetch=fake_fetch, wiki_fallback=lambda q: [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(opened, [self.GOOD_HIT["url"]])
        self.assertIsNotNone(unique_read_year_source(YEAR, rows))

    def test_only_aliases_trigger_wikipedia_fallback_and_source_quote(self):
        from neral_page_reader import read_top_pages
        from neral_v42_quality import unique_read_year_source, year_source_pipeline_code
        opened = []
        fallback_requests = []
        def fake_fetch(url, query):
            opened.append(url)
            return {"url": url, "text": self.GOOD_TEXT,
                    "year_clip_code": "year_preserved_in_extract"}
        def wikipedia(query):
            fallback_requests.append(query)
            return [self.GOOD_HIT]
        pages = read_top_pages(self.BAD_HITS, self.APOLLO_QUERY,
                               fetch=fake_fetch, wiki_fallback=wikipedia)
        self.assertEqual(opened, [self.GOOD_HIT["url"]])
        self.assertEqual(len(fallback_requests), 1)
        self.assertEqual(pages[0]["url"], self.GOOD_HIT["url"])
        candidate = unique_read_year_source(YEAR, pages)
        self.assertIsNotNone(candidate)
        self.assertIn(candidate["quote"], pages[0]["page_text"])
        self.assertEqual(year_source_pipeline_code(
            YEAR, pages, {"web1": pages[0]["page_text"]}),
            "candidate_available")

    def test_relevant_metadata_but_off_topic_page_uses_wikipedia(self):
        from neral_page_reader import read_top_pages
        false_hit = {"title": "Apollo 11 Moon landing", "url": "https://example.org/page",
                     "snippet": "The Apollo 11 mission landed astronauts on the Moon"}
        opened = []
        def fake_fetch(url, query):
            opened.append(url)
            text = ("A new public product unrelated to space exploration."
                    if url == false_hit["url"] else self.GOOD_TEXT)
            return {"url": url, "text": text,
                    "year_clip_code": "year_preserved_in_extract"}
        rows = read_top_pages([false_hit], self.APOLLO_QUERY, fetch=fake_fetch,
                              wiki_fallback=lambda q: [self.GOOD_HIT])
        self.assertIn(self.GOOD_HIT["url"], opened)
        self.assertTrue(any("1969" in p["page_text"] for p in rows if p["page_read"]))

    def test_web_search_calls_existing_wikipedia_fallback_with_bad_bing_hits(self):
        from neral_web_search import WebSearch
        class FakeResponse:
            def __init__(self, content_type):
                self.headers = {"Content-Type": content_type}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, n): return b"<html>no hits</html>"
        class FakeOpener:
            def open(self, req, timeout):
                mime = ("application/rss+xml"
                        if "bing.com" in req.full_url else "text/html")
                return FakeResponse(mime)
        lookup = []
        def wiki(self_, query, on_provider=None, language="de"):
            lookup.append((query, language))
            return ([self.GOOD_HIT] if language == "de" else [
                {"title": "Apollo 11 - Moon landing NASA",
                 "url": "https://en.wikipedia.org/wiki/Apollo_11",
                 "snippet": "NASA mission Apollo 11 reached the Moon"}])
        with patch("neral_web_search.parse_search_results", return_value=[]), \
             patch("neral_web_search.parse_bing_rss", return_value=self.BAD_HITS), \
             patch.object(WebSearch, "search_wikipedia", new=wiki):
            results = WebSearch(opener=FakeOpener()).search(self.APOLLO_QUERY)
        self.assertTrue(lookup)
        self.assertIn("Apollo 11", lookup[0][0])
        self.assertTrue(any("wikipedia.org" in hit["url"] for hit in results["results"][:3]))
        self.assertFalse(any(hit["url"].startswith("https://www.apollo.")
                             for hit in results["results"][:3]))

    def test_no_sneaking_in_wrong_mission_or_year(self):
        from neral_page_reader import read_top_pages
        from neral_v42_quality import unique_read_year_source
        wrong = {"title": "Apollo 12 lunar mission",
                 "url": "https://en.wikipedia.org/wiki/Apollo_12",
                 "snippet": "NASA Apollo 12 landed on the Moon"}
        def fake_fetch(url, query):
            return {"url": url, "text": "Apollo 12 landed on the Moon in 1969.",
                    "year_clip_code": "year_preserved_in_extract"}
        rows = read_top_pages(self.BAD_HITS+[wrong], self.APOLLO_QUERY,
                              fetch=fake_fetch, wiki_fallback=lambda q: [])
        self.assertIsNone(unique_read_year_source(YEAR, rows))


if __name__=="__main__":
    unittest.main()
