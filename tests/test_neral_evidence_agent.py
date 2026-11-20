import json
import unittest
from unittest.mock import Mock, patch

from neral_evidence_agent import run_evidence_agent
from neral_quantity_plans import quantity_plan
from neral_page_reader import MainText, _public_address, read_top_pages


class EvidenceAgentTest(unittest.TestCase):
    def setUp(self):
        self.question = "250 Gramm zu 31,1035 Gramm je Feinunze. Was kostet das Gold in Euro?"
        self.plan = {"tool": "python_sandbox", "parts": ["Materialwert"]}
        self.worker = {"locale": "de", "calculations": [
            {"label": "Materialwert", "expr": "u1 / u2 * w1_1", "unit": "EUR"}], "missing": []}
        self.pages = Mock(return_value=[{"url": "https://gold.example.com/price", "page_read": True,
                                        "page_text": "Gold: 3.367,35 EUR pro Feinunze"}])
        self.search = Mock()
        self.search.search.return_value = {"results": [{"url": "https://gold.example.com/price"}]}
        self.execute = Mock(return_value={"status": "PASSED", "output": "27065.6839262456"})

    def run_agent(self, packets):
        model = Mock(side_effect=[json.dumps(packet) for packet in packets])
        return run_evidence_agent(self.question, model, self.execute, web_query="Goldpreis",
                                  search=self.search, page_reader=self.pages)

    def test_read_then_compute_with_exact_quote(self):
        result = self.run_agent([self.plan, self.worker, {"ok": True, "issue": ""}])
        self.assertTrue(result["agent_checked"])
        self.assertFalse(result["verified"])
        self.assertIn("27065,68 EUR", result["text"])
        self.assertIn("Decimal('3367.35')", self.execute.call_args.args[0])
        self.search.search.assert_called_once_with("Goldpreis")

    def test_missing_price_never_reaches_sandbox(self):
        self.worker["missing"] = ["Goldpreis EUR/Feinunze"]
        self.worker["calculations"] = []
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertEqual(result["agent_attempts"], 3)
        self.assertIn("Goldpreis EUR/Feinunze", result["text"])
        self.execute.assert_not_called()

    def test_invented_literal_never_reaches_sandbox(self):
        self.worker["calculations"][0]["expr"] = "u1 / u2 * 0"
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertFalse(result["agent_checked"])
        self.execute.assert_not_called()

    def test_web_calculation_cannot_use_only_user_numbers(self):
        self.worker["calculations"][0]["expr"] = "u1 / u2"
        result = self.run_agent([self.plan, self.worker, self.worker, self.worker])
        self.assertIn("externen Quelle", result["text"])
        self.execute.assert_not_called()

    def test_new_standalone_request_ignores_prior_turns_in_every_agent_stage(self):
        prior = [
            {"role": "user", "content": "ALT_AUFGABE Goldkurs 3.500 Euro, Ben darf nie Dienstag"},
            {"role": "assistant", "content": "ALT_ANTWORT Gold 28.000 Euro, Ben ausgeschlossen"},
            {"role": "user", "content": "ALT_Z3 Nur Ada Montag, Cem niemals Mittwoch"},
        ]
        fresh = ("Ein neues Team besteht aus Ada, Ben und Cem. Verteile die "
                 "drei Personen auf Montag, Dienstag und Mittwoch. "
                 "Ada muss am Montag arbeiten und Ben darf nicht Dienstag.")
        packets = [
            {"tool": "z3_solver", "parts": ["Drei Personen auf drei Tage zuordnen"]},
            {"schema": {"variables": [
                {"name": "ada", "min": 1, "max": 3},
                {"name": "ben", "min": 1, "max": 3},
                {"name": "cem", "min": 1, "max": 3},
            ], "constraints": [
                {"rule_id": "R1", "description": "Alle arbeiten an verschiedenen Tagen",
                 "expr": "And(ada != ben, ada != cem, ben != cem)"},
                {"rule_id": "R2", "description": "Ada Montag", "expr": "ada == 1"},
                {"rule_id": "R3", "description": "Ben nicht Dienstag", "expr": "ben != 2"},
            ], "objective": "find_one"},
             "legend": "Montag=1 Dienstag=2 Mittwoch=3"},
            {"ok": True, "issue": ""},
        ]
        calls = []
        def respond(system, raw, budget, schema):
            calls.append((system, json.loads(raw)))
            return json.dumps(packets.pop(0))
        with patch("neral_evidence_agent.compile_z3_json",
                   return_value="print('1 3 2')"):
            result = run_evidence_agent(
                fresh, respond,
                Mock(return_value={"status": "PASSED", "output": "1 3 2"}),
                history=prior)
        self.assertTrue(result["agent_checked"])
        self.assertIn("Z3:", result["text"])
        self.assertEqual(len(calls), 3)
        for system, payload in calls:
            self.assertNotIn("history", payload)
            self.assertNotIn("ALT_", json.dumps(payload, ensure_ascii=False))
            self.assertEqual(payload["question"], fresh)

    def test_independent_web_comparison_does_not_reuse_old_z3_rules(self):
        prior = [{"role": "user", "content": "ALT_Z3: Ada Freitag und Ben nie Montag"},
                 {"role": "assistant", "content": "ALT_ECHO: Zwei Schichtgruppen"}]
        model_calls = []
        packets = [
            {"tool": "direct_reasoning", "parts": ["Firefox Merkmale vergleichen"]},
            {"answer": "Firefox verwendet die Gecko-Engine.",
             "citations": ["p1"]},
            {"ok": True, "issue": ""},
        ]
        def respond(system, raw, budget, schema):
            model_calls.append(json.loads(raw))
            return json.dumps(packets.pop(0))
        web = Mock(return_value=[{
            "url": "https://www.mozilla.org/en-US/firefox/",
            "page_read": True,
            "page_text": "Firefox uses the Gecko rendering engine for pages."
        }])
        search = Mock()
        search.search.return_value={"results": [{"url":"https://www.mozilla.org/en-US/firefox/"}]}
        result = run_evidence_agent("Womit rendert Firefox Webseiten?",respond,
                                    Mock(),history=prior,web_query="Firefox rendering engine",
                                    search=search,page_reader=web)
        self.assertTrue(result["agent_checked"])
        self.assertEqual(len(model_calls),3)
        for data in model_calls:
            self.assertNotIn("history",data)
            self.assertNotIn("ALT_",json.dumps(data,ensure_ascii=False))

    def test_checker_rejects_and_repairs(self):
        result = self.run_agent([self.plan, self.worker, {"ok": False, "issue": "Einheit erklären"},
                                 self.plan, self.worker, {"ok": True, "issue": ""}])
        self.assertTrue(result["agent_checked"])
        self.assertEqual(result["agent_attempts"], 2)
        self.search.search.assert_called_once()

    def test_comparison_citations_are_copied_from_source_passages(self):
        self.pages.return_value[0]["page_text"] = (
            "Gerät A nimmt den Ton für spätere Wiedergabe auf.\n"
            "Gerät B spielt gespeicherte Töne ab.")
        plan = {"tool": "direct_reasoning", "parts": ["Funktionen vergleichen"]}
        worker = {"answer": "A nimmt auf und B spielt ab.", "citations": ["p1", "p2"]}
        result = self.run_agent([plan, worker, {"ok": True, "issue": ""}])
        self.assertIn("[web1] „Gerät A nimmt den Ton für spätere Wiedergabe auf.“", result["text"])
        self.assertIn("[web1] „Gerät B spielt gespeicherte Töne ab.“", result["text"])
        self.execute.assert_not_called()


    def test_qualitative_web_comparison_checks_both_sides_not_literal_comparison(self):
        from neral_evidence_agent import _qualitative_comparison_issue
        question = ("Suche Webseiten und vergleiche Mozilla Firefox mit Google Chrome: "
                    "Welche Engine und welche Tracking-Eigenschaften haben sie?")
        evidence = {
            "user": question,
            "web1": "Mozilla Firefox uses the Gecko browser engine. Firefox blocks tracking.",
            "web2": "Google Chrome uses the Blink browser engine. Chrome has privacy controls.",
        }
        citations = [
            {"source": "web1", "quote": "Mozilla Firefox uses the Gecko browser engine."},
            {"source": "web2", "quote": "Google Chrome uses the Blink browser engine."},
        ]
        sources = [
            {"page_read": True, "url": "https://mozilla.example.org/engine"},
            {"page_read": True, "url": "https://chrome.example.org/engine"},
        ]
        answer = ("Firefox verwendet Gecko und schützt vor Tracking. "
                  "Chrome verwendet Blink und bietet Datenschutzeinstellungen. "
                  "Das ist eine Gegenüberstellung zweier getrennt belegter Merkmale.")
        self.assertIsNone(_qualitative_comparison_issue(
            question, answer, citations, evidence, sources))

    def test_qualitative_web_comparison_rejects_missing_web_read(self):
        from neral_evidence_agent import _qualitative_comparison_issue
        question = "Vergleiche Firefox mit Chrome: Engine und Tracking."
        evidence = {"user": question, "web1": "Firefox Gecko", "web2": "Chrome Blink"}
        citations = [{"source": "web1", "quote": "Firefox Gecko"},
                     {"source": "web2", "quote": "Chrome Blink"}]
        sources = [{"page_read": False}, {"page_read": False}]
        self.assertIn("Webseite", _qualitative_comparison_issue(
            question, "Firefox Gecko, Chrome Blink.", citations, evidence, sources))

    def test_qualitative_web_comparison_rejects_one_sided_answer(self):
        from neral_evidence_agent import _qualitative_comparison_issue
        question = "Vergleiche Firefox mit Chrome: Engine."
        evidence = {"user": question, "web1": "Firefox Gecko", "web2": "Chrome Blink"}
        citations = [{"source": "web1", "quote": "Firefox Gecko"},
                     {"source": "web2", "quote": "Chrome Blink"}]
        sources = [{"page_read": True}, {"page_read": True}]
        self.assertIn("Chrome", _qualitative_comparison_issue(
            question, "Firefox verwendet Gecko.", citations, evidence, sources))

    def test_qualitative_web_comparison_rejects_invented_number(self):
        from neral_evidence_agent import _qualitative_comparison_issue
        question = "Vergleiche Firefox mit Chrome: Tracking."
        evidence = {"user": question, "web1": "Firefox blocks tracking.",
                    "web2": "Chrome has privacy settings."}
        citations = [{"source": "web1", "quote": "Firefox blocks tracking."},
                     {"source": "web2", "quote": "Chrome has privacy settings."}]
        sources = [{"page_read": True}, {"page_read": True}]
        self.assertIn("Zahl", _qualitative_comparison_issue(
            question, "Firefox blockiert 98 % aller Tracker, Chrome bietet Einstellungen.",
            citations, evidence, sources))

    def test_qualitative_comparison_has_bounded_contradiction_review(self):
        question = ("Suche Webseiten und vergleiche Mozilla Firefox mit Google Chrome: "
                    "Welche Engine verwenden sie?")
        web = Mock(return_value=[
            {"url": "https://mozilla.example.org/", "page_read": True,
             "page_text": "Mozilla Firefox uses the Gecko engine."},
            {"url": "https://chrome.example.org/", "page_read": True,
             "page_text": "Google Chrome uses the Blink engine."},
        ])
        search = Mock()
        search.search.return_value = {"results": [{"url": "https://mozilla.example.org/"},
                                                   {"url": "https://chrome.example.org/"}]}
        packets = [
            {"tool": "direct_reasoning", "parts": ["Beide Engines gegenüberstellen"]},
            {"left": "verwendet Gecko.", "right": "verwendet Blink.",
             "synthesis": "Beide verwenden unterschiedliche Rendering-Engines.",
             "citations": ["p1", "p2"]},
            {"contradiction": False, "claim": "", "source_quote": ""},
        ]
        calls = []
        def respond(system, raw, budget, schema):
            calls.append((system, json.loads(raw)))
            return json.dumps(packets.pop(0))
        result = run_evidence_agent(question, respond, Mock(), web_query="Firefox Chrome engines",
                                    search=search, page_reader=web)
        self.assertTrue(result["agent_checked"])
        self.assertEqual(result["agent_attempts"], 1)
        self.assertIn("Firefox: verwendet Gecko.", result["text"])
        self.assertEqual(result["tool_used"], "direct_reasoning")
        self.assertIn("[web1]", result["text"])
        self.assertIn("[web2]", result["text"])
        self.assertEqual(len(calls), 3)
        self.assertNotIn("history", calls[-1][1])

    def test_qualitative_comparison_rejects_cited_contradiction(self):
        question = "Vergleiche Firefox mit Chrome: Engine."
        web = Mock(return_value=[
            {"url": "https://mozilla.example.org/", "page_read": True,
             "page_text": "Firefox uses Gecko engine."},
            {"url": "https://chrome.example.org/", "page_read": True,
             "page_text": "Chrome uses Blink engine."},
        ])
        search = Mock()
        search.search.return_value={"results":[{"url":"https://mozilla.example.org/"},
                                                 {"url":"https://chrome.example.org/"}]}
        packets = [
            {"tool": "direct_reasoning", "parts": ["Engines"]},
            {"left": "verwendet Blink.", "right": "verwendet Gecko.",
             "synthesis": "Die beiden Engines unterscheiden sich.",
             "citations": ["p1", "p2"]},
            {"contradiction": True, "claim": "Firefox: verwendet Blink.",
             "source_quote": "Firefox uses Gecko engine."},
            {"tool": "direct_reasoning", "parts": ["Engines"]},
            {"left": "verwendet Gecko.", "right": "verwendet Blink.",
             "synthesis": "Beide verwenden unterschiedliche Rendering-Engines.",
             "citations": ["p1", "p2"]},
            {"contradiction": False, "claim": "", "source_quote": ""},
        ]
        def respond(system, raw, budget, schema):
            return json.dumps(packets.pop(0))
        result = run_evidence_agent(question, respond, Mock(), web_query="Firefox Chrome engines",
                                    search=search, page_reader=web)
        self.assertTrue(result["agent_checked"])
        self.assertEqual(result["agent_attempts"], 2)
        self.assertIn("Firefox: verwendet Gecko.", result["text"])


    def test_comparison_adds_literal_quotes_for_both_products(self):
        question = "Vergleiche Mozilla Firefox mit Google Chrome: Welche Engine?"
        web = Mock(return_value=[
            {"url": "https://mozilla.example.org/", "page_read": True,
             "page_text": "Mozilla Firefox uses Gecko for rendering."},
            {"url": "https://chrome.example.org/", "page_read": True,
             "page_text": "Google Chrome uses Blink for rendering."},
        ])
        search = Mock()
        search.search.return_value = {"results": [{"url": "https://mozilla.example.org/"},
                                                   {"url": "https://chrome.example.org/"}]}
        packets = [
            {"tool": "direct_reasoning", "parts": ["Vergleich beider Engines"]},
            {"left": "nutzt Gecko.", "right": "nutzt Blink.",
             "synthesis": "Die Rendering-Engines unterscheiden sich.",
             "citations": ["p2"]},
            {"contradiction": False, "claim": "", "source_quote": ""},
        ]
        def respond(system, raw, budget, schema):
            return json.dumps(packets.pop(0))
        result = run_evidence_agent(question, respond, Mock(),
                                    web_query="Firefox Chrome engines",
                                    search=search, page_reader=web)
        self.assertTrue(result["agent_checked"])
        self.assertIn("Firefox: nutzt Gecko.", result["text"])
        self.assertIn("[web1] „Mozilla Firefox uses Gecko for rendering.“", result["text"])
        self.assertIn("[web2] „Google Chrome uses Blink for rendering.“", result["text"])
        self.assertEqual(result["agent_attempts"], 1)


    def test_detects_comparison_forms_and_three_entities(self):
        from neral_evidence_agent import _comparison_entities
        cases = (
            ("Firefox vs Chrome", ("Firefox", "Chrome")),
            ("Vergleich Firefox und Chrome", ("Firefox", "Chrome")),
            ("Unterschied zwischen Firefox und Chrome?", ("Firefox", "Chrome")),
            ("Vergleiche Mozilla Firefox mit Google Chrome: Datenschutz",
             ("Firefox", "Chrome")),
            ("Vergleich Firefox, Chrome und Edge: Browser-Engines",
             ("Firefox", "Chrome", "Edge")),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(_comparison_entities(question), expected)

    def test_entity_queries_focus_on_subject_and_user_topic(self):
        from neral_evidence_agent import _entity_search_query
        user_query = ("Suche aktuelle Webseiten und vergleiche Mozilla Firefox "
                      "mit Google Chrome: Welche Browser-Engine verwenden sie und "
                      "welcher Datenschutz und Tracking-Schutz wird beschrieben?")
        firefox = _entity_search_query(user_query, "Firefox", ("Firefox", "Chrome"), 0)
        chrome = _entity_search_query(user_query, "Chrome", ("Firefox", "Chrome"), 0)
        self.assertIn("Firefox", firefox)
        self.assertIn("Chrome", chrome)
        self.assertNotIn("Chrome", firefox)
        self.assertNotIn("Firefox", chrome)
        self.assertIn("Tracking", firefox)
        self.assertLessEqual(len(firefox), 150)

    def test_three_way_comparison_has_grounded_sources_for_every_subject(self):
        question = "Vergleich Firefox, Chrome und Edge: Browser-Engine"
        facts = {
            "Firefox": "Mozilla Firefox verwendet die Gecko Browser Engine.",
            "Chrome": "Google Chrome verwendet die Blink Browser Engine.",
            "Edge": "Microsoft Edge verwendet die Blink Browser Engine.",
        }
        class Search:
            def search(self, query):
                subject = next(name for name in facts if name in query)
                return {"results": [{"url": f"https://{subject.lower()}.example.org/"}]}
        def reader(hits, query):
            url = hits[0]["url"]
            subject = next(name for name in facts if name.lower() in url)
            return [{"url": url, "page_read": True, "page_text": facts[subject]}]
        calls=[]
        def once(sys,raw,budget,schema):
            calls.append(raw)
            return json.dumps({"facts":[
                {"entity":"Firefox","fact":"Gecko"},
                {"entity":"Chrome","fact":"Blink"},
                {"entity":"Edge","fact":"Blink"},
            ]})
        result = run_evidence_agent(question,once,Mock(),
                                    web_query="Firefox Chrome Edge Browser Engine",
                                    search=Search(), page_reader=reader)
        self.assertTrue(result["agent_checked"])
        self.assertEqual(len(calls),1)
        self.assertEqual(result["agent_attempts"], 1)
        self.assertEqual(len(result["sources"]), 3)
        self.assertIn("Edge: Blink", result["text"])
        self.assertIn("[web3]", result["text"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["tool_used"], "direct_reasoning")

    def test_missing_entity_pages_are_researched_in_second_attempt(self):
        question = "Vergleiche Firefox mit Chrome: Engine."
        sites = {
            "https://not-firefox.example.com/": "Ein Browser hat eine Rendering Engine namens Blink.",
            "https://chrome.example.com/": "Google Chrome verwendet die Blink Rendering Engine.",
            "https://mozilla.example.com/": "Mozilla Firefox verwendet die Gecko Rendering Engine.",
        }
        class Search:
            def __init__(self):
                self.queries = []
            def search(self, query):
                self.queries.append(query)
                if "Firefox" in query:
                    page = ("https://not-firefox.example.com/"
                            if sum("Firefox" in q for q in self.queries) == 1
                            else "https://mozilla.example.com/")
                elif "Chrome" in query:
                    page = "https://chrome.example.com/"
                else:
                    raise AssertionError("Search did not separate entities: " + query)
                return {"results": [{"url": page}]}
        search = Search()
        def reader(hits, query):
            return [{"url": h["url"], "page_read": True,
                     "page_text": sites[h["url"]]} for h in hits]
        replies = [
            {"tool": "direct_reasoning", "parts": ["Engine vergleichen"]},
            {"tool": "direct_reasoning", "parts": ["Engine vergleichen"]},
            {"left": "verwendet Gecko.", "right": "verwendet Blink.",
             "synthesis": "Unterschiedliche Rendering Engines.",
             "citations": ["p1", "p2"]},
            {"contradiction": False, "claim": "", "source_quote": ""},
        ]
        def respond(system, raw, budget, schema):
            return json.dumps(replies.pop(0))
        result = run_evidence_agent(question, respond, Mock(),
                                    web_query="Firefox Chrome rendering engine",
                                    search=search, page_reader=reader)
        self.assertTrue(result["agent_checked"])
        self.assertEqual(result["agent_attempts"], 2)
        self.assertGreaterEqual(len(search.queries), 3)
        self.assertTrue(any("Firefox" in q and "Chrome" not in q
                            for q in search.queries))
        self.assertTrue(any("Chrome" in q and "Firefox" not in q
                            for q in search.queries))
        self.assertIn("Firefox", result["text"])
        self.assertIn("Chrome", result["text"])
        self.assertTrue(any("mozilla.example.com" in s["url"]
                            for s in result["sources"]))

    def test_unavailable_entity_never_gets_fictitious_citation(self):
        question = "Vergleich Firefox und Chrome: Engine."
        search = Mock()
        search.search.return_value = {"results":[{"url":"https://chrome.example.com/"}]}
        reader = Mock(return_value=[
            {"url":"https://chrome.example.com/", "page_read":True,
             "page_text":"Chrome uses Blink browser rendering engine."}])
        packets = [{"tool":"direct_reasoning","parts":["Vergleich"]}] * 3
        def respond(system, raw, budget, schema):
            return json.dumps(packets.pop(0))
        result = run_evidence_agent(question, respond, Mock(),
                                    web_query="Firefox Chrome rendering engine",
                                    search=search, page_reader=reader)
        self.assertFalse(result["agent_checked"])
        self.assertIn("Firefox", result["text"])
        self.assertEqual(result["agent_attempts"], 3)
        self.assertFalse(any("Firefox" in s.get("page_text","")
                             for s in result["sources"]))


class PageReaderTest(unittest.TestCase):
    def test_extracts_text_without_navigation_or_scripts(self):
        p = MainText()
        p.feed('<body><nav>Menü</nav><main><script>code()</script>'
               '<table><tr><td>1 Feinunze</td><td>3.367,35 EUR</td></tr></table>'
               '<p hidden>versteckt</p><p>Messung</p></main><footer>Links</footer>')
        self.assertEqual(p.text("Gold EUR"), "| 1 Feinunze | 3.367,35 EUR\nMessung")

    def test_uses_only_three_public_unique_results(self):
        urls = ["https://127.0.0.1/", "https://a.example.com/", "https://a.example.com/",
                "https://b.example.com/", "https://c.example.com/", "https://d.example.com/"]
        fetch = Mock(side_effect=lambda url, _: {"url": url, "text": "x" * 5100})
        pages = read_top_pages([{"url": url} for url in urls], "topic", fetch=fetch)
        self.assertEqual([page["url"] for page in pages], urls[1:2] + urls[3:5])
        from neral_page_reader import MAX_TEXT
        self.assertTrue(all(len(page["page_text"]) == MAX_TEXT for page in pages))

    def test_entity_page_reader_skips_irrelevant_ranked_hits(self):
        from neral_page_reader import read_entity_pages
        hits = [{"url": f"https://source{i}.example.com/"} for i in range(1, 6)]
        def fake_reader(batch, query):
            output = []
            for hit in batch:
                url = hit["url"]
                content = ("Mozilla Firefox verwendet Gecko als Rendering Engine."
                           if "source4" in url else
                           "Google Chrome verwendet Blink als Rendering Engine.")
                output.append({"url": url, "page_read": True,
                               "page_text": content})
            return output
        pages = read_entity_pages(hits, "Firefox engine", "Firefox",
                                  reader=fake_reader)
        self.assertEqual(len(pages), 1)
        self.assertIn("source4", pages[0]["url"])

    def test_entity_coverage_requires_read_page_with_facts_not_search_preview(self):
        from neral_page_reader import entity_evidence_excerpt
        self.assertIsNone(entity_evidence_excerpt("Firefox", "Firefox"))
        self.assertIsNone(entity_evidence_excerpt(
            "Chrome uses Blink browser engine.", "Firefox"))
        self.assertIsNotNone(entity_evidence_excerpt(
            "Firefox uses Gecko rendering engine for pages.", "Firefox"))

    def test_rejects_nonpublic_dns_answers(self):
        addresses = [(0, 0, 0, "", ("93.184.215.14", 443)), (0, 0, 0, "", ("127.0.0.1", 443))]
        with patch("neral_page_reader.socket.getaddrinfo", return_value=addresses):
            with self.assertRaises(ValueError):
                _public_address("mixed.example.com")


class ExactQuantityTest(unittest.TestCase):
    def test_rate_uses_same_metal_currency_and_quote_basis(self):
        question = "Was ist der Materialwert eines 125-Gramm-Silberbarrens in Euro (1 Feinunze = 31,1035 Gramm)?"
        evidence = {"user": question, "web1": "| GOLD | 3.360,00 €/oz\n| SILBER | 44,00 $/oz | 40,50 €/oz | 1,30 €/g"}
        plan, numbers = quantity_plan(question, evidence)
        self.assertEqual(numbers["price"]["token"], "40,50")
        self.assertEqual(plan["calculations"][1]["expr"], "mass / grams_per_ounce * price")
        self.assertIsNone(quantity_plan(question, {"user": question, "web1": "SILBER 44,00 $/oz"}))

    def test_inventory_requires_the_complete_source(self):
        question = ("Jan stellt 4 Kisten mit je 10 Flaschen in den Raum. 3 Flaschen zerbrechen. "
                    "In jedes übrige Glas füllt er 0,25 Liter Saft. Der Behälter enthält vorher 20 Liter. "
                    "Was bleibt im Behälter? Wie viele gefüllte Flaschen stehen bereit?")
        _, numbers = quantity_plan(question, {"user": question})
        self.assertEqual(numbers["lost"]["token"], "3")
        self.assertIsNone(quantity_plan(question + " Weitere 2 bleiben leer.", {"user": question}))


if __name__ == "__main__":
    unittest.main()
