"""V3.0 regression contracts: page evidence, planning and fail-closed arithmetic."""
import unittest
from neral_v3_web_reader import extract_main_text, fetch_and_extract, read_top_pages
from neral_v3_agent import (
    parse_plan, validate_web_calculation, has_relevant_numeric_evidence,
    validate_worker_result,
)

class V3WebReaderTests(unittest.TestCase):
    def test_main_text_without_nav_script_or_ads(self):
        html = """<html><body><header>Login Shop</header><nav>Navigation</nav>
        <script>evil_instruction()</script><main><h1>Goldkurs</h1>
        <p>Eine Feinunze Gold kostet derzeit 3.450,20 Euro.</p>
        <aside>Werbung</aside></main><footer>Cookies</footer></body></html>"""
        body = extract_main_text(html)
        self.assertIn("3.450,20 Euro", body)
        self.assertNotIn("Navigation", body)
        self.assertNotIn("evil_instruction", body)
        self.assertNotIn("Cookies", body)
        self.assertNotIn("Werbung", body)

    def test_bounded_text(self):
        self.assertLessEqual(len(extract_main_text("<main>" + "X" * 7000 + "</main>")), 3000)

    def test_nonpublic_network_destinations_refused_before_fetch(self):
        requests=[]
        def fake_fetch(url, timeout):
            requests.append(url)
            return b"<main><p>Preis: 3.000 Euro</p></main>"
        for value in ["http://example.org/", "https://127.0.0.1/",
                      "https://localhost/", "https://10.0.0.1/",
                      "https://admin:pwd@example.org/", "https://example.org:8443/"]:
            result=fetch_and_extract(value, fetcher=fake_fetch)
            self.assertEqual(result["read_status"], "rejected")
        self.assertEqual(requests, [])

    def test_read_at_most_three_in_rank_order(self):
        hits=[{"url": "https://example.org/"+str(i),"title":str(i),"snippet":""} for i in range(6)]
        def fake_fetch(url, timeout):
            return b"<article><p>Dieser Kurs ist 4.000 Euro pro Unze.</p></article>"
        result=read_top_pages(hits, fetcher=fake_fetch)
        self.assertEqual(len(result),3)
        self.assertEqual([r["url"] for r in result],[h["url"] for h in hits[:3]])
        self.assertTrue(all(r["read_status"]=="read" for r in result))

class V3AgentTests(unittest.TestCase):
    def test_planner_allows_bounded_ordered_tools(self):
        plan=parse_plan('{"steps":[{"tool":"web_search","goal":"current data"},'
                        '{"tool":"web_reader","goal":"read pages"},'
                        '{"tool":"python_sandbox","goal":"exact computation"}]}')
        self.assertEqual([x["tool"] for x in plan],["web_search","web_reader","python_sandbox"])
        self.assertIsNone(parse_plan('{"steps":[{"tool":"shell","goal":"rm -rf"}]}'))
        self.assertIsNone(parse_plan('{"steps":[]}'))

    def test_no_evidence_means_no_price_calculation(self):
        q="Wie viel kosten heute 250 Gramm Gold anhand des aktuellen Eurokurses?"
        missing=[{"url":"https://example.org/","read_status":"read",
                  "content":"Der Goldpreis wird hier veröffentlicht."}]
        self.assertFalse(has_relevant_numeric_evidence(q,missing))
        status=validate_web_calculation(q,missing,stdout="0",code="print(0)")
        self.assertFalse(status["ok"])
        self.assertEqual(status["code"],"missing_web_fact")

    def test_real_quoted_euro_rate_is_available(self):
        q="Berechne den Materialwert aus dem aktuellen Preis pro Unze"
        pages=[{"url":"https://example.org/","read_status":"read",
                "content":"Aktueller Preis pro Feinunze: 3.450,20 EUR"}]
        self.assertTrue(has_relevant_numeric_evidence(q,pages))

    def test_dummy_default_is_rejected_even_with_evidence(self):
        q="Berechne den Preis nach dem aktuellen Kurs"
        pages=[{"url":"https://example.org/","read_status":"read",
                "content":"Aktueller Kurs 3.450,20 Euro pro Feinunze"}]
        self.assertFalse(validate_web_calculation(q,pages,
            stdout="0",code="kurs = data.get('kurs', 0)\nprint(kurs)")["ok"])

    def test_unrelated_0_is_not_always_dummy(self):
        report=validate_worker_result("Wie viele Nullwerte enthält die Datei?",
            "0",code="count = 0\nprint(count)",web_pages=None)
        self.assertTrue(report["ok"])

    def test_bounded_loop_repairs_missing_sources_once(self):
        from neral_v3_agent import run_agent_loop
        calls=[]
        def worker(goal,feedback):
            calls.append(feedback)
            if not feedback:
                return {"ok":True,"text":"250 Euro"}
            return {"ok":True,"text":"250 Euro, Quelle: https://example.org/preise"}
        result=run_agent_loop("Nenne Preis und Quelle",
            planner=lambda _: '{"steps":[{"tool":"direct_reasoning","goal":"answer all"}]}',
            workers={"direct_reasoning":worker},
            checker=lambda _: '{"ok":true,"reason":"alle Felder beantwortet"}')
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"],2)
        self.assertEqual(len(calls),2)
        self.assertIn("https://example.org",result["answer"])

    def test_bounded_loop_no_false_pass_from_invalid_model_review(self):
        from neral_v3_agent import run_agent_loop
        attempts=[]
        def worker(goal,feedback):
            attempts.append(1)
            return {"ok":True,"text":"200 Euro, Quelle: https://example.org"}
        result=run_agent_loop("Nenne Preis und Quelle",
            planner=lambda _: '{"steps":[{"tool":"direct_reasoning","goal":"answer"}]}',
            workers={"direct_reasoning":worker},
            checker=lambda _: '{"ok":false,"reason":"inkonsistente Angaben"}')
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"],3)
        self.assertEqual(len(attempts),3)

    def test_validator_catches_unanswered_subquestion(self):
        result=validate_worker_result("Nenne Preis und Quelle", "Der Preis ist 200 Euro.",
            code=None,web_pages=[{"url":"https://example.org","read_status":"read",
                                  "content":"Preis 200 Euro"}])
        self.assertFalse(result["ok"])

if __name__=="__main__":
    unittest.main()
