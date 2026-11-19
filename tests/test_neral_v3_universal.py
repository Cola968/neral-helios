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

class V3HttpIntegrationTests(unittest.TestCase):
    def test_stream_returns_page_read_evidence_and_source_url(self):
        import json
        from unittest.mock import patch
        from flask import Flask
        from neral_web_api import register_web_search_api

        class StubProvider:
            def search(self,query,on_provider=None):
                return {"query":query,"provider":"Stub","results":[
                    {"title":"Goldpreis aktuell","url":"https://example.org/gold",
                     "snippet":"Goldpreis und Kurs für Euro"}]}
        app=Flask(__name__)
        register_web_search_api(app,StubProvider())
        with patch("neral_web_api.verify_public_https_url",
                   side_effect=lambda url:url), patch(
                   "neral_web_api.read_top_pages", return_value=[{
                       "url":"https://example.org/gold",
                       "read_status":"read","content":"Feinunze: 3.450,20 EUR"}]):
            response=app.test_client().post("/api/web/search/stream",
                json={"query":"Goldpreis aktuell","verify_links":True,
                      "read_pages":True})
            events=[json.loads(line) for line in response.response]
        self.assertEqual(response.status_code,200)
        self.assertIn("reading",[event.get("state") for event in events])
        result=next(event["result"] for event in events
                    if event["type"]=="result")
        self.assertEqual(result["pages_read"],1)
        # Fresh searches return provider metadata in previews rather than
        # marking the destination as independently reachable by HEAD.
        seen=result["previews"] or result["results"]
        self.assertIn("3.450,20 EUR",seen[0]["content"])
        self.assertEqual(seen[0]["url"],"https://example.org/gold")

    def test_old_search_schema_remains_compatible_with_new_full_page_data(self):
        from neral_v21_attachment_data import prepare_web_dataset
        short={"title":"Gold","snippet":"Preis","url":"https://example.org/gold"}
        old=prepare_web_dataset([short])
        self.assertEqual(old["pages_read"],0)
        rich={**short,"read_status":"read","content":"Eine Unze 3.500 Euro"}
        newest=prepare_web_dataset([rich])
        self.assertEqual(newest["pages_read"],1)
        self.assertIn(b"3.500 Euro",newest["staged_files"]["data/data1.json"])
        with self.assertRaises(ValueError):
            prepare_web_dataset([{**short,"read_status":"unavailable",
                                  "content":"Erfundener Betrag 9999 Euro"}])

    def test_missing_web_rate_returns_diagnostic_without_model_call(self):
        from neral_v3_runtime import answer_v3
        fake=object()
        response=answer_v3(fake, [
            {"role":"system","content":"System"},
            {"role":"user","content":"Wie viel kostet Gold aktuell?"}],
            "Luna",web_stage={"web_pages":[{
                "url":"https://example.org/gold","read_status":"read",
                "content":"Preis folgt später"}]})
        self.assertFalse(response["model_generated"])
        self.assertFalse(response["sandbox_executed"])
        self.assertIn("nicht eindeutig gefunden",response["text"])
        self.assertNotEqual(response["text"].strip(),"0")


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
