"""Live news has an explicit no-model, no-evidence feed-only fallback."""
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, Mock, patch
from neral_web_search import WebSearch
from neral_v43_live_web import search_live_sources
from neral_v43_verified_start import build_verified_app
from neral_three_tier_runtime import LUNA, RAW_9B
from neral_agent_autonomy import OllamaLocalBackend

Q="Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
H={"url":"https://news.google.com/rss/articles/fixture","title":"BER: Flughafen Berlin Brandenburg meldet neue Verbindungen - Testzeitung","published_at":datetime.now(timezone.utc).isoformat(),"provider":"Google News RSS"}

class FeedFallbackTests(unittest.TestCase):
    def test_feed_only_retains_real_link_but_never_article(self):
        engine=WebSearch(opener=MagicMock())
        with patch.object(engine,"search_news",return_value=[H]), patch.object(engine,"search_bing_news",return_value=[]):
            out=search_live_sources(engine,Q,reader=Mock())
        self.assertEqual(out["status"],"headlines_only")
        self.assertEqual(out["context"],"")
        self.assertEqual(out["sources"][0]["url"],H["url"])
        self.assertFalse(out["sources"][0]["page_read"])
        self.assertEqual(out["sources"][0]["source_kind"],"feed_headline")

    def test_stale_unrelated_and_unsafe_headlines_rejected(self):
        engine=Mock()
        engine.search.return_value={"provider":"Google News RSS","results":[],"headlines":[
            {**H,"published_at":(datetime.now(timezone.utc)-timedelta(days=30)).isoformat()},
            {**H,"title":"Flughafen Frankfurt"},
            {**H,"url":"https://127.0.0.1/private"},
        ]}
        self.assertEqual(search_live_sources(engine,Q,reader=Mock())["status"],"empty")

    def test_city_or_publisher_name_alone_is_not_airport_evidence(self):
        engine=Mock()
        engine.search.return_value={"provider":"Google News RSS","results":[],"headlines":[
            {**H,"title":"Totensonntag in München - muenchen.de"},
            {**H,"url":H["url"]+"2","title":"München-Riem: Erinnerungen - Abendzeitung München"},
            {**H,"url":H["url"]+"3","title":"Flughafen München erweitert Angebot - Testzeitung"},
        ]}
        out=search_live_sources(engine,"Was gibt es Neues am Flughafen MUC?",reader=Mock())
        self.assertEqual(out["status"],"headlines_only")
        self.assertEqual(len(out["sources"]),1)
        self.assertTrue(out["sources"][0]["title"].startswith("Flughafen München"))

    def test_opt_in_stages_distinguish_headlines_from_dropped_reads(self):
        candidate = {**H, "url": "https://publisher.example.com/daily/observations",
                     "title": "Coral reefs survey results"}
        headline = {**H, "title": candidate["title"]}
        found = {"provider": "fixture", "headlines": [headline]}
        query = "Latest news about coral reefs"
        tests = (
            ([], [], 0, 0, 0, 0),
            ([candidate], [], 1, 0, 0, 0),
            ([candidate], [{**candidate, "page_read": True,
                            "page_text": "An unrelated private-text sentinel."}], 1, 1, 1, 0),
            ([candidate], [{**candidate, "page_read": True, "page_text": "Coral reefs report.",
                            "date_origin": "publisher_metadata",
                            "published_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()}],
             1, 1, 0, 1),
        )
        for hits, pages, requested, read, rejected, provenance in tests:
            with self.subTest(requested=requested, read=read):
                search = Mock()
                search.search.return_value = {**found, "results": hits}
                reader = Mock(return_value=pages)
                normal = search_live_sources(search, query, reader=reader)
                measured = search_live_sources(search, query, reader=reader, include_diagnostics=True)
                counts = measured.pop("stage_counts")
                self.assertEqual(normal, measured)
                self.assertEqual(measured["status"], "headlines_only")
                self.assertEqual(measured["context"], "")
                self.assertEqual(counts["eligible_candidates"], requested)
                self.assertEqual(counts["reader_candidates"], requested)
                self.assertEqual(counts["reader_read_bodies"], read)
                self.assertEqual(counts["rejected_relevance"], rejected)
                self.assertEqual(counts["rejected_news_provenance"], provenance)
                self.assertEqual(counts["display_only_headlines"], 1)
                self.assertTrue(all(type(value) is int for value in counts.values()))

    def test_publication_metadata_does_not_turn_short_body_into_read_article(self):
        title = "Coral reefs survey results"
        publisher = {**H, "url": "https://publisher.example.com/daily/observations",
                     "title": title, "provider": "fixture"}
        headline = {**H, "title": title}
        engine = Mock()
        engine.search.return_value = {"provider": "fixture", "results": [publisher],
                                      "headlines": [headline]}
        query = "Latest news about coral reefs"
        short_body = "Coral reefs report."
        long_body = ("Researchers surveyed coral reefs and counted juvenile fish "
                     "across several coastal sites. " * 4).strip()

        for body, article_chars, accepted in (
                (short_body, None, False),
                (short_body, 600, False),
                (long_body, 80, False),
                (long_body, "600", False),
                (long_body, len(long_body), True)):
            with self.subTest(article_chars=article_chars, accepted=accepted):
                page = {**publisher, "page_read": True, "page_text": body,
                        "date_origin": "publisher_metadata"}
                if article_chars is not None:
                    page["article_chars"] = article_chars
                result = search_live_sources(engine, query, reader=Mock(return_value=[page]),
                                             include_diagnostics=True)
                self.assertEqual(result["status"], "ok" if accepted else "headlines_only")
                self.assertEqual(result["sources"][0]["page_read"], accepted)
                self.assertEqual(bool(result["context"]), accepted)
                self.assertEqual(result["stage_counts"]["rejected_article_body"], int(not accepted))
                self.assertEqual(result["sources"][0]["url"],
                                 publisher["url"] if accepted else headline["url"])

    def test_actual_flask_route_returns_feed_listing_without_model(self):
        legacy=OllamaLocalBackend("qwen3:8b")
        luna=Mock(model=LUNA,adapter_sha256=None,loaded=True)
        raw=Mock(model=RAW_9B,adapter_sha256=None,loaded=True)
        rt,app=build_verified_app(legacy,luna,raw,execute_python=Mock())
        with patch("neral_web_search.WebSearch.search",return_value={"provider":"Google News RSS","results":[],"headlines":[H]}):
            with app.test_client() as c:
                base="http://127.0.0.1:9690"
                c.get("/",base_url=base)
                csrf=c.get("/api/neral/preview/session",base_url=base).json["csrf"]
                r=c.post("/api/neral/chat/three-tier",base_url=base,headers={"X-Neral-CSRF":csrf},
                         json={"tier":"Quartz","prompt":Q,"history":[],"force_sandbox":False})
        self.assertEqual(r.status_code,200,r.get_json())
        self.assertEqual(r.json["web_search_status"],"headlines_only")
        self.assertFalse(r.json["verified"])
        self.assertFalse(r.json["model_generated"])
        self.assertFalse(r.json["web_sources"][0]["page_read"])
        self.assertIn("Originalartikel konnten nicht gelesen werden",r.json["text"])
        raw.respond_chat.assert_not_called()
        luna.respond_chat.assert_not_called()

if __name__=="__main__":
    unittest.main()
