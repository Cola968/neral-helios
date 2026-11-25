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
