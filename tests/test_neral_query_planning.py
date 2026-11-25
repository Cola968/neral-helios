"""Query variants preserve entities and improve multilingual discovery."""
import unittest
from datetime import datetime,timezone
from unittest.mock import MagicMock,patch
from neral_web_search import news_query_variants,WebSearch
from neral_v43_live_web import search_live_sources

class QueryPlanning(unittest.TestCase):
    def test_news_variants_preserve_mission(self):
        variants=news_query_variants("Welche aktuellen Nachrichten gibt es zur NASA Artemis Mission?")
        self.assertTrue(all("nasa" in q.casefold() and "artemis" in q.casefold() for q in variants))
        self.assertLessEqual(len(variants),3)
    def test_news_variants_normalize_inflected_subject(self):
        variants=news_query_variants("Welche aktuellen Nachrichten gibt es zu künstlicher Intelligenz?")
        self.assertIn("künstliche intelligenz", [q.casefold() for q in variants])
        self.assertIn("artificial intelligence", [q.casefold() for q in variants])
    def test_later_query_can_return_relevant_headlines(self):
        engine=WebSearch(opener=MagicMock())
        stamp=datetime.now(timezone.utc).isoformat()
        hit={"url":"https://news.google.com/rss/articles/example","title":"Artificial intelligence improves research","published_at":stamp}
        def google(query,on_provider=None,**kwargs):
            return [hit] if "artificial intelligence" in query.casefold() else []
        with patch.object(engine,"search_bing_news",return_value=[]),patch.object(engine,"search_news",side_effect=google):
            result=search_live_sources(engine,"Welche aktuellen Nachrichten gibt es zu künstlicher Intelligenz?")
        self.assertEqual(result["status"],"headlines_only")
        self.assertEqual(result["sources"][0]["title"],hit["title"])
    def test_english_science_news_starts_with_english_feeds(self):
        engine = WebSearch(opener=MagicMock())
        q = "What is the latest news about the James Webb Space Telescope?"
        with patch.object(engine, "search_bing_news", return_value=[]) as bing, patch.object(engine, "search_news", return_value=[]) as google:
            result = engine.search(q, news_only=True)
        self.assertEqual(result["results"], [])
        for method in (bing, google):
            first = method.call_args_list[0]
            self.assertEqual(first.kwargs.get("language"), "en")
            self.assertIn("space telescope", first.args[0].casefold())

    def test_headline_count_does_not_block_search_for_direct_sources(self):
        engine = WebSearch(opener=MagicMock())
        q = "Welche aktuellen Nachrichten gibt es zum Deutschen Bundestag?"
        stamp = datetime.now(timezone.utc).isoformat()
        headlines = [{"url": f"https://news.google.com/rss/articles/opaque{i}",
                      "title": f"Bundestag debates draft {i}", "published_at": stamp}
                     for i in range(3)]
        direct = {"url": "https://example.org/politics/bundestag-debate",
                  "title": "German parliament debates budget", "published_at": stamp}
        def google(query, on_provider=None, **kwargs):
            return [direct] if "german parliament" in query.casefold() else headlines
        with patch.object(engine, "search_bing_news", return_value=[]), patch.object(engine, "search_news", side_effect=google):
            result = engine.search(q, news_only=True)
        self.assertIn(direct["url"], [hit["url"] for hit in result["results"]])

    def test_generic_weather_intent_retains_place_and_day(self):
        from neral_web_search import live_query_variants
        variants=live_query_variants("Wie ist das Wetter morgen in Berlin?")
        self.assertTrue(all("berlin" in q.casefold() for q in variants))
        self.assertTrue(any("morgen" in q.casefold() or "tomorrow" in q.casefold() for q in variants))
    def test_alternative_queries_cannot_invent_topics(self):
        variants=news_query_variants("Aktuelle Nachrichten über Vulkan Island")
        self.assertTrue(all("vulkan" in q.casefold() and "island" in q.casefold() for q in variants))

if __name__=="__main__":
    unittest.main()
