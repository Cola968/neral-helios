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
