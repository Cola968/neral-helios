"""Query variants preserve entities and improve multilingual discovery."""
import unittest
from datetime import datetime,timezone,timedelta
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

    def test_read_publisher_version_replaces_feed_duplicate_and_survives_cap(self):
        engine = WebSearch()
        q = "Aktuelle Nachrichten über künstliche Intelligenz"
        now = datetime.now(timezone.utc)
        known = {"url": "https://example.org/news/artificial-intelligence",
                 "title": "Artificial intelligence in research", "published_at": now.isoformat()}
        unrelated_reads = [{"url": f"https://example.org/news/artificial-intelligence-{i}",
                            "title": f"Artificial intelligence briefing {i}", "published_at": now.isoformat()}
                           for i in range(10)]
        body = "Artificial intelligence is used in scientific research and requires human validation. " * 5
        verified = {**known, "published_at": (now - timedelta(days=1)).isoformat(),
                    "page_read": True, "page_text": body, "article_chars": len(body),
                    "date_origin": "publisher_metadata", "article_truncated": False}
        with patch.object(engine, "search_bing_news", return_value=[known, *unrelated_reads]), \
                patch.object(engine, "search_news", return_value=[]), \
                patch("neral_public_news.discover_public_news", return_value=[verified]):
            result = search_live_sources(engine, q, reader=MagicMock(return_value=[]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual([source["url"] for source in result["sources"]], [known["url"]])
        self.assertEqual(result["sources"][0]["date_origin"], "publisher_metadata")

    def test_rescue_can_change_feed_locale_for_language_neutral_entity(self):
        engine = WebSearch(opener=MagicMock())
        q = "Aktuelle Nachrichten zur NASA Artemis Mission"
        stamp = datetime.now(timezone.utc).isoformat()
        source = {"url": "https://example.org/space/nasa-artemis-report",
                  "title": "NASA Artemis public report", "published_at": stamp}
        def google(query, on_provider=None, *, language="de"):
            return [source] if language == "en" and "nasa" in query.casefold() and "artemis" in query.casefold() else []
        with patch.object(engine, "search_bing_news", return_value=[]), patch.object(engine, "search_news", side_effect=google) as feed:
            result = engine.search(q, news_only=True)
        self.assertEqual([hit["url"] for hit in result["results"]], [source["url"]])
        self.assertLessEqual(feed.call_count, 3)

    def test_airport_rescue_query_uses_english_name_without_losing_airport(self):
        engine = WebSearch(opener=MagicMock())
        q = "Was gibt es Neues am Flughafen MUC?"
        stamp = datetime.now(timezone.utc).isoformat()
        source = {"url": "https://example.org/aviation/munich-airport",
                  "title": "Munich Airport public announcement", "published_at": stamp}
        def google(query, on_provider=None, *, language="de"):
            return [source] if language == "en" and "munich airport" in query.casefold() else []
        with patch.object(engine, "search_bing_news", return_value=[]), patch.object(engine, "search_news", side_effect=google):
            result = engine.search(q, news_only=True)
        self.assertEqual([hit["url"] for hit in result["results"]], [source["url"]])

    def test_opaque_rescue_tries_distinct_publishers_with_existing_caps(self):
        from neral_news_articles import discover_publisher_articles

        engine = WebSearch()
        query = "Latest news about coral reefs"
        stamp = datetime.now(timezone.utc).isoformat()
        def opaque(i, origin, headline):
            return {"url": f"https://news.google.com/rss/articles/fixture{i}",
                    "publisher_url": origin, "title": headline, "published_at": stamp}
        headlines = [
            opaque(1, "https://www.daily.example.com/", "Coral reefs survey shows change - Daily"),
            opaque(2, "https://daily.example.com/", "Coral reefs survey broadens coverage - Daily"),
            opaque(3, "https://independent.example.com/", "Coral reefs survey records recovery - Independent"),
            opaque(4, "https://daily.example.com/", "Coral reefs study adds new locations - Daily"),
        ]
        direct = {"url": "https://independent.example.com/environment/study",
                  "title": "Coral reefs survey records recovery"}
        article = ("<article><h1>Coral reefs survey records recovery</h1>"
                   "<p>Teams monitoring coral reefs published new evidence of recovery in a periodic survey across multiple areas and separately reported conditions at each site.</p>"
                   "<p>The coral reefs survey uses a consistent field protocol; the researchers released records and methods so others can verify the published observations.</p></article>")
        def fallback(query, on_provider=None):
            return [direct] if query.endswith(" site:independent.example.com") else []
        with patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=headlines), \
                patch.object(engine, "search_bing", side_effect=fallback) as provider, \
                patch("neral_news_articles.discover_publisher_articles", return_value=[]) as publishers, \
                patch("neral_public_news.discover_public_news", return_value=[]), \
                patch("neral_page_reader.fetch_and_extract", return_value={"url": direct["url"], "markup": article}) as fetch:
            # Stub only the opaque discovery pass, not the actual page reader.
            publishers.side_effect = lambda candidates, q, **kwargs: (
                discover_publisher_articles(candidates, q, **kwargs) if kwargs else [])
            result = search_live_sources(engine, query)
        fetch.assert_called_once_with(direct["url"], query, include_markup=True)
        self.assertEqual([c.args[0].split(" site:")[1] for c in provider.call_args_list],
                         ["daily.example.com", "independent.example.com"])
        self.assertEqual(publishers.call_args_list[0].args[0], [headlines[i] for i in (0, 2, 1, 3)])
        self.assertEqual(result["status"], "ok")
        self.assertEqual([source["url"] for source in result["sources"]], [direct["url"]])

    def test_short_opaque_title_does_not_consume_a_publisher_lookup_slot(self):
        engine = WebSearch(opener=MagicMock())
        stamp = datetime.now(timezone.utc).isoformat()
        headlines = [
            {"url": f"https://news.google.com/rss/articles/shortslot{i}",
             "publisher_url": f"https://{publisher}.example.com/",
             "published_at": stamp, "title": headline}
            for i, (publisher, headline) in enumerate((
                ("daily", "Coral reefs"), ("independent", "Coral reefs survey records recovery"),
                ("daily", "Coral reefs survey shows change"),
            ))
        ]
        with patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=headlines), \
                patch.object(engine, "search_bing", return_value=[]) as provider:
            engine.search("Latest news about coral reefs", news_only=True)
        self.assertEqual([c.args[0].split(" site:")[1] for c in provider.call_args_list],
                         ["independent.example.com", "daily.example.com"])

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
