"""Fresh news evidence must be recent, readable and not a reference page."""
import unittest
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
from unittest.mock import Mock, MagicMock, patch
from urllib.parse import quote

from neral_web_search import WebSearch, parse_bing_rss, is_fresh_news_query
from neral_v43_live_web import search_live_sources

Q="Was gibt es Neues am Flughafen MUC?"
BER="Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
URL="https://www.munich-airport.de/meldung-123"
BODY="Der Flughafen München meldet neue Verbindungen."
NOW=datetime.now(timezone.utc)
DATE=NOW.isoformat()
HIT={"url":URL,"title":"Flughafen München meldet neue Verbindungen","snippet":BODY,"published_at":DATE}


class FreshNewsEvidence(unittest.TestCase):
    def test_colloquial_news_intent(self):
        self.assertTrue(is_fresh_news_query(Q))

    def test_feed_direct_publisher_link_and_date(self):
        link="https://www.bing.com/news/apiclick.aspx?url="+quote(URL,safe="")
        stamp=format_datetime(NOW)
        xml=f"<rss><channel><item><title>MUC News</title><link>{link}</link><pubDate>{stamp}</pubDate></item></channel></rss>"
        rows=parse_bing_rss(xml,max_age_days=7,now_utc=NOW)
        self.assertEqual(rows[0]["url"],URL)
        self.assertIn("published_at",rows[0])

    def test_news_redirect_to_private_address_is_rejected(self):
        link="https://www.bing.com/news/apiclick.aspx?url="+quote("https://127.0.0.1/admin",safe="")
        xml=f"<rss><channel><item><title>MUC News</title><link>{link}</link><pubDate>{format_datetime(NOW)}</pubDate></item></channel></rss>"
        self.assertEqual(parse_bing_rss(xml,max_age_days=7,now_utc=NOW),[])

    def test_news_only_does_not_return_undated_reference(self):
        search=WebSearch(opener=MagicMock())
        with patch.object(search,"search_news",return_value=[]), patch.object(search,"search_bing_news",return_value=[]), patch.object(search,"search_wikipedia",return_value=[HIT]) as wiki:
            result=search.search(Q,news_only=True)
        self.assertEqual(result["results"],[])
        wiki.assert_not_called()

    def test_news_only_does_not_accept_opaque_aggregator_as_article(self):
        search=WebSearch(opener=MagicMock())
        hit={**HIT,"url":"https://news.google.com/rss/articles/opaque"}
        with patch.object(search,"search_news",return_value=[hit]), patch.object(search,"search_bing_news",return_value=[]), patch.object(search,"search_bing",return_value=[]):
            result=search.search(Q,news_only=True)
        self.assertEqual(result["results"],[])

    def test_recent_direct_article_accepted(self):
        search=Mock()
        search.search.return_value={"provider":"fixture","results":[HIT]}
        page={**HIT,"page_read":True,"page_text":BODY}
        out=search_live_sources(search,Q,reader=Mock(return_value=[page]))
        self.assertEqual(out["status"],"ok")
        self.assertIn("published_at",out["sources"][0])

    def test_stale_or_missing_date_never_counts_as_news(self):
        for stamp in (None,(NOW-timedelta(days=40)).isoformat()):
            with self.subTest(stamp=stamp):
                hit={**HIT,"published_at":stamp}
                search=Mock()
                search.search.return_value={"provider":"fixture","results":[hit]}
                page={**hit,"page_read":True,"page_text":BODY}
                out=search_live_sources(search,Q,reader=Mock(return_value=[page]))
                self.assertEqual(out["status"],"empty")

    def test_general_fact_query_does_not_require_news_date(self):
        search=Mock()
        search.search.return_value={"provider":"fixture","results":[HIT]}
        page={**HIT,"page_read":True,"page_text":BODY}
        self.assertEqual(search_live_sources(search,"Flughafen MUC",reader=Mock(return_value=[page]))["status"],"ok")


if __name__=="__main__":
    unittest.main()
