"""Airport entity relevance regression: no network or GPU."""
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from neral_web_search import WebSearch, ranked_topic_hits, rewrite_search_query
from neral_v43_live_web import search_live_sources

QUERY = "Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
WRONG = {"url":"https://www.frankfurt-airport.com/de.html","title":"Flughafen Frankfurt","snippet":"Aktuelle Flugziele"}
RIGHT = {"published_at":datetime.now(timezone.utc).isoformat(),"url":"https://ber.berlin-airport.de/de.html","title":"Flughafen Berlin Brandenburg","snippet":"Neue Entwicklung am Flughafen BER"}
MUNICH = {"url":"https://www.munich-airport.de/","title":"Flughafen München","snippet":"Airport MUC"}


class AirportRelevance(unittest.TestCase):
    def test_ber_metadata_rejects_other_airports(self):
        self.assertEqual(ranked_topic_hits([WRONG, MUNICH, RIGHT],QUERY), [RIGHT])

    def test_same_rule_generalizes_to_other_airport_codes(self):
        self.assertEqual(ranked_topic_hits([RIGHT, WRONG, MUNICH],"Flughafen MUC Nachrichten"), [MUNICH])

    def test_query_preserves_code_and_expands_airport_name(self):
        q = rewrite_search_query(QUERY).casefold()
        self.assertIn("ber", q.split())
        self.assertIn("berlin", q.split())
        self.assertIn("brandenburg", q.split())

    def test_wrong_page_body_is_never_live_evidence(self):
        engine=Mock()
        engine.search.return_value={"provider":"test","results":[RIGHT]}
        reader=Mock(return_value=[{"url":RIGHT["url"],"title":RIGHT["title"],"page_read":True,"page_text":"Flughafen Frankfurt meldet neue Strecken. München wächst."}])
        out=search_live_sources(engine, QUERY, reader=reader)
        self.assertEqual(out["status"],"empty")
        self.assertEqual(out["sources"],[])

    def test_relevant_read_body_is_accepted(self):
        engine=Mock()
        engine.search.return_value={"provider":"test","results":[RIGHT]}
        reader=Mock(return_value=[{"url":RIGHT["url"],"title":RIGHT["title"],"page_read":True,"page_text":"Am Flughafen Berlin Brandenburg gibt es neue Verbindungen."}])
        self.assertEqual(search_live_sources(engine,QUERY,reader=reader)["status"],"ok")

    def test_off_topic_first_provider_triggers_bing(self):
        engine=WebSearch(opener=MagicMock())
        response=engine.opener.open.return_value.__enter__.return_value
        response.status=200
        response.headers={"Content-Type":"text/html, application/rss+xml, xml"}
        response.read.return_value=b"<html></html>"
        with patch("neral_web_search.parse_search_results",return_value=[WRONG]), patch("neral_web_search.parse_bing_rss",return_value=[RIGHT]):
            result=engine.search(QUERY)
        self.assertEqual(result["results"],[RIGHT])
        self.assertEqual(result["provider"],"Bing RSS")

if __name__=="__main__":
    unittest.main()
