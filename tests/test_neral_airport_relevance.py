"""Airport entity relevance regression: no network or GPU."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch

from neral_web_search import WebSearch, ranked_topic_hits, rewrite_search_query
from neral_v43_live_web import search_live_sources

QUERY = "Welche aktuellen Nachrichten gibt es zum Flughafen BER?"
WRONG = {"url":"https://www.frankfurt-airport.com/de.html","title":"Flughafen Frankfurt","snippet":"Aktuelle Flugziele"}
RIGHT = {"published_at":datetime.now(timezone.utc).isoformat(),"url":"https://ber.berlin-airport.de/de.html","title":"Flughafen Berlin Brandenburg","snippet":"Neue Entwicklung am Flughafen BER"}
MUNICH = {"url":"https://www.munich-airport.de/","title":"Flughafen München","snippet":"Airport MUC"}


def _read_article(airport):
    body = (
        f"Am {airport} gibt es neue Verbindungen. Die Fluggesellschaften kündigten an, den "
        "Sommerflugplan um Regionalstrecken zu erweitern und die Abflugzeiten später zu veröffentlichen.\n"
        "Für die Reiseplanung stellt der Betreiber die Routen und weitere Hinweise getrennt bereit. "
        "Zusätzliche Änderungen am Flugplan sollen in der nächsten öffentlichen Mitteilung folgen."
    )
    return {**RIGHT, "page_read": True, "page_text": body, "article_chars": len(body)}


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
        reader=Mock(return_value=[_read_article("Flughafen Frankfurt")])
        out=search_live_sources(engine, QUERY, reader=reader)
        self.assertEqual(out["status"],"empty")
        self.assertEqual(out["sources"],[])

    def test_relevant_read_body_is_accepted(self):
        engine=Mock()
        engine.search.return_value={"provider":"test","results":[RIGHT]}
        reader=Mock(return_value=[_read_article("Flughafen Berlin Brandenburg")])
        self.assertEqual(search_live_sources(engine,QUERY,reader=reader)["status"],"ok")

    def test_dual_airport_coverage_only_counts_accepted_article_bodies(self):
        query = "Welche aktuellen Nachrichten gibt es zum Flughafen BER und zum Flughafen MUC?"
        now = datetime.now(timezone.utc)
        stamp = (now - timedelta(days=1)).isoformat()
        stale = (now - timedelta(days=30)).isoformat()
        berlin = "https://publisher.example.org/news/ber-muc-comparison.html"
        munich = "https://different.example.org/news/regional-routes.html"
        # Metadata names Munich even when the accepted visible article does not.
        title = "Verbindungen an den Flughäfen BER und MUC"

        def markup(airport, published):
            paragraphs = _read_article(airport)["page_text"].splitlines()
            body = "".join(f"<p>{p}</p>" for p in paragraphs)
            return (f'<meta property="article:published_time" content="{published}">'
                    f"<article><h1>{title}</h1>{body}</article>")

        for munich_published, expected_observed in (
            (None, ["BER"]), (stale, ["BER"]), (stamp, ["BER", "MUC"])
        ):
            with self.subTest(munich_published=munich_published):
                engine = Mock()
                urls = [berlin] if munich_published is None else [berlin, munich]
                engine.search.return_value = {"provider": "fixture", "results": [
                    {"url": url, "title": title, "published_at": stamp} for url in urls
                ]}
                pages = {berlin: {"url": berlin, "markup": markup("Flughafen Berlin Brandenburg", stamp)}}
                if munich_published is not None:
                    pages[munich] = {"url": munich, "markup": markup("Munich Airport", munich_published)}
                fetch = Mock(side_effect=lambda url, *args, **kwargs: pages[url])
                with patch("neral_page_reader.fetch_and_extract", fetch):
                    result = search_live_sources(engine, query, include_diagnostics=True)
                self.assertEqual(result["status"], "ok")
                self.assertEqual({row["url"] for row in result["sources"]},
                                 {berlin, munich} if munich_published == stamp else {berlin})
                self.assertEqual(result["airport_coverage"], {
                    "requested": ["BER", "MUC"],
                    "observed_in_accepted_bodies": expected_observed,
                    "not_observed": [code for code in ("BER", "MUC") if code not in expected_observed],
                })
                self.assertEqual({call.args[0] for call in fetch.call_args_list}, set(urls))
                self.assertEqual(fetch.call_count, len(urls))

    def test_single_airport_and_default_payloads_do_not_add_coverage(self):
        engine = Mock()
        engine.search.return_value = {"provider": "fixture", "results": [RIGHT]}
        reader = Mock(return_value=[_read_article("Flughafen Berlin Brandenburg")])
        single = search_live_sources(engine, QUERY, reader=reader, include_diagnostics=True)
        self.assertEqual(single["status"], "ok")
        self.assertNotIn("airport_coverage", single)
        query = "Aktuelle Nachrichten zum Flughafen BER und Flughafen MUC"
        default = search_live_sources(engine, query, reader=reader)
        self.assertEqual(default["status"], "ok")
        self.assertNotIn("airport_coverage", default)

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
