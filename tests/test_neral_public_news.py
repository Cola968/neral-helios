"""Public news index is discovery only; publication must come from the article."""
import unittest,json
from datetime import datetime,timezone,timedelta
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import parse_qs, urlsplit
from neral_public_news import parse_gdelt_candidates, article_publication_date, read_gdelt_candidate
from neral_v43_live_web import search_live_sources
from neral_web_search import WebSearch

NOW=datetime.now(timezone.utc)
STAMP=NOW.isoformat()
SEEN=NOW.strftime("%Y%m%dT%H%M%SZ")
URL="https://example.org/science/artemis-test"
TITLE="NASA prepares Artemis mission"
P1="NASA is preparing Artemis tests with a new series of mission activities. Engineers are verifying key spacecraft components and documenting the preparation."
P2="The Artemis team will continue the mission preparation using additional technical tests and a review of the work already completed by NASA."
def html(meta):
    return f'<html><head>{meta}</head><body><article><h1>{TITLE}</h1><p>{P1}</p><p>{P2}</p></article></body></html>'
class PublicNews(unittest.TestCase):
    def test_index_date_is_not_publication_date(self):
        rows=parse_gdelt_candidates(json.dumps({"articles":[
            {"url":URL,"title":TITLE,"seendate":SEEN},
            {"url":"https://127.0.0.1/a","title":TITLE,"seendate":SEEN},
            {"url":"https://example.org/old","title":TITLE,"seendate":(NOW-timedelta(days=30)).strftime("%Y%m%dT%H%M%SZ")}
        ]}),now=NOW)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["url"],URL)
        self.assertIn("indexed_at",rows[0])
        self.assertNotIn("published_at",rows[0])
    def test_gdelt_keeps_topic_candidates_beyond_first_ten_without_extra_requests(self):
        query = "Latest news about coral reefs"
        title = "Coral reefs survey results"
        candidate = "https://publisher.example.org/environment/reef-survey"
        beyond_limit = "https://publisher.example.org/environment/reef-beyond-limit"
        rows = [
            {"url": f"https://publisher.example.org/civic/roadworks-{i}",
             "title": f"Council schedules road repairs {i}", "seendate": SEEN}
            for i in range(15)
        ]
        rows[11] = {"url": candidate, "title": title, "seendate": SEEN}
        rows.append({"url": beyond_limit, "title": title, "seendate": SEEN})
        response = MagicMock(status=200, headers={"Content-Type": "application/json"})
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"articles": rows}).encode()
        paragraphs = (
            "<p>Researchers monitoring coral reefs published a new set of observations. "
            "Their field teams surveyed reef habitats and recorded changes in local conditions.</p>"
            "<p>The coral reefs survey describes which sites were monitored and how the results were verified. "
            "The research team published a table of observations alongside the article.</p>"
        )
        markup = (f'<meta property="article:published_time" content="{STAMP}">'
                  f"<article><h1>{title}</h1>{paragraphs}</article>")
        engine = WebSearch()
        engine.opener = Mock(open=Mock(return_value=response))
        with patch.object(engine, "search_bing_news", return_value=[]), \
                patch.object(engine, "search_news", return_value=[]), \
                patch("neral_public_news._LAST", None), \
                patch("neral_page_reader.fetch_and_extract", return_value={"url": candidate, "markup": markup}) as fetch:
            result = search_live_sources(engine, query, include_diagnostics=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["url"] for row in result["sources"]], [candidate])
        self.assertEqual(result["sources"][0]["published_at"], STAMP)
        self.assertEqual(result["sources"][0]["date_origin"], "publisher_metadata")
        self.assertEqual(result["stage_counts"]["discovery_read_bodies"], 1)
        fetch.assert_called_once_with(candidate, query, include_markup=True)
        engine.opener.open.assert_called_once()
        request = engine.opener.open.call_args.args[0]
        self.assertEqual(parse_qs(urlsplit(request.full_url).query)["maxrecords"], ["15"])

    def test_gdelt_topic_filter_keeps_original_order_and_ten_candidate_cap(self):
        relevant = [
            {"url": f"https://publisher.example.org/environment/reef-{i}",
             "title": f"Coral reefs survey results {i}", "seendate": SEEN}
            for i in range(13)
        ]
        # An earlier off-topic representation of a URL may not consume it.
        unrelated = {**relevant[0], "title": "Council schedules road repairs"}
        rows = parse_gdelt_candidates(json.dumps({"articles": [unrelated, *relevant]}),
                                     now=NOW, query="Latest news about coral reefs")
        self.assertEqual([row["url"] for row in rows], [row["url"] for row in relevant[:10]])
        self.assertTrue(all("published_at" not in row for row in rows))

    def test_publication_metadata_not_modification(self):
        self.assertEqual(article_publication_date(html(f'<meta property="article:published_time" content="{STAMP}">')),STAMP)
        self.assertIsNone(article_publication_date(html(f'<meta property="article:modified_time" content="{STAMP}">')))
    def test_read_requires_visible_article_and_recent_publication(self):
        hit={"url":URL,"title":TITLE,"indexed_at":STAMP}
        self.assertIsNone(read_gdelt_candidate(hit,"Aktuelle Nachrichten zur NASA Artemis Mission",fetch=lambda u:{"url":u,"markup":html("")}))
        meta=f'<meta property="article:published_time" content="{STAMP}">'
        result=read_gdelt_candidate(hit,"Aktuelle Nachrichten zur NASA Artemis Mission",fetch=lambda u:{"url":u,"markup":html(meta)})
        self.assertTrue(result["page_read"])
        self.assertEqual(result["published_at"],STAMP)
        self.assertIn(P2,result["page_text"])
    def test_article_jsonld_publication_can_verify_visible_publisher_body(self):
        article = {"@graph": [{"@type": "Organization", "datePublished": "2000-01-01"},
                              {"@type": ["NewsArticle", "Article"], "datePublished": STAMP}]}
        markup = html('<script type="application/ld+json">' + json.dumps(article) + '</script>')
        hit = {"url": URL, "title": TITLE, "indexed_at": STAMP}
        out = read_gdelt_candidate(hit, "Aktuelle Nachrichten zur NASA Artemis Mission",
                                   fetch=lambda url: {"url": url, "markup": markup})
        self.assertIsNotNone(out)
        self.assertEqual(out["published_at"], STAMP)
        self.assertIn(P1, out["page_text"])

    def test_jsonld_ignores_nonarticle_and_updated_dates_and_rejects_conflicts(self):
        def ld(value):
            return '<script type="application/ld+json">' + json.dumps(value) + '</script>'
        self.assertIsNone(article_publication_date(html(ld(
            {"@type": "WebSite", "datePublished": STAMP}))))
        self.assertIsNone(article_publication_date(html(ld(
            {"@type": "NewsArticle", "dateModified": STAMP}))))
        self.assertIsNone(article_publication_date(html(
            f'<meta property="article:published_time" content="{STAMP}">' +
            ld({"@type": "https://schema.org/NewsArticle", "datePublished": "2000-01-01"}))))

    def test_paywall_or_wrong_topic_does_not_count(self):
        hit={"url":URL,"title":TITLE,"indexed_at":STAMP}
        meta=f'<meta property="article:published_time" content="{STAMP}">'
        self.assertIsNone(read_gdelt_candidate(hit,"Aktuelle Nachrichten zur NASA Artemis Mission",fetch=lambda u:{"url":u,"markup":html(meta).replace(P2,"Nur für Abonnenten")}))
        self.assertIsNone(read_gdelt_candidate(hit,"Aktuelle Nachrichten zur Bundesliga",fetch=lambda u:{"url":u,"markup":html(meta)}))
if __name__=="__main__":
    unittest.main()
