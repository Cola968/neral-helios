"""Public official news index: only bounded, dated same-origin discoveries."""
import unittest
from datetime import datetime,timezone
from unittest.mock import Mock, patch
from neral_v43_live_web import route_live_query
from neral_official_news import parse_official_news_index, discover_official_news

BASE="https://corporate.berlin-airport.de/de/unternehmen-presse/presseportal/pressemitteilungen.html"
CLOCK=datetime(2026,11,25,tzinfo=timezone.utc)

class OfficialNewsTests(unittest.TestCase):
    def test_colloquial_request_routes_to_web(self):
        self.assertTrue(route_live_query("Was gibt es Neues am Flughafen MUC?"))

    def test_index_only_discovers_recent_same_origin_dated_articles(self):
        html='<a href="/presse/2026-11-23-news.html">23. November 2026 Neue Verbindungen am BER</a><a href="/presse/2026-10-10-old.html">10. Oktober 2026 BER alt</a><a href="https://other.example.com/news">24. November 2026 BER fremd</a><a href="/undatiert">BER</a>'
        rows=parse_official_news_index(html,BASE,now=CLOCK)
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]["url"].endswith("2026-11-23-news.html"))
        self.assertEqual(rows[0]["published_at"],"2026-11-23T00:00:00+00:00")

    def test_conflicting_dates_and_future_entries_rejected(self):
        html='<a href="/2026-10-01-old.html">24. November 2026 BER</a><a href="/2026-11-30-future.html">30. November 2026 BER</a>'
        self.assertEqual(parse_official_news_index(html,BASE,now=CLOCK),[])

    def test_authority_atom_uses_same_host_https_and_explicit_publication(self):
        feed = "https://press.example.org/feeds/civic.xml"
        stamp = "2026-11-22T12:00:00+00:00"
        def entry(href, title, date_tag="published", date=stamp):
            return (f'<entry><title>{title}</title><link href="{href}"/>'
                    f'<{date_tag}>{date}</{date_tag}></entry>')
        markup = '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join([
            entry("http://press.example.org/news/terminal", "Flughafen München ändert Terminalzugang"),
            entry("http://other.example.org/news/external", "Flughafen München fremd"),
            entry("/news/undated", "Flughafen München undatiert", "updated"),
            entry("/news/past", "Flughafen München alt", date="2026-11-10T12:00:00+00:00"),
            entry("/news/street", "Die Stadt erneuert eine Straße"),
        ]) + '</feed>'
        fetch = Mock(return_value={"url": feed, "markup": markup})
        with patch("neral_official_news.OFFICIAL_NEWS_INDEXES", {}), \
                patch("neral_official_news.OFFICIAL_NEWS_FEEDS", {"MUC": feed}, create=True):
            rows = discover_official_news("Was gibt es Neues am Flughafen MUC?", fetch=fetch, now=CLOCK)
        self.assertEqual([row["url"] for row in rows], ["https://press.example.org/news/terminal"])
        self.assertEqual(rows[0]["published_at"], stamp)
        self.assertNotIn("page_text", rows[0])
        fetch.assert_called_once_with(feed)

    def test_unknown_entity_does_not_contact_any_official_index(self):
        fetch=Mock()
        self.assertEqual(discover_official_news("Aktuelle Nachrichten zur NASA",fetch=fetch),[])
        fetch.assert_not_called()

    def test_discovery_does_not_treat_index_as_article(self):
        fetch=Mock(return_value={"url":BASE,"markup":'<a href="/presse/2026-11-23-news.html">23. November 2026 Neue Verbindungen am BER</a>'})
        rows=discover_official_news("Aktuelle Nachrichten am Flughafen BER",fetch=fetch,now=CLOCK)
        self.assertEqual(len(rows),1)
        self.assertNotEqual(rows[0]["url"],BASE)
        self.assertNotIn("page_text",rows[0])

if __name__=="__main__":
    unittest.main()
