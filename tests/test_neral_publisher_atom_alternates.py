"""Original article recovery from a publisher's declared Atom feed."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from neral_v43_live_web import route_live_query, search_live_sources
from neral_web_search import WebSearch


HOME = "https://science.example.org/"
FEED = HOME + "atom.xml"
ORIGINAL = HOME + "environment/coastal-coral-reefs-survey"
TITLE = "Coastal coral reefs survey reports changes"
OPAQUE = "https://news.google.com/rss/articles/fixture"
OFF_HOST = (
    '<link rel="alternate" type="text/html" hreflang="fr" '
    'href="https://translation.example.org/article" />'
)
ALTERNATES = OFF_HOST + '<link rel="alternate" type="application/pdf" href="/download.pdf" />'
ARTICLE = (
    f"<article><h1>{TITLE}</h1>"
    "<p>The coastal coral reefs survey recorded multiple sites around the coast "
    "and compared new observations with baseline conditions measured in the previous season.</p>"
    "<p>The researchers describe how the coral reefs observations were recorded at "
    "each monitoring site and where details about the field team and sampling plan were published.</p>"
    "</article>"
)


@pytest.mark.parametrize("order", ["original-first", "original-last", "original-before-offhost", "no-eligible"])
def test_publisher_atom_recovery_ignores_ineligible_alternates(order):
    query = route_live_query("Latest news about coral reefs")
    stamp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    article_link = '<link rel="alternate" type="text/html" href="/environment/coastal-coral-reefs-survey" />'
    links = {
        "original-first": article_link + ALTERNATES,
        "original-last": ALTERNATES + article_link,
        "original-before-offhost": article_link + OFF_HOST,
        "no-eligible": ALTERNATES,
    }[order]
    atom = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<entry>{links}<title>{TITLE}</title></entry></feed>"
    )
    row = {"url": OPAQUE, "title": TITLE, "published_at": stamp, "publisher_url": HOME}

    def fetch(url, *_args, **_kwargs):
        if url == HOME:
            return {"url": url, "markup": '<link rel="alternate" type="application/atom+xml" href="/atom.xml">'}
        if url == FEED:
            return {"url": url, "markup": atom}
        if url == ORIGINAL:
            return {"url": url, "markup": ARTICLE}
        return None

    engine = WebSearch()
    with patch.object(engine, "search_bing_news", return_value=[]), \
            patch.object(engine, "search_news", return_value=[row]), \
            patch.object(engine, "search_bing", return_value=[]), \
            patch("neral_public_news.discover_public_news", return_value=[]), \
            patch("neral_page_reader.fetch_and_extract", side_effect=fetch) as transport:
        out = search_live_sources(engine, query, include_diagnostics=True)

    requested = [call.args[0] for call in transport.call_args_list]
    if order == "no-eligible":
        assert out["status"] == "headlines_only"
        assert out["context"] == ""
        assert requested == [HOME, FEED, HOME + "sitemap.xml", OPAQUE]
    else:
        assert out["status"] == "ok"
        assert [row["url"] for row in out["sources"]] == [ORIGINAL]
        assert out["sources"][0]["date_origin"] == "feed"
        assert requested == [HOME, FEED, ORIGINAL]
