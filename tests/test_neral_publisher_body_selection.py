"""Publisher discovery must save its result slots for relevant original bodies."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from neral_v43_live_web import route_live_query, search_live_sources
from neral_web_search import WebSearch


def test_repeated_off_topic_article_bodies_do_not_crowd_out_a_later_original():
    query = route_live_query("Latest news about coral reefs")
    stamp = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    title = "Coral reefs survey reports new field results"
    good_body = (
        "<p>The coral reefs survey describes the independent field study and compares marine species "
        "at more than a dozen monitoring sites across the region. Results varied by site.</p>"
        "<p>Researchers published the complete coral reefs survey and documented their sampling "
        "protocol, including station locations and how observations were independently reviewed.</p>"
    )
    wrong_body = (
        "<p>The municipal road maintenance programme describes resurfacing work across a number of "
        "intersections. The contractor submitted updated street closures for review.</p>"
        "<p>The unrelated roadworks plan also publishes the city transport timetable, which is "
        "scheduled to change when the resurfacing work finishes later in the year.</p>"
    )
    homes = [f"https://paper{i}.example.org/" for i in range(5)]
    good_article = homes[-1] + "news/field-results"
    opaque = [
        {"url": f"https://news.google.com/rss/articles/edition-{i}", "publisher_url": home,
         "title": title + " - Paper", "published_at": stamp}
        for i, home in enumerate(homes)
    ]

    def transport(url, *args, **kwargs):
        if url in homes:
            return {"url": url, "markup": f'<a href="/news/field-results">{title}</a>'}
        if url.endswith("/news/field-results"):
            paragraphs = good_body if url == good_article else wrong_body
            return {"url": url, "markup": f"<article><h1>{title}</h1>{paragraphs}</article>"}
        return None

    engine = WebSearch()
    with patch.object(engine, "search_bing_news", return_value=[]), \
            patch.object(engine, "search_news", return_value=opaque), \
            patch.object(engine, "search_bing", return_value=[]) as targeted, \
            patch("neral_public_news.discover_public_news", return_value=[]), \
            patch("neral_page_reader.fetch_and_extract", side_effect=transport) as fetch:
        result = search_live_sources(engine, query, include_diagnostics=True)

    assert result["status"] == "ok"
    assert [row["url"] for row in result["sources"]] == [good_article]
    assert result["sources"][0]["date_origin"] == "feed"
    assert result["stage_counts"]["discovery_read_bodies"] == 1
    assert result["stage_counts"]["reader_candidates"] == 0
    assert "unrelated roadworks" not in result["context"]
    assert "complete coral reefs survey" in result["context"]
    assert targeted.call_count == 2
    assert [call.args[0] for call in fetch.call_args_list].count(good_article) == 1
