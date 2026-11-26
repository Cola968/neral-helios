"""Bounded forecast search recovery, without network or model calls."""
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from neral_page_reader import read_top_pages
from neral_v43_live_web import route_live_query, search_live_sources
from neral_web_search import WebSearch, live_query_variants


QUESTION = "Wie ist das Wetter morgen in Berlin?"
PUBLISHER = "https://weather.example.org/forecast/germany"
FORECAST_PAGE = (
    "Wetter in Berlin: In dieser fiktiven Vorhersage werden morgen 17 Grad erwartet. "
    "Die Prognose gilt für die Stadt Berlin und enthält Temperaturen für mehrere Tageszeiten."
)


def respond(data, mime):
    response = BytesIO(data)
    response.status = 200
    response.headers = {"Content-Type": mime}
    return response


def ddg_html(title):
    return (
        f'<a class="result__a" href="{PUBLISHER}">{title}</a>'
        '<div class="result__snippet">Fiktive Prognose: 99 Grad</div>'
    ).encode()


def bing_xml():
    return (
        f"<rss><channel><item><title>Wetter morgen in Berlin</title><link>{PUBLISHER}</link>"
        "<description>Fiktive Prognose: 99 Grad</description></item></channel></rss>"
    ).encode()


@pytest.mark.parametrize("initial,alternate_topic,expected", [
    ("empty", "Wetter morgen in Berlin", "ok"),
    ("empty", "Wetter morgen in Potsdam", "empty"),
    ("empty", None, "empty"),
    ("bing", None, "ok"),
    ("ddg", None, "ok"),
], ids=("recover-on-alternate", "reject-wrong-city", "empty-alternate",
        "skip-if-bing-found", "skip-if-primary-found"))
def test_weather_fallback_uses_bounded_alternative_and_reads_original(
    initial, alternate_topic, expected,
):
    primary, alternate = live_query_variants(QUESTION)
    assert primary.casefold() != alternate.casefold()
    assert all({"berlin", "morgen"}.issubset(q.casefold().split()) for q in (primary, alternate))

    class Opener:
        def __init__(self):
            self.calls = []

        def open(self, req, timeout):
            assert timeout <= 3
            uri = urlsplit(req.full_url)
            query = parse_qs(uri.query)["q"][0]
            self.calls.append((uri.hostname, query))
            if uri.hostname == "html.duckduckgo.com":
                title = ("Wetter morgen in Berlin" if initial == "ddg" else None) if query == primary else alternate_topic
                return respond(ddg_html(title) if title else b"<html>No results</html>", "text/html")
            assert uri.hostname == "www.bing.com"  # Never synthesize a forecast from Wikipedia.
            if initial == "bing":
                return respond(bing_xml(), "application/rss+xml")
            if query == primary:
                return respond(b"<rss><channel></channel></rss>", "application/rss+xml")
            raise HTTPError(req.full_url, 429, "fixture throttled", {}, None)

    reads = []
    def fetch(url, question):
        assert question == QUESTION
        reads.append(url)
        return {"url": url, "text": FORECAST_PAGE}

    def reader(hits, query):
        return read_top_pages(hits, query, fetch=fetch)

    opener = Opener()
    query = route_live_query(QUESTION)
    result = search_live_sources(WebSearch(opener=opener), query, reader=reader)

    assert result["status"] == expected
    required = [("html.duckduckgo.com", primary)]
    if initial != "ddg":
        required.append(("www.bing.com", primary))
    if initial == "empty":
        required.extend([("www.bing.com", alternate), ("html.duckduckgo.com", alternate)])
    assert opener.calls == required
    if expected == "ok":
        assert reads == [PUBLISHER]
        assert result["sources"][0]["url"] == PUBLISHER
        assert "17 Grad" in result["context"]
        assert "99 Grad" not in result["context"]
        assert result["provider"] == {
            "ddg": "DuckDuckGo HTML",
            "bing": "Bing RSS",
            "empty": "DuckDuckGo HTML alternative query",
        }[initial]
    else:
        assert reads == []
        assert result["sources"] == []
        assert result["context"] == ""


def test_weather_alternate_does_not_recontact_rate_limited_primary():
    primary, alternate = live_query_variants(QUESTION)
    calls = []

    class Opener:
        def open(self, req, timeout):
            uri = urlsplit(req.full_url)
            calls.append((uri.hostname, parse_qs(uri.query)["q"][0]))
            if uri.hostname == "html.duckduckgo.com":
                raise HTTPError(req.full_url, 429, "fixture throttled", {}, None)
            assert uri.hostname == "www.bing.com"
            return respond(b"<rss><channel></channel></rss>", "application/rss+xml")

    def reader(*_):
        pytest.fail("No source was discoverable")

    result = search_live_sources(WebSearch(opener=Opener()), QUESTION, reader=reader)
    assert result["status"] == "empty"
    assert calls == [("html.duckduckgo.com", primary),
                     ("www.bing.com", primary), ("www.bing.com", alternate)]
