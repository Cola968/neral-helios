"""The normal question path tolerates broken providers without using snippets."""
import json
from http.client import BadStatusLine, IncompleteRead
from urllib.parse import urlsplit

import pytest

from neral_page_reader import read_top_pages
from neral_v43_live_web import route_live_query, search_live_sources
from neral_web_search import WebSearch


QUESTION = "Welche Version von Python ist aktuell?"
SOURCE = "https://www.python.org/downloads/release/python-3182/"
PARTIAL = b'<a class="result__a" href="https://example.org/unread">Python 9.9.9</a>'
RSS = (
    "<rss><channel><item><title>Python 3.18.2 release</title>"
    f"<link>{SOURCE}</link><description>Python 3.18.2</description>"
    "</item></channel></rss>"
).encode()


class Response:
    status = 200

    def __init__(self, body, mime):
        self.body = body
        self.headers = {"Content-Type": mime}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, limit):
        assert limit > len(self.body)
        return self.body


@pytest.mark.parametrize("failure", [
    IncompleteRead(PARTIAL, 30),
    BadStatusLine("invalid provider response"),
])
def test_general_question_falls_back_from_broken_first_provider(failure):
    class Opener:
        def __init__(self):
            self.hosts = []

        def open(self, request, timeout):
            assert timeout <= 3
            host = urlsplit(request.full_url).hostname
            self.hosts.append(host)
            if host == "html.duckduckgo.com":
                raise failure
            assert host == "www.bing.com"
            return Response(RSS, "application/rss+xml")

    def fetch(url, query):
        assert query == QUESTION and url == SOURCE
        return {"url": url, "text": "Python 3.18.2 is the latest Python release."}

    def reader(hits, query):
        return read_top_pages(hits, query, fetch=fetch)

    opener = Opener()
    assert route_live_query(QUESTION, benchmark=True) is None
    query = route_live_query(QUESTION)
    result = search_live_sources(WebSearch(opener=opener), query, reader=reader)
    assert result["status"] == "ok"
    assert result["sources"][0]["url"] == SOURCE
    assert "9.9.9" not in result["context"]
    assert opener.hosts == ["html.duckduckgo.com", "www.bing.com"]


@pytest.mark.parametrize("broken", [None, [], {"query": None}, {"query": []},
                                    IncompleteRead(b'{"query":', 30), {"query": {"search": []}}])
def test_unusable_reference_response_is_contained_before_next_language(broken):
    class Opener:
        def __init__(self):
            self.wiki_hosts = []

        def open(self, request, timeout):
            host = urlsplit(request.full_url).hostname
            if host == "html.duckduckgo.com":
                return Response(b"", "text/html")
            if host == "www.bing.com":
                raise IncompleteRead(RSS, 10)
            self.wiki_hosts.append(host)
            if host == "de.wikipedia.org":
                if isinstance(broken, Exception):
                    raise broken
                return Response(json.dumps(broken).encode(), "application/json")
            assert host == "en.wikipedia.org"
            return Response(json.dumps({"query": {"search": [{
                "title": "Apollo 11", "snippet": "NASA lunar Moon landing mission"
            }]}}).encode(), "application/json")

    def fetch(url, _):
        assert url == "https://en.wikipedia.org/wiki/Apollo_11"
        return {"url": url, "text": "Apollo 11 was the NASA space mission that landed on the Moon in 1969."}

    def reader(hits, query):
        return read_top_pages(hits, query, fetch=fetch, wiki_fallback=lambda *_: [])

    opener = Opener()
    query = route_live_query("In welchem Jahr landete Apollo 11 auf dem Mond?")
    result = search_live_sources(WebSearch(opener=opener), query, reader=reader)
    assert result["status"] == "ok"
    assert [s["url"] for s in result["sources"]] == ["https://en.wikipedia.org/wiki/Apollo_11"]
    assert "en.wikipedia.org" in opener.wiki_hosts
