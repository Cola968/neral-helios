"""Exercise the browser's Flask chat endpoint without network or model inference."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from html import escape
from io import BytesIO
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest

from neral_three_tier_runtime import LUNA, RAW_9B, ThreeTierRuntime, _contextual_model_turns
from neral_web_search import WebSearch


ORIGIN = "http://127.0.0.1:9690"
CHAT_ROUTE = "/api/neral/chat/three-tier"
FEED_SNIPPET = "Synthetic feed-only amount: 999."
EMPTY_RSS = b"<rss><channel></channel></rss>"


class FixtureFeeds:
    def __init__(self, rss):
        self.rss = rss
        self.calls = []

    def open(self, request, timeout):
        parsed = urlsplit(request.full_url)
        self.calls.append((parsed.hostname, parsed.path))
        allowed = {
            ("www.bing.com", "/news/search"): self.rss,
            ("news.google.com", "/rss/search"): EMPTY_RSS,
        }
        assert request.get_method() == "GET"
        assert 0 < timeout <= 3.0
        payload = BytesIO(allowed[(parsed.hostname, parsed.path)])
        payload.status = 200
        payload.headers = {"Content-Type": "application/rss+xml"}
        return payload


def _rss(title, url, stamp):
    return (
        "<rss><channel><item>"
        f"<title>{escape(title)}</title><link>{escape(url)}</link>"
        f"<description>{FEED_SNIPPET}</description>"
        f"<pubDate>{format_datetime(stamp)}</pubDate>"
        "</item></channel></rss>"
    ).encode("utf-8")


def _post_chat(monkeypatch, feeds, prompt, page, answer=None):
    # Use the actual preview factory, route, session bootstrap, exposed status
    # and CSRF checks. Neither the fixture nor tests ever start a socket.
    from neral_preview_server import create_preview_app

    def forbidden(*args, **kwargs):
        pytest.fail("Real transport or sandbox must not be used")

    monkeypatch.setattr("socket.getaddrinfo", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    fetch = Mock(return_value=page)
    monkeypatch.setattr("neral_page_reader.fetch_and_extract", fetch)
    model = Mock(model=RAW_9B, loaded=True, adapter_sha256=None)
    luna = Mock(model=LUNA, loaded=True, adapter_sha256=None)
    contexts = []

    def fake_answer(turns, **kwargs):
        assert answer is not None, "Headline-only paths may not generate an answer"
        # The real HF backend uses this same boundary before tokenization.
        contexts.append(_contextual_model_turns(turns)[-1]["content"])
        return answer

    model.respond_chat.side_effect = fake_answer
    luna.respond_chat.side_effect = forbidden
    runtime = ThreeTierRuntime(luna, model, execute_python=forbidden,
                               v43_profile=True, evidence_agent=False)
    legacy = Mock(model="qwen3:8b")
    search = WebSearch(opener=feeds)
    app = create_preview_app(legacy, search_service=search, three_tier_runtime=runtime)
    app.testing = True
    with app.test_client() as client:
        assert client.get("/", base_url=ORIGIN).status_code == 200
        session_info = client.get("/api/neral/preview/session", base_url=ORIGIN)
        assert session_info.status_code == 200
        headers = {"Origin": ORIGIN, "X-Neral-CSRF": session_info.get_json()["csrf"]}
        response = client.post(CHAT_ROUTE, base_url=ORIGIN, headers=headers,
                               json={"tier": "Obsidian", "prompt": prompt})
        status = client.get(CHAT_ROUTE + "/status", base_url=ORIGIN)
    assert response.status_code == 200, response.get_json()
    assert status.status_code == 200
    assert feeds.calls[0] == ("www.bing.com", "/news/search")
    assert ("news.google.com", "/rss/search") in feeds.calls
    return response.get_json(), status.get_json()["web_search"], fetch, model, contexts


def _fixture_stamps():
    feed = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=1)
    return feed, feed - timedelta(days=2)


@pytest.mark.parametrize("prompt,title,url,paragraphs,publisher_date,answer_fact", [
    (
        "Was gibt es Neues am Flughafen MUC?",
        "Flughafen München veröffentlicht Test-Flugplan",
        "https://airport-news.example.org/muenchen/flugplan-update",
        (
            "Der Flughafen München kündigt im Beispielbericht genau 14 zusätzliche Wochenverbindungen "
            "an. Die fiktiven Verbindungen werden in den Testflugplan aufgenommen und einzeln ausgewiesen.",
            "Der Flughafen München erläutert die Testdaten in einem getrennten Abschnitt für die "
            "Reisenden. Angaben zu Abflugzeiten und Fluggesellschaften sind im Beispiel noch nicht enthalten.",
        ),
        True, "14 zusätzliche Wochenverbindungen",
    ),
    (
        "Latest news about coral reefs",
        "Coral reefs survey reports new observations",
        "https://science-news.example.org/science/coral-reefs-survey",
        (
            "The fictional coral reefs survey recorded 18 monitoring sites. Researchers used a fixed "
            "sampling plan to visit selected habitats and catalog conditions in the surrounding waters.",
            "The coral reefs report also describes the independent review of the observations. Each site "
            "was documented in the synthetic dataset, which accompanied the report as a separate table.",
        ),
        False, "18 monitoring sites",
    ),
], ids=("muc-publisher-date", "coral-reefs-feed-date"))
def test_preview_news_requires_read_original_and_preserves_date_origin(
    monkeypatch, prompt, title, url, paragraphs, publisher_date, answer_fact,
):
    feed, publisher = _fixture_stamps()
    metadata = f'<meta name="pubdate" content="{publisher.isoformat()}">' if publisher_date else ""
    html = metadata + f"<article><h1>{title}</h1>" + "".join(
        f"<p>{part}</p>" for part in paragraphs
    ) + "</article>"
    answer = f"{answer_fact}. Quelle: {url}"
    body, health, fetch, model, contexts = _post_chat(
        monkeypatch, FixtureFeeds(_rss(title, url, feed)), prompt, {"url": url, "markup": html}, answer,
    )

    assert body["web_search_status"] == health["last_status"] == "ok"
    assert body["verified"] is False  # A readable article is not independent truth verification.
    assert answer_fact in body["text"]
    assert url in body["text"]
    fetch.assert_called_once_with(url, prompt, include_markup=True)
    model.respond_chat.assert_called_once()
    [source] = body["web_sources"]
    assert source["url"] == url
    assert source["page_read"] is True
    assert source["article_truncated"] is False
    assert source["article_chars"] == len("\n\n".join(paragraphs))
    assert "page_text" not in source
    expected_stamp = (publisher if publisher_date else feed).isoformat()
    assert source["date_origin"] == ("publisher_metadata" if publisher_date else "feed")
    assert source["published_at"] == expected_stamp
    [context] = contexts
    assert "UNVERTRAUTE WEBQUELLEN" in context
    assert "\n\n".join(paragraphs) in context
    label = "Artikel-Veröffentlichung" if publisher_date else "Veröffentlichung laut Nachrichtenfeed"
    assert f"{label}: {expected_stamp}" in context
    assert FEED_SNIPPET not in context


@pytest.mark.parametrize("reason", ("publisher-date-stale", "only-teaser"))
def test_preview_news_unreadable_original_is_attributed_headline_only(monkeypatch, reason):
    prompt = "Latest news about coral reefs"
    title = "Coral reefs survey reports new observations"
    url = "https://science-news.example.org/science/coral-reefs-survey"
    feed, _ = _fixture_stamps()
    if reason == "publisher-date-stale":
        old = (feed - timedelta(days=30)).isoformat()
        markup = (
            f'<meta name="pubdate" content="{old}"><article><h1>{title}</h1>'
            "<p>The coral reefs survey recorded the temperature and salinity of several sites, "
            "measured on the same dates for all stations using a consistent field protocol.</p>"
            "<p>Field researchers analyzed recorded conditions at each coral reefs study site, "
            "documenting the methods in a separate data file used to review the public report.</p></article>"
        )
    else:
        markup = f"<article><h1>{title}</h1><p>Fictional coral reefs teaser only.</p></article>"
    body, health, fetch, model, contexts = _post_chat(
        monkeypatch, FixtureFeeds(_rss(title, url, feed)), prompt, {"url": url, "markup": markup},
    )

    fetch.assert_called_once_with(url, prompt, include_markup=True)
    model.respond_chat.assert_not_called()
    assert contexts == []
    assert body["web_search_status"] == health["last_status"] == "headlines_only"
    assert body["web_search_error"] == "no_readable_sources"
    assert body["model_generated"] is False
    assert body["verified"] is False
    assert body["safe_to_promote"] is False
    assert "keine geprüfte Zusammenfassung" in body["text"]
    assert title in body["text"]
    assert FEED_SNIPPET not in body["text"]
    [source] = body["web_sources"]
    assert source == {"url": url, "title": title, "published_at": feed.isoformat(),
                      "provider": "Bing News RSS", "page_read": False, "source_kind": "feed_headline"}
