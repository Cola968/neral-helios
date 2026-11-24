"""CPU-only regression tests. No network, model weights or self-hosted runner."""
import io
from threading import Lock
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest
from flask import Flask

from neral_v43_live_web import (route_live_query, search_live_sources,
                                search_diagnostics)
from neral_web_search import WebSearch, WebSearchError


@pytest.mark.parametrize("question,expected", [
    ("Welche Nachrichten gibt es heute zur NASA?", True),
    ("Wer ist derzeit CEO von SpaceX?", True),
    ("In welchem Jahr landete Apollo 11?", True),
    ("Suche nach aktuellen BER-Nachrichten.", True),
    ("Prüfe die Aussage, dass die NASA 1958 gegründet wurde.", True),
    ("Belege die Fakten über die Gründung der NASA.", True),
    ("Warum ist der Himmel blau?", False),
    ("Hallo, wie geht es dir?", False),
    ("Berechne 17 * 23.", False),
    ("Schreibe Python-Code.", False),
    ("Was ist mein Passwort?", False),
    ("Suche im Internet nach meinem Passwort", False),
])
def test_router(question, expected):
    assert bool(route_live_query(question)) is expected


def test_benchmark_and_attachment_guards():
    q = "Wer ist heute CEO von SpaceX?"
    assert route_live_query(q, benchmark=True) is None
    assert route_live_query(q, explicit_query="NASA aktuell",
                            benchmark=True) is None
    assert route_live_query(q, force_sandbox=True) is None
    assert route_live_query(q, attachments=True) is None
    assert route_live_query(q, explicit_query="Meine private Adresse") is None


PAGE = {"url": "https://www.nasa.gov/history/",
        "title": "NASA History",
        "page_read": True,
        "page_text": "NASA was established in 1958. The public history is documented here."}
HITS = [{"url": PAGE["url"], "title": PAGE["title"],
         "snippet": "NASA 1958"}]


def test_context_from_read_pages_only():
    engine = Mock()
    engine.search.return_value = {"provider": "fixture", "results": HITS}
    reader = Mock(return_value=[PAGE])
    out = search_live_sources(engine, "NASA Gründung", reader=reader)
    assert out["status"] == "ok"
    assert "1958" in out["context"]
    assert "UNVERTRAUTE WEBQUELLEN" in out["context"]
    assert out["sources"] == [{"url": PAGE["url"], "title": PAGE["title"],
                               "page_read": True}]
    reader.assert_called_once_with(HITS, "NASA Gründung")


def test_snippets_are_not_evidence_and_empty_fails_closed():
    engine = Mock()
    engine.search.return_value = {"provider": "fixture", "results": HITS}
    result = search_live_sources(engine, "NASA", reader=lambda *_: [
        {"url": PAGE["url"], "title": PAGE["title"],
         "page_read": False, "snippet": "Maybe 1958", "page_text": ""}])
    assert result["status"] == "empty"
    assert result["context"] == ""
    assert not result["sources"]


class FakeResponse:
    def __init__(self, data, mime, status=200):
        self.data = data
        self.status = status
        self.headers = {"Content-Type": mime}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, n):
        return self.data[:n]


DDG = (b'<a class="result__a" href="https://www.nasa.gov/">NASA</a>'
       b'<div class="result__snippet">Space agency</div>')
BING = (b'<rss><channel><item><title>NASA launches</title>'
        b'<link>https://www.nasa.gov/launches/</link>'
        b'<description>Launch details</description></item></channel></rss>')


def test_429_immediate_fallback_and_diagnostic_classification():
    class Provider:
        def __init__(self):
            self.urls = []

        def open(self, req, timeout):
            self.urls.append(req.full_url)
            assert timeout <= 3
            if "duckduckgo" in req.full_url:
                raise HTTPError(req.full_url, 429, "private upstream body",
                                {}, io.BytesIO(b"secret"))
            return FakeResponse(BING, "application/rss+xml")
    provider = Provider()
    search = WebSearch(opener=provider)
    result = search.search("NASA")
    assert result["provider"] == "Bing RSS"
    assert len(result["results"]) == 1
    assert len(provider.urls) == 2
    assert search.status()["last_status"] == "ok"
    assert search.status()["last_error"] is None
    assert "NASA" not in str(search.status())


def test_202_retry_then_parse_success():
    class Provider:
        def __init__(self):
            self.n = 0

        def open(self, req, timeout):
            self.n += 1
            if self.n == 1:
                return FakeResponse(b"challenge", "text/html", status=202)
            return FakeResponse(DDG, "text/html")
    p = Provider()
    with patch("neral_web_search.time.sleep") as sleep:
        found = WebSearch(opener=p).search("NASA")
    assert p.n == 2
    sleep.assert_called_once()
    assert len(found["results"]) == 1


def test_all_providers_fail_does_not_leak_exception_details():
    class Provider:
        def open(self, req, timeout):
            raise HTTPError(req.full_url, 429, "SECRET_ERROR", {},
                            io.BytesIO(b"secret"))
    search = WebSearch(opener=Provider())
    with pytest.raises(WebSearchError) as exc:
        search.search("NASA aktuell")
    assert "SECRET" not in str(exc.value)
    assert search.status()["last_status"] == "unavailable"
    assert search.status()["last_error"] == "upstream_rate_limit"


def test_search_timeout_classified_without_model_answer():
    engine = Mock()
    engine.search.side_effect = TimeoutError("private network info")
    result = search_live_sources(engine, "NASA", reader=Mock())
    assert result["status"] == "timeout"
    assert result["error"] == "timeout"
    assert "private" not in str(result)


@pytest.fixture
def api():
    from neral_three_tier_runtime import (ThreeTierRuntime, LUNA, RAW_9B,
                                          register_three_tier_api)
    model = Mock(model=RAW_9B, loaded=True, adapter_sha256=None)
    luna = Mock(model=LUNA, loaded=True, adapter_sha256=None)
    runtime = ThreeTierRuntime(luna, model, execute_python=Mock(),
                               v43_profile=True, evidence_agent=False)
    runtime._verified_source = Mock(return_value=None)
    search = Mock()
    search.search.return_value = {"provider": "fixture", "results": HITS}
    search.status.return_value = {"backend": "mock",
                                  "last_status": "ok", "last_error": None}
    app = Flask(__name__)
    app.secret_key = b"a" * 32
    register_three_tier_api(app, runtime, busy=Lock(), search_service=search)
    return app.test_client(), model, luna, search


@pytest.mark.parametrize("tier", ["Amber", "Quartz", "Obsidian"])
def test_full_http_live_routing_and_prompt_injection(api, tier):
    client, model, luna, search = api
    active = luna if tier == "Amber" else model
    active.respond_chat.return_value = "NASA wurde 1958 gegründet."
    with patch("neral_page_reader.read_top_pages", return_value=[PAGE]) as reader:
        response = client.post("/api/neral/chat/three-tier", json={
            "tier": tier, "prompt": "Wann wurde die NASA gegründet?"})
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["web_search_status"] == "ok"
    assert data["web_sources"][0]["page_read"] is True
    assert data["verified"] is False
    search.search.assert_called_once()
    reader.assert_called_once()
    active.respond_chat.assert_called()
    # A Mock does not run the actual FrozenQwen35HF tokenization function.
    # The source context is request-local, injected only at model-tokenization.
    from neral_three_tier_runtime import _contextual_model_turns, _WEB_SOURCE_CONTEXT
    token = _WEB_SOURCE_CONTEXT.set("UNVERTRAUTE WEBQUELLEN: NASA was established in 1958.")
    try:
        augmented = _contextual_model_turns([
            {"role": "system", "content": "Trusted guard"},
            {"role": "user", "content": "Wann wurde die NASA gegründet?"}])
        assert augmented[0]["content"] == "Trusted guard"
        assert "1958" in augmented[1]["content"]
        assert "UNVERTRAUTE WEBQUELLEN" in augmented[1]["content"]
    finally:
        _WEB_SOURCE_CONTEXT.reset(token)
    assert _WEB_SOURCE_CONTEXT.get() is None


def test_http_benchmark_disables_search_even_when_explicit(api):
    client, model, luna, search = api
    model.respond_chat.return_value = "Nicht live geprüft."
    with patch("neral_page_reader.read_top_pages") as reader:
        response = client.post("/api/neral/chat/three-tier", json={
            "tier": "Obsidian", "prompt": "Wann wurde die NASA gegründet?",
            "web_query": "NASA Gruendung", "benchmark_isolated": True})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["web_search_status"] == "disabled_benchmark"
    search.search.assert_not_called()
    reader.assert_not_called()
    assert "UNVERTRAUTE WEBQUELLEN" not in str(model.respond_chat.call_args)


@pytest.mark.parametrize("failure,expected", [
    (TimeoutError("secret"), "web_search_timeout"),
    (WebSearchError("private secret"), "web_search_unavailable"),
])
def test_http_fail_closed_no_draft(api, failure, expected):
    client, model, luna, search = api
    search.search.side_effect = failure
    resp = client.post("/api/neral/chat/three-tier", json={
        "tier": "Obsidian", "prompt": "Wann wurde die NASA gegründet?"})
    assert resp.status_code == 503
    assert resp.get_json()["error_code"] == expected
    model.respond_chat.assert_not_called()
    assert "secret" not in str(resp.get_json())


def test_health_is_local_and_surfaces_last_status(api):
    client, model, luna, search = api
    result = client.get("/health")
    assert result.status_code == 503  # startup not loaded in test
    assert result.get_json()["web_search"]["backend"] == "mock"
    assert result.get_json()["web_search"]["last_status"] == "ok"
    search.search.assert_not_called()
    status = client.get("/api/neral/chat/three-tier/status").get_json()
    assert status["v43_profile"] is True
    assert status["web_search"]["configured"] is True


def test_no_search_service_configured_snapshot():
    assert search_diagnostics(None)["configured"] is False
    assert search_diagnostics(None)["live_enabled"] is False


def test_health_updates_when_provider_found_hits_but_pages_unreadable():
    class Provider:
        def open(self, req, timeout):
            return FakeResponse(DDG, "text/html")
    engine = WebSearch(opener=Provider())
    result = search_live_sources(engine, "NASA", reader=lambda *_: [])
    assert result["status"] == "empty"
    assert engine.status()["last_status"] == "empty"
    assert engine.status()["last_error"] == "no_readable_sources"


def test_model_context_never_promotes_remote_text_to_system_or_leaks():
    from neral_three_tier_runtime import _contextual_model_turns, _WEB_SOURCE_CONTEXT
    turns = [{"role": "system", "content": "TRUSTED SYSTEM"},
             {"role": "user", "content": "question"}]
    token = _WEB_SOURCE_CONTEXT.set("UNTRUSTED WEB DATA")
    try:
        augmented = _contextual_model_turns(turns)
        assert augmented[0]["content"] == "TRUSTED SYSTEM"
        assert augmented[1]["content"] == "question\n\nUNTRUSTED WEB DATA"
        assert turns[1]["content"] == "question"
    finally:
        _WEB_SOURCE_CONTEXT.reset(token)
    assert _contextual_model_turns(turns) is turns
