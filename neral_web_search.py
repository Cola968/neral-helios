"""Opt-in no-key web search for all NERAL plans; no external model calls.

Only an explicitly supplied search query is sent to DuckDuckGo HTML. Result
pages are never fetched by the server. Search HTML and snippets are untrusted.
The provider's unofficial HTML interface may change or rate-limit requests.
"""
from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from ipaddress import ip_address
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
MAX_RESPONSE_BYTES = 600_000
MAX_RESULTS = 5
MAX_QUERY_CHARS = 160


class WebSearchError(ValueError):
    """Public-safe validation or search-provider failure."""


class WebSearchInputError(WebSearchError):
    """Invalid user search text."""


class WebSearchRateLimitError(WebSearchError):
    """Local search rate limit exceeded."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise WebSearchError("Die Suchanbieter-Weiterleitung wurde blockiert.")


def fixed_search_opener():
    # Never inherit HTTP(S)_PROXY from the host.
    return build_opener(ProxyHandler({}), NoRedirect())


def safe_result_url(value: str) -> str | None:
    """Extract HTTPS public-domain links, not IPs, localhost or intranet URLs."""
    if not isinstance(value, str) or not 1 <= len(value) <= 2048:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value) or "\\" in value:
        return None
    if re.search(r"%(?:0[0-9a-f]|1[0-9a-f]|7f)", value, re.I):
        return None
    if value.startswith("//duckduckgo.com/l/?"):
        value = "https:" + value
    try:
        parsed = urlsplit(value)
        if parsed.scheme == "https" and parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/l/":
            targets = parse_qs(parsed.query, keep_blank_values=False).get("uddg", [])
            if len(targets) != 1:
                return None
            value = targets[0]
            parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None or parsed.password is not None:
            return None
        if parsed.port not in (None, 443):
            return None
        host = parsed.hostname
        if not host or len(host) > 253 or not re.fullmatch(r"[a-zA-Z0-9.-]+", host):
            return None
        if host.endswith(".") or not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
                                         for part in host.split(".")):
            return None
        if "." not in host or host.endswith((".local", ".localhost", ".internal", ".test", ".invalid")):
            return None
        try:
            ip_address(host)
            return None
        except ValueError:
            pass
        if re.search(r"%(?:0[0-9a-f]|1[0-9a-f]|7f)", value, re.I):
            return None
        return value
    except ValueError:
        return None


def _clean(text: str, length: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:length]


class _ResultsParser(HTMLParser):
    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.items: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.capture: str | None = None
        self.root_tag: str | None = None
        self.buffer: list[str] = []
        self.seen: set[str] = set()

    def _flush(self):
        if not self.current or len(self.items) >= self.limit:
            return
        url = safe_result_url(self.current.get("url", ""))
        title = _clean(self.current.get("title", ""), 180)
        if url and title and url not in self.seen:
            self.items.append({"title": title, "url": url,
                               "snippet": _clean(self.current.get("snippet", ""), 350)})
            self.seen.add(url)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._flush()
            self.current = {"url": attrs.get("href", ""), "title": "", "snippet": ""}
            self.capture = "title"
            self.root_tag = tag
            self.buffer = []
        elif self.current is not None and "result__snippet" in classes:
            self.capture = "snippet"
            self.root_tag = tag
            self.buffer = []

    def handle_data(self, data):
        if self.capture and len(self.buffer) < 40:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == self.root_tag and self.capture:
            if self.current is not None:
                self.current[self.capture] = _clean("".join(self.buffer), 350)
            self.capture = None
            self.root_tag = None
            self.buffer = []

    def finish(self):
        self._flush()
        return self.items


def parse_search_results(markup: str, *, limit: int = MAX_RESULTS) -> list[dict[str, str]]:
    if not isinstance(markup, str) or not 1 <= limit <= MAX_RESULTS:
        raise WebSearchError("Ungültige Suchantwort oder Ergebnisgrenze.")
    parser = _ResultsParser(limit)
    try:
        parser.feed(markup)
        parser.close()
    except (ValueError, TypeError):
        raise WebSearchError("Suchanbieter-Antwort konnte nicht verarbeitet werden.") from None
    return parser.finish()


class WebSearch:
    """A bounded, explicitly requested search. No external API key needed."""
    def __init__(self, *, opener=None, max_per_minute: int = 6):
        if not 1 <= max_per_minute <= 60:
            raise ValueError("Ungültiges Suchlimit.")
        self.opener = opener if opener is not None else fixed_search_opener()
        self.max_per_minute = max_per_minute
        self._hits = deque()
        self._lock = threading.Lock()

    def search(self, query: str) -> dict:
        if not isinstance(query, str) or any(ord(c) < 32 or ord(c) == 127 for c in query):
            raise WebSearchInputError("Bitte nur einen normalen Suchbegriff eingeben.")
        cleaned = _clean(query, MAX_QUERY_CHARS + 1)
        if not 2 <= len(cleaned) <= MAX_QUERY_CHARS:
            raise WebSearchInputError("Suchbegriff muss 2 bis 160 Zeichen lang sein.")
        with self._lock:
            now = time.monotonic()
            while self._hits and now - self._hits[0] >= 60:
                self._hits.popleft()
            if len(self._hits) >= self.max_per_minute:
                raise WebSearchRateLimitError("Zu viele Websuchen. Bitte in einer Minute erneut versuchen.")
            self._hits.append(now)
        req = Request(SEARCH_ENDPOINT + "?q=" + quote_plus(cleaned), headers={
            "User-Agent": "Mozilla/5.0 (compatible; NeralLocalSearch/1.0)",
            "Accept": "text/html", "Accept-Language": "de,en;q=0.8",
        }, method="GET")
        try:
            with self.opener.open(req, timeout=7.0) as response:
                mime = response.headers.get("Content-Type", "")
                if "text/html" not in mime.lower():
                    raise WebSearchError("Suchanbieter hat kein HTML zurückgegeben.")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise WebSearchError("Suchanbieter-Antwort überschreitet das Limit.")
            results = parse_search_results(raw.decode("utf-8", errors="replace"))
        except (HTTPError, URLError, TimeoutError, OSError):
            raise WebSearchError("Websuche nicht erreichbar. Prüfe Internetverbindung oder Suchanbieter.") from None
        return {"query": cleaned, "provider": "DuckDuckGo HTML", "external_request": True,
                "model_generated": False, "results": results,
                "notice": "Nur dieser Suchbegriff wurde an DuckDuckGo übertragen. Externe Treffer sind ungeprüft."}
