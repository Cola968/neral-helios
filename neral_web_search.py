"""Opt-in no-key web search for all NERAL plans; no external model calls.

Only an explicitly supplied query is sent to DuckDuckGo HTML and (if needed)
Bing RSS. Result pages are never fetched. Both unofficial providers can block
or change their output; search hits are untrusted data, never instructions.
"""
from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from html import unescape
from xml.etree import ElementTree
from ipaddress import ip_address
import re
import socket
import ssl
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
BING_ENDPOINT = "https://www.bing.com/search"
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


# Link checks use pinned public IPs, authenticated TLS and no implicit redirects.
# Search previews are not treated as independent verification of page content.
def _probe_https(url: str, address: str, method: str = "HEAD") -> tuple[int, str | None]:
    import http.client
    import socket
    import ssl
    from urllib.parse import urlsplit
    target = urlsplit(url)

    class _PinnedHTTPS(http.client.HTTPSConnection):
        def connect(self):
            raw = socket.create_connection((address, 443), timeout=2.5)
            try:
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
            except Exception:
                raw.close()
                raise

    connection = _PinnedHTTPS(target.hostname, timeout=2.5, context=ssl.create_default_context())
    try:
        path = target.path or "/"
        if target.query:
            path += "?" + target.query
        connection.request(method, path, headers={
            "User-Agent": "NERAL-LinkCheck/1.0", "Accept": "text/html, application/xhtml+xml, application/pdf",
            "Range": "bytes=0-255" if method == "GET" else "bytes=0-0",
        })
        response = connection.getresponse()
        return response.status, response.getheader("Location")
    finally:
        connection.close()


# Minimal website favicon retrieval, used only when the user has opted into
# web search and the UI displays a verified public HTTPS source.
# Browser pages never contact a third-party favicon aggregator.
MAX_FAVICON_BYTES = 65536


def _request_favicon(host: str, address: str) -> tuple[bytes, str]:
    """Download /favicon.ico over TLS with DNS pinned to a validated public IP.

    No redirects, no HTML parsing, no cookies and no arbitrary URL or port.
    """
    import http.client

    class _PinnedIconHTTPS(http.client.HTTPSConnection):
        def connect(self):
            raw = socket.create_connection((address, 443), timeout=2.5)
            try:
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
            except Exception:
                raw.close()
                raise

    conn = _PinnedIconHTTPS(host, timeout=2.5, context=ssl.create_default_context())
    try:
        conn.request("GET", "/favicon.ico", headers={
            "User-Agent": "NERAL-Favicon/1.0", "Accept": "image/x-icon,image/png,image/webp,image/jpeg,image/gif",
        })
        response = conn.getresponse()
        if response.status != 200:
            return b"", ""
        if response.length is not None and response.length > MAX_FAVICON_BYTES:
            return b"", ""
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        return response.read(MAX_FAVICON_BYTES + 1), content_type
    finally:
        conn.close()


def fetch_public_favicon(host: str) -> tuple[bytes, str] | None:
    """Get small raster/icon bytes from a public HTTPS hostname, never internal IPs.

    The image type is determined from bytes, not an untrusted Content-Type.
    A missing favicon produces a normal local UI fallback; never fetch SVG.
    """
    import http.client
    from urllib.parse import urlsplit
    if not isinstance(host, str) or not re.fullmatch(r"[a-zA-Z0-9.-]{4,253}", host):
        return None
    if not safe_result_url("https://" + host + "/favicon.ico"):
        return None
    try:
        entries = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM,
                                      proto=socket.IPPROTO_TCP)
        ips = {row[4][0] for row in entries}
        if not ips or any(not ip_address(ip).is_global for ip in ips):
            return None
        address = sorted(ips, key=lambda ip: (":" in ip, ip))[0]
        data, _declared_type = _request_favicon(host, address)
        if not data or len(data) > MAX_FAVICON_BYTES:
            return None
        if data.startswith(b"\x00\x00\x01\x00"):
            return data, "image/x-icon"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return data, "image/png"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return data, "image/gif"
        if data.startswith(b"\xff\xd8\xff"):
            return data, "image/jpeg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return data, "image/webp"
        return None
    except (OSError, ValueError, TimeoutError, http.client.HTTPException):
        return None


def verify_public_https_url(value: str) -> str | None:
    """Return a publicly reachable HTTPS URL, or nothing. Not content verification.

    Never contact IP addresses supplied by an untrusted result directly, DNS
    must resolve only to globally routable addresses, and the checked socket is
    pinned to the validated address while the TLS hostname remains authenticated.
    """
    import socket
    from urllib.parse import urljoin, urlsplit
    url = safe_result_url(value)
    for _ in range(3):  # at most two redirects
        if not url:
            return None
        host = urlsplit(url).hostname
        try:
            entries = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM,
                                         proto=socket.IPPROTO_TCP)
            ips = {entry[4][0] for entry in entries}
            if not ips or any(not ip_address(ip).is_global for ip in ips):
                return None
            # Use the first resolved public socket address; avoid DNS rebinding.
            address = sorted(ips, key=lambda ip: (":" in ip, ip))[0]
            status, location = _probe_https(url, address)
            if status == 405:  # Some real sites reject HEAD.
                status, location = _probe_https(url, address, "GET")
            if 200 <= status < 300:
                return url
            if status in (301, 302, 303, 307, 308) and location:
                url = safe_result_url(urljoin(url, location))
                continue
            return None
        except (OSError, TimeoutError, ValueError, ssl.SSLError):
            return None
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


def parse_bing_rss(xml: str, *, limit: int = MAX_RESULTS) -> list[dict[str, str]]:
    """Parse untrusted Bing RSS without DTDs, external entity expansion or fetching."""
    if not isinstance(xml, str) or not 1 <= limit <= MAX_RESULTS or len(xml) > MAX_RESPONSE_BYTES:
        raise WebSearchError("Ungültige RSS-Antwort.")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)", xml, flags=re.IGNORECASE):
        raise WebSearchError("RSS mit externen Entitäten nicht erlaubt.")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        raise WebSearchError("Ungültige RSS-Antwort.") from None
    if root.tag != "rss":
        raise WebSearchError("Kein RSS-Suchfeed empfangen.")
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in root.findall("./channel/item")[:60]:
        url = safe_result_url(item.findtext("link", default=""))
        title = _clean(unescape(re.sub(r"<[^>]*>", "", item.findtext("title", default=""))), 180)
        snippet = _clean(unescape(re.sub(r"<[^>]*>", "", item.findtext("description", default=""))), 350)
        if url and title and url not in seen:
            output.append({"title": title, "url": url, "snippet": snippet})
            seen.add(url)
        if len(output) >= limit:
            break
    return output


def compact_fact_snippets(hits: list[dict], *, limit: int = 3) -> list[str]:
    """Bounded *unverified search excerpts*, not fetched-page facts.

    Every line uses at most 72 title + 38 host + 120 excerpt UTF-8 bytes
    plus delimiters; at most 3 lines remain well below 800 UTF-8 bytes.
    Search text is data, never an instruction or independently verified fact.
    """
    if type(hits) is not list or type(limit) is not int or not 1 <= limit <= 3:
        raise WebSearchError("Ungültige Trefferliste oder Evidenzgrenze.")

    def clipped(raw: str, size: int) -> str:
        if type(raw) is not str:
            return ""
        plain = _clean(unescape(re.sub(r"<[^>]*>", " ", raw)), 3000)
        return plain.encode("utf-8")[:size].decode("utf-8", "ignore").rstrip()

    output: list[str] = []
    seen: set[str] = set()
    for hit in hits[:MAX_RESULTS]:
        if type(hit) is not dict:
            continue
        url = safe_result_url(hit.get("url", ""))
        if not url or url in seen:
            continue
        title = clipped(hit.get("title", ""), 72)
        snippet = clipped(hit.get("snippet", ""), 120)
        host = clipped(urlsplit(url).hostname or "", 38)
        if not title or not snippet or not host:
            continue
        output.append(f"[{len(output)+1}] {title} | {host} | {snippet}")
        seen.add(url)
        if len(output) >= limit:
            break
    # Conservative UTF-8 byte ceiling; byte count is not a model-token count.
    if len("\n".join(output).encode("utf-8")) >= 800:
        raise WebSearchError("Suchauszüge überschreiten das Budget.")
    return output


class WebSearch:
    """A bounded, explicitly requested search. No external API key needed."""
    def __init__(self, *, opener=None, max_per_minute: int = 6):
        if not 1 <= max_per_minute <= 60:
            raise ValueError("Ungültiges Suchlimit.")
        self.opener = opener if opener is not None else fixed_search_opener()
        self.max_per_minute = max_per_minute
        self._hits = deque()
        self._lock = threading.Lock()

    def search(self, query: str, on_provider=None) -> dict:
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
        contacted = []

        def request_page(url: str, accept: str, expected_type: str) -> str:
            # Only fixed providers below. Never follow redirect or use proxies.
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NeralLocalSearch/1.0)",
                "Accept": accept, "Accept-Language": "de,en;q=0.8",
            }, method="GET")
            with self.opener.open(req, timeout=7.0) as response:
                mime = response.headers.get("Content-Type", "").lower()
                if expected_type not in mime:
                    raise WebSearchError("Suchanbieter lieferte unerwarteten Inhaltstyp.")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise WebSearchError("Suchanbieter-Antwort überschreitet das Limit.")
            return raw.decode("utf-8", errors="replace")

        contacted.append("DuckDuckGo HTML")
        if on_provider is not None:
            on_provider("DuckDuckGo HTML")
        ddg_problem = False
        try:
            markup = request_page(SEARCH_ENDPOINT + "?q=" + quote_plus(cleaned), "text/html", "text/html")
            results = parse_search_results(markup)
        except (HTTPError, URLError, TimeoutError, OSError, WebSearchError) as exc:
            if isinstance(exc, WebSearchError) and "überschreitet" in str(exc):
                raise  # Oversized responses are a hard failure.
            ddg_problem = True
            results = []

        provider = "DuckDuckGo HTML"
        if not results:
            contacted.append("Bing RSS")
            if on_provider is not None:
                on_provider("Bing RSS")
            rss_url = BING_ENDPOINT + "?q=" + quote_plus(cleaned) + "&format=rss"
            try:
                xml = request_page(rss_url, "application/rss+xml, application/xml, text/xml", "xml")
                results = parse_bing_rss(xml)
                provider = "Bing RSS"
            except (HTTPError, URLError, TimeoutError, OSError, WebSearchError):
                if ddg_problem:
                    raise WebSearchError("Websuche bei beiden Anbietern nicht verfügbar.") from None
                results = []

        return {"query": cleaned, "provider": provider, "providers_contacted": contacted,
                "external_request": True, "model_generated": False, "results": results,
                "notice": "Nur dieser Suchbegriff wurde an " + " und ".join(contacted) +
                          " übertragen. Externe Treffer sind ungeprüft."}
