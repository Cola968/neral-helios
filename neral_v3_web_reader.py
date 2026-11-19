"""NERAL V3: bounded, no-key public-HTTPS page reader for opted-in web searches.

Only result URLs already discovered by a user-requested web search qualify.
Never forward credentials/cookies, follow redirects, use proxies or contact
non-public DNS answers. External page text is UNTRUSTED DATA, not instructions.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from ipaddress import ip_address
import http.client
import re
import socket
import ssl
from urllib.parse import urlsplit
from neral_web_search import safe_result_url

MAX_PAGE_BYTES = 320_000
MAX_PAGE_CHARS = 3000
MAX_PAGES = 3
TIMEOUT_SECONDS = 3.5
_OMIT = frozenset({"script", "style", "noscript", "nav", "header", "footer",
                    "aside", "form", "svg", "canvas", "menu", "iframe"})
_LINE_TAGS = frozenset({"p", "div", "br", "li", "h1", "h2", "h3", "h4",
                        "section", "tr", "td", "th", "article", "main"})

class _MainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.active_main = 0
        self.body = []
        self.main = []
        self.count = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _OMIT:
            self.skip += 1
        if self.skip:
            return
        if tag in ("article", "main"):
            self.active_main += 1
        if tag in _LINE_TAGS:
            self.body.append(" ")
            if self.active_main:
                self.main.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _OMIT:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag in _LINE_TAGS:
            self.body.append(" ")
            if self.active_main:
                self.main.append(" ")
        if tag in ("main", "article"):
            self.active_main = max(0, self.active_main - 1)

    def handle_data(self, data):
        if self.skip or self.count > 35_000:
            return
        self.count += len(data)
        self.body.append(data)
        if self.active_main:
            self.main.append(data)

def extract_main_text(markup: bytes | str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Extract readable main/article text; fall back to visible body text."""
    if not isinstance(markup, (bytes, str)) or not 1 <= max_chars <= MAX_PAGE_CHARS:
        return ""
    if isinstance(markup, bytes):
        markup = markup[:MAX_PAGE_BYTES].decode("utf-8", "replace")
    parser = _MainText()
    try:
        parser.feed(markup[:MAX_PAGE_BYTES])
        parser.close()
    except (ValueError, TypeError):
        return ""
    preferred = " ".join("".join(parser.main).split())
    body = " ".join("".join(parser.body).split())
    text = preferred if len(preferred) >= 45 else body
    return text[:max_chars]


def _download_public_html(url: str, timeout: float) -> bytes:
    """DNS pinning + verified TLS. Reject any private/mixed IP response."""
    parsed = urlsplit(url)
    host = parsed.hostname
    if not host or safe_result_url(url) != url:
        raise ValueError("Unsafe URL")
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM,
                                    proto=socket.IPPROTO_TCP)
    ips = {row[4][0] for row in addresses}
    if not ips or any(not ip_address(ip).is_global for ip in ips):
        raise ValueError("Non-public DNS answer")
    address = sorted(ips, key=lambda x: (":" in x, x))[0]

    class _PinnedHTTPS(http.client.HTTPSConnection):
        def connect(self):
            sock = socket.create_connection((address, 443), timeout=timeout)
            try:
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
            except Exception:
                sock.close()
                raise

    conn = _PinnedHTTPS(host, timeout=timeout, context=ssl.create_default_context())
    try:
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        conn.request("GET", target, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NERAL-PublicPageReader/3.0)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "identity",
            "Connection": "close",
        })
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError("Page declined request; redirects not followed")
        media = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        if media not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("Non-HTML document")
        size = response.getheader("Content-Length")
        if size and size.isdecimal() and int(size) > MAX_PAGE_BYTES:
            raise ValueError("Page too large")
        raw = response.read(MAX_PAGE_BYTES + 1)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError("Page too large")
        return raw
    finally:
        conn.close()


def fetch_and_extract(url: str, *, timeout: float = TIMEOUT_SECONDS,
                      fetcher=None) -> dict[str, str]:
    """Read one safe search result. Failure is explicit, never fabricated."""
    safe = safe_result_url(url)
    if safe is None or safe != url:
        return {"url": str(url)[:190], "content": "", "read_status": "rejected"}
    if not isinstance(timeout, (int, float)) or not 0.5 <= timeout <= 4.0:
        raise ValueError("Page timeout out of range")
    retrieve = _download_public_html if fetcher is None else fetcher
    try:
        data = retrieve(safe, timeout)
        if not isinstance(data, (bytes, str)) or len(data) > MAX_PAGE_BYTES:
            raise ValueError("Unexpected page body")
        text = extract_main_text(data)
        return {"url": safe, "content": text,
                "read_status": "read" if text else "empty"}
    except (OSError, ValueError, TimeoutError, ssl.SSLError,
            http.client.HTTPException, UnicodeError):
        return {"url": safe, "content": "", "read_status": "unavailable"}


def read_top_pages(hits: list[dict], *, fetcher=None,
                   limit: int = MAX_PAGES) -> list[dict]:
    """Concurrent top-3 reads, original ranking preserved, no extra site crawl."""
    if not isinstance(hits, list) or not 1 <= limit <= MAX_PAGES:
        raise ValueError("Invalid page list or limit")
    selected = []
    seen = set()
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        safe = safe_result_url(hit.get("url"))
        if safe and safe not in seen:
            selected.append(hit)
            seen.add(safe)
        if len(selected) >= limit:
            break
    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=min(MAX_PAGES, len(selected))) as pool:
        tasks = [pool.submit(fetch_and_extract, hit["url"], fetcher=fetcher)
                 for hit in selected]
        results = [task.result() for task in tasks]
    return [{**hit, **reading} for hit, reading in zip(selected, results)]
