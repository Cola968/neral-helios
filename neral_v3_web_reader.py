"""V3 bounded HTTPS page reader. Retrieved page content is untrusted data."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from html import unescape
import http.client
from ipaddress import ip_address
import re
import socket
import ssl
from urllib.parse import urlsplit

from neral_web_search import safe_result_url

MAX_PAGE_BYTES = 160_000
MAX_PAGE_TEXT = 3_000
MAX_PAGES = 3
TIMEOUT = 3.5

_SKIP = frozenset(("script", "style", "svg", "nav", "footer", "header",
                   "aside", "noscript", "form", "button", "menu", "select",
                   "iframe", "template"))


class _MainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.body = []
        self.main = []
        self.title = []
        self.title_open = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self.title_open = True
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        return

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self.title_open = False
        if tag in self.stack:
            while self.stack:
                current = self.stack.pop()
                if current == tag:
                    break

    def handle_data(self, data):
        if self.title_open:
            self.title.append(data[:200])
        if any(tag in _SKIP for tag in self.stack):
            return
        if not data.strip():
            return
        if sum(len(s) for s in self.body) > MAX_PAGE_TEXT * 4:
            return
        cleaned = re.sub(r"\s+", " ", unescape(data)).strip()
        if cleaned:
            self.body.append(cleaned)
            if "main" in self.stack or "article" in self.stack:
                self.main.append(cleaned)


def extract_main_text(markup: str, limit: int = MAX_PAGE_TEXT) -> str:
    """Extract article/main first; reject non-document noise, bound model input."""
    if type(markup) is not str:
        raise ValueError("HTML expected")
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_TEXT:
        raise ValueError("Invalid text limit")
    parser = _MainText()
    parser.feed(markup[:MAX_PAGE_BYTES])
    parser.close()
    main = re.sub(r"\s+", " ", " ".join(parser.main)).strip()
    body = re.sub(r"\s+", " ", " ".join(parser.body)).strip()
    return (main if len(main) >= 35 else body)[:limit]


def _public_addresses(host: str) -> list[str]:
    entries = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM,
                                  proto=socket.IPPROTO_TCP)
    addresses = {item[4][0] for item in entries}
    if not addresses or any(not ip_address(ip).is_global for ip in addresses):
        return []
    return sorted(addresses, key=lambda ip: (":" in ip, ip))


def fetch_and_extract(url: str, timeout: float = TIMEOUT) -> dict | None:
    """One pinned TLS GET of a public, DNS-validated HTTPS host, no redirects.

    Never use urllib's proxy settings or follow 3xx redirects. Even if DNS
    changes, the socket goes only to the validated public address, while TLS
    verifies the original hostname. Scripts are never executed.
    """
    safe = safe_result_url(url)
    if not safe or not 0.25 <= timeout <= 4.0:
        return None
    host = urlsplit(safe).hostname
    try:
        addresses = _public_addresses(host)
        if not addresses:
            return None
        address = addresses[0]
        target = urlsplit(safe)

        class _PinnedHTTPS(http.client.HTTPSConnection):
            def connect(self):
                sock = socket.create_connection((address, 443), timeout=timeout)
                try:
                    self.sock = self._context.wrap_socket(sock, server_hostname=host)
                except Exception:
                    sock.close()
                    raise

        conn = _PinnedHTTPS(host, timeout=timeout, context=ssl.create_default_context())
        try:
            path = target.path or "/"
            if target.query:
                path += "?" + target.query
            conn.request("GET", path, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NeralPageReader/3.0)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "Accept-Language": "de,en;q=0.8",
            })
            response = conn.getresponse()
            mime = (response.getheader("Content-Type") or "").lower()
            if response.status != 200 or not (
                    "text/html" in mime or "application/xhtml+xml" in mime):
                return None
            raw = response.read(MAX_PAGE_BYTES + 1)
            if len(raw) > MAX_PAGE_BYTES:
                return None
        finally:
            conn.close()
        html = raw.decode("utf-8", errors="replace")
        main_text = extract_main_text(html)
        if len(main_text) < 35:
            return None
        return {"url":safe, "title": "", "text":main_text, "read":True}
    except (OSError, ValueError, TimeoutError, ssl.SSLError,
            http.client.HTTPException, UnicodeError):
        return None


def read_top_pages(hits: list[dict], *, max_pages: int = MAX_PAGES,
                   fetcher=None) -> list[dict]:
    """Read up to three distinct top provider hits in parallel, never inside Docker."""
    if type(hits) is not list or type(max_pages) is not int or not 1 <= max_pages <= 3:
        raise ValueError("Invalid search result collection")
    fetch = fetcher if fetcher is not None else fetch_and_extract
    selected = []
    seen = set()
    for hit in hits:
        if type(hit) is not dict:
            continue
        url = safe_result_url(hit.get("url"))
        if not url or url in seen:
            continue
        selected.append((url, str(hit.get("title", ""))[:120]))
        seen.add(url)
        if len(selected) >= max_pages:
            break
    if not selected:
        return []
    records = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, url, TIMEOUT): i
                   for i, (url, _) in enumerate(selected)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                record = future.result()
            except Exception:
                continue
            if (type(record) is dict and record.get("read") in (None, True)
                    and record.get("url") == selected[i][0]
                    and type(record.get("text")) is str
                    and len(record["text"].strip()) >= 35):
                records[i] = {"url": selected[i][0],
                              "title":selected[i][1],
                              "text": record["text"][:MAX_PAGE_TEXT],
                              "read":True}
    return [records[i] for i in sorted(records)]
