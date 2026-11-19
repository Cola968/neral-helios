"""Bounded retrieval of public search results. Page content is untrusted data."""

import http.client
import re
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, wait
from html.parser import HTMLParser
from ipaddress import ip_address
from urllib.parse import urljoin, urlsplit

from neral_web_search import safe_result_url

MAX_TEXT = 3000
MAX_BYTES = 768 * 1024
TIMEOUT = 4.0
_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="page-reader")
_blocks = {"p", "div", "section", "li", "tr", "br", "h1", "h2", "h3", "dt", "dd"}
_ignored = {"nav", "aside", "footer", "script", "style", "noscript", "svg", "template",
            "iframe", "form", "button", "title", "head"}
_void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "param", "source", "track", "wbr"}


class MainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.all = []
        self.main = []

    def _add(self, data):
        if not any(hidden for _, hidden in self.stack):
            self.all.append(data)
            if any(tag in {"main", "article"} for tag, _ in self.stack):
                self.main.append(data)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        hidden = (tag in _ignored or "hidden" in attributes or
                  attributes.get("aria-hidden") == "true" or
                  attributes.get("role") in {"navigation", "banner", "contentinfo"})
        if tag in _blocks:
            self._add("\n")
        elif tag in {"td", "th"}:
            self._add(" | ")
        if tag not in _void:
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in _blocks:
            self._add("\n")
        for index in range(len(self.stack)-1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self._add(data)

    def text(self, query):
        lines = list(dict.fromkeys(
            line for part in "".join(self.main or self.all).splitlines()
            if (line := " ".join(part.split()))))
        if len("\n".join(lines)) <= MAX_TEXT:
            return "\n".join(lines)
        terms = set(re.findall(r"\b[^\W\d_]{3,}\b", query.casefold())) - {
            "der", "die", "das", "den", "dem", "bitte", "aktuell", "nach", "und", "von", "was"}
        def priority(line):
            lower = line.casefold()
            matches = sum(term in lower for term in terms)
            # Keep numerical table rows and prices near the top even when a
            # table's topic is only stated in its heading.
            measured = bool(re.search(r"\d\s*(?:[%€$£]|(?:[a-z]{2,3})\b)", lower))
            tabular = lower.startswith("|") and any(ch.isdigit() for ch in lower)
            return matches * 2 + (4 if measured or tabular else 0)

        ranked = sorted(range(len(lines)), key=lambda i: (-priority(lines[i]), i))
        selected = set()
        remaining = MAX_TEXT
        for i in ranked:
            if i in selected:
                continue
            if len(lines[i]) + 1 <= remaining:
                selected.add(i)
                remaining -= len(lines[i]) + 1
            if remaining < 80:
                break
        # A single long relevant paragraph still provides an excerpt.
        if not selected or (ranked and len(lines[ranked[0]]) >= MAX_TEXT):
            return lines[ranked[0]][:MAX_TEXT]
        return "\n".join(lines[i] for i in sorted(selected))


class _PinnedConnection(http.client.HTTPSConnection):
    def __init__(self, host, address, timeout):
        super().__init__(host, port=443, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except OSError:
            raw.close()
            raise


def _public_address(host):
    addresses = list(dict.fromkeys(row[4][0] for row in
                                  socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
    if not addresses or any(not ip_address(a).is_global or ip_address(a).is_multicast
                            for a in addresses):
        raise ValueError("Non-public destination")
    return next((a for a in addresses if ':' not in a), addresses[0])


def fetch_and_extract(url, query=""):
    """Read one public HTTPS page, rechecking and pinning every redirect."""
    deadline = time.monotonic() + TIMEOUT
    for _ in range(3):
        url = safe_result_url(url)
        if url is None:
            return None
        target = urlsplit(url)
        connection = None
        try:
            address = _public_address(target.hostname)
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            connection = _PinnedConnection(target.hostname, address, left)
            path = (target.path or "/") + ("?" + target.query if target.query else "")
            connection.request("GET", path, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NERAL-PageReader/3.0)",
                "Accept": "text/html, application/xhtml+xml, text/plain",
                "Accept-Encoding": "identity", "Accept-Language": "de,en;q=0.8",
            })
            sock = connection.sock
            with connection.getresponse() as response:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location:
                        return None
                    url = urljoin(url, location)
                    continue
                media = response.headers.get_content_type()
                if (response.status != 200 or media not in
                        {"text/html", "application/xhtml+xml", "text/plain"} or
                        response.getheader("Content-Encoding", "identity").lower() != "identity"):
                    return None
                data = bytearray()
                while len(data) < MAX_BYTES:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    if sock is not None:
                        sock.settimeout(left)
                    block = response.read1(min(32768, MAX_BYTES-len(data)))
                    if not block:
                        break
                    data.extend(block)
                markup = data.decode(response.headers.get_content_charset() or "utf-8", "replace")
            parser = MainText()
            if media == "text/plain":
                parser.handle_data(markup)
            else:
                parser.feed(markup)
                parser.close()
            text = parser.text(query)
            return {"url": url, "text": text} if text.strip() else None
        except (OSError, ValueError, LookupError, http.client.HTTPException):
            return None
        finally:
            if connection is not None:
                connection.close()
    return None


def read_top_pages(hits, query, *, fetch=fetch_and_extract):
    """Return up to three unique results in rank order within one deadline."""
    sources = {}
    for hit in hits:
        url = safe_result_url(hit.get("url"))
        if url and url not in sources:
            sources[url] = hit
        if len(sources) == 3:
            break
    tasks = {url: _executor.submit(fetch, url, query) for url in sources}
    done, pending = wait(tasks.values(), timeout=TIMEOUT)
    for task in pending:
        task.cancel()
    result = []
    for url, task in tasks.items():
        page = task.result() if task in done else None
        hit = sources[url]
        result.append({"url": page["url"] if page else url,
                       "title": str(hit.get("title", ""))[:160],
                       "snippet": str(hit.get("snippet", ""))[:350],
                       "page_text": page["text"][:MAX_TEXT] if page else "",
                       "page_read": page is not None, "reachable": page is not None,
                       "discovered": True})
    return result


def entity_evidence_excerpt(text, entity):
    """Literal page-text excerpt with a named subject and descriptive facts.

    Never use a search title/snippet, hostname or generated text as evidence.
    The exact returned substring can safely be attached as a citation.
    """
    if not isinstance(text, str) or not isinstance(entity, str) or not entity.strip():
        return None
    name = re.compile(r"(?<!\w)" + re.escape(entity.strip()) + r"(?!\w)", re.I)
    for match in name.finditer(text):
        start = text.rfind("\n", 0, match.start()) + 1
        finish = text.find("\n", match.end())
        finish = len(text) if finish == -1 else finish
        # Include one adjacent line when the name occurs in a heading.
        if finish - start < 35:
            next_end = text.find("\n", finish + 1)
            finish = len(text) if next_end == -1 else next_end
        start = max(start, match.start() - 100)
        finish = min(finish, match.end() + 300)
        excerpt = text[start:finish].strip()
        without_name = name.sub(" ", excerpt)
        facts = re.findall(r"[^\W\d_]{3,}", without_name, re.UNICODE)
        if len(facts) >= 3 and len(excerpt) >= 20:
            return excerpt
    return None


def read_entity_pages(hits, query, entity, *, reader=read_top_pages,
                      max_pages=2, exclude_urls=()):
    """Read ranked HTTPS candidates until an actual page covers *entity*.

    The first three search hits can all be off-topic. Inspect at most nine
    unique candidates in three bounded batches; never trust search previews.
    The supplied reader is the same hardened pinned-TLS reader used normally.
    """
    found = []
    seen = set(exclude_urls)
    candidates = []
    for hit in hits if isinstance(hits, list) else ():
        if not isinstance(hit, dict):
            continue
        url = safe_result_url(hit.get("url"))
        if url and url not in seen:
            seen.add(url)
            candidates.append(hit)
        if len(candidates) == 9:
            break
    for offset in range(0, len(candidates), 3):
        batch = reader(candidates[offset:offset+3], query)
        for page in batch if isinstance(batch, list) else ():
            if not isinstance(page, dict) or page.get("page_read") is not True:
                continue
            if not safe_result_url(page.get("url")):
                continue
            if entity_evidence_excerpt(page.get("page_text"), entity) is None:
                continue
            found.append(page)
            if len(found) == max_pages:
                return found
        # A relevant read page is sufficient; do not burn retries on more sites.
        if found:
            break
    return found
