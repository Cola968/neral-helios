"""CPU-only, bounded Kiwix retrieval. No memory, model, or internet imports."""
from __future__ import annotations

import datetime
from html.parser import HTMLParser
import http.client
import os
import re
import threading
import time
from urllib.parse import urlencode, urljoin, urlsplit
import xml.etree.ElementTree as ET

BASE = "http://127.0.0.1:9691"
BUDGET = 0.45
MAX_BYTES = 512 * 1024
_SLOT = threading.BoundedSemaphore(1)
_STOP = set("was ist sind wer war waren wie funktioniert funktioniert erkläre erklaere mir bitte definition von der die das ein eine eines und oder im in zu zum zur den des dem auf für of the a an what is are who was how does explain define tell me about does".split())
_FRESH = re.compile(r"\b(?:heute|gestern|morgen|aktuell\w*|neuest\w*|derzeit|momentan|jetzt|news|nachrichten|live|latest|today|yesterday|tomorrow|current\w*|recent\w*|breaking|this week|preis\w*|kurs\w*|wetter|flugplan\w*|öffnungszeiten|präsident\w*|bundeskanzler\w*|ministerpräsident\w*|ceo|prime minister|president|exchange rate|weather|stock price|version)\b", re.I)
_START = re.compile(r"^\s*(?:was (?:ist|sind|war|bedeutet)|wer (?:ist|war)|wie (?:funktioniert|entsteht)|erkläre|erklaere|definiere|what (?:is|are|was|does)|who (?:is|was)|how does|explain|define|wann (?:war|wurde)|wo (?:liegt|liegt)|welche|wie hoch ist)\b", re.I)
_PRIVATE = re.compile(r"\b(?:mein(?:e|er|en|em|es)?|unser(?:e|er|en|em|es)?|my|our|passwort|password|api.key|token|adresse|address)\b", re.I)
_TASK = re.compile(r"\b(?:berechne|rechne|calculate|solve|programmiere|schreibe|implementiere|debugge|code|übersetze|translate)\b|https?://|[\x60]", re.I)


def _words(value):
    return {w for w in re.findall(r"[^\W_]+", value.casefold()) if len(w) > 2 and w not in _STOP}


def fresh_question(question):
    year = datetime.date.today().year
    return bool(_FRESH.search(question) or any(
        int(y) >= year - 1 for y in re.findall(r"\b20\d{2}\b", question)))


def eligible_question(question):
    return (isinstance(question, str) and 2 <= len(question.strip()) <= 160
            and all(ch.isprintable() for ch in question)
            and not _PRIVATE.search(question) and not _TASK.search(question)
            and bool(_START.search(question) or fresh_question(question))
            and bool(_words(question)))


def enabled():
    return os.environ.get("NERAL_V42_OFFLINE_RAG", "0") == "1"


def _get(path, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    # Direct loopback connection: no proxy, no redirects, no external URL.
    conn = http.client.HTTPConnection("127.0.0.1", 9691, timeout=remaining)
    try:
        conn.request("GET", path, headers={"Accept-Encoding": "identity"})
        response = conn.getresponse()
        if response.status != 200:
            raise OSError("Kiwix HTTP status")
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES or time.monotonic() > deadline:
            raise TimeoutError("Kiwix response budget")
        return data
    finally:
        conn.close()


def _article_path(link):
    url = urlsplit(urljoin(BASE + "/search", link))
    if (url.scheme != "http" or url.netloc != "127.0.0.1:9691" or
            not url.path.startswith("/content/") or url.query or url.fragment or
            "\\" in url.path or re.search(r"(?:^|/)\.\.(?:/|$)|%2e|%5c", url.path, re.I)):
        return None
    return url.path


class _Lead(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.lead = []
        self.box = []
        self.after_heading = False

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h3"):
            self.after_heading = True
        if tag not in ("area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"):
            self.stack.append((tag, dict(attrs).get("class") or ""))
        elif tag == "br":
            self.handle_data(" ")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.handle_data(" ")

    def handle_endtag(self, tag):
        if tag in ("p", "tr", "td", "th"):
            self.handle_data(" ")
        for idx in range(len(self.stack) - 1, -1, -1):
            if self.stack[idx][0] == tag:
                del self.stack[idx:]
                break

    def handle_data(self, data):
        tags = {tag for tag, _ in self.stack}
        classes = " ".join(cls for _, cls in self.stack).casefold()
        if tags & {"script", "style", "nav", "footer", "header", "sup", "noscript"} or any(
                x in classes for x in ("navbox", "hatnote", "reference", "sidebar", "metadata")):
            return
        if "table" in tags and "infobox" in classes and tags & {"td", "th"}:
            if len(self.box) < 200:
                self.box.append(data)
        elif "p" in tags and not self.after_heading and "table" not in tags:
            if len(self.lead) < 300:
                self.lead.append(data)

    def context(self):
        lead = re.sub(r"\s+", " ", "".join(self.lead)).strip()[:1400]
        box = re.sub(r"\s+", " ", "".join(self.box)).strip()[:500]
        return lead + ("\nInfobox: " + box if box else "")


def _lookup(question, deadline):
    terms = _words(question)
    query = " ".join(w for w in re.findall(r"[^\W_]+", question)
                     if w.casefold() in terms)
    if not query:
        return []
    lang = "eng" if re.match(r"\s*(?:what|who|how|explain|define)\b", question, re.I) else "deu"
    params = {"pattern": query, "books.filter.lang": lang, "format": "xml", "pageLength": 8}
    data = _get("/search?" + urlencode(params), deadline)
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        return []
    root = ET.fromstring(data)
    candidates = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()[:180]
        path = _article_path(item.findtext("link") or "")
        matches = len(terms & _words(title))
        if path and matches and matches / len(terms) >= 0.5:
            candidates.append((matches, title, path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    results = []
    seen = set()
    for _, title, path in candidates:
        if time.monotonic() >= deadline or len(results) == 3:
            break
        if path in seen:
            continue
        seen.add(path)
        try:
            article = _get(path, deadline)
            parser = _Lead()
            parser.feed(article.decode("utf-8", errors="replace"))
            context = parser.context()
        except (OSError, ValueError, http.client.HTTPException):
            continue
        if len(context) >= 80:
            results.append({"title": title, "url": BASE + path,
                            "context": context, "source_type": "offline_zim",
                            "book": path.split("/")[2]})
    return results


def local_wiki_search(question):
    """Return up to three relevant extracts; total caller wait <=450 ms (nominal)."""
    if not isinstance(question, str) or not 2 <= len(question) <= 160 or fresh_question(question):
        return []
    if not _SLOT.acquire(blocking=False):
        return []
    deadline = time.monotonic() + BUDGET
    done = threading.Event()
    result = []
    def worker():
        try:
            result.extend(_lookup(question, deadline))
        except (OSError, ValueError, ET.ParseError, http.client.HTTPException):
            pass
        finally:
            _SLOT.release()
            done.set()
    try:
        threading.Thread(target=worker, name="neral-kiwix-cpu", daemon=True).start()
    except RuntimeError:
        _SLOT.release()
        return []
    return result if done.wait(max(0, deadline - time.monotonic())) else []


def lookup_with_fallback(question, web, render, *, local=None):
    """Keep the original Web V4 callback intact; never use stale ZIM for fresh facts."""
    if fresh_question(question):
        return web()
    try:
        hits = (local or local_wiki_search)(question)
    except (OSError, ValueError):
        hits = []
    return render(hits) if hits else web()
