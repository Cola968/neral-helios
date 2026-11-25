"""Bounded retrieval of public search results. Page content is untrusted data."""

import http.client
import re
import socket
import ssl
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, wait
from html.parser import HTMLParser
from ipaddress import ip_address
from urllib.parse import urljoin, urlsplit

from neral_web_search import (safe_result_url, numbered_query_subject,
                              search_topic_terms, topical_search_hit,
                              ranked_topic_hits, airport_entity_codes,
                              airport_text_matches)

MAX_TEXT = 4600
MAX_BYTES = 768 * 1024
MAX_MARKUP_BYTES = 1600000


_YEAR_CLIP_CODES = frozenset((
    "not_year_query", "year_absent_from_page", "year_preserved_in_extract",
    "year_lost_in_clip",
))


def _bounded_year_clip_code(query, original, excerpt):
    """Only a four-way code; never retain or emit the web page or query."""
    if not isinstance(query, str) or re.search(r"(?i)\b(?:year|jahr)\b", query) is None:
        return "not_year_query"
    year_pattern = r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)"
    if re.search(year_pattern, original) is None:
        return "year_absent_from_page"
    return ("year_preserved_in_extract" if re.search(year_pattern, excerpt)
            else "year_lost_in_clip")
TIMEOUT = 4.0
CONNECT_TIMEOUT = 2.5
_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="page-reader")
_blocks = {"p", "div", "section", "li", "tr", "br", "h1", "h2", "h3", "dt", "dd", "figcaption"}
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
        self.infobox = []
        self._infobox_depth = None
        self.year_clip_code = "not_year_query"

    def _add(self, data):
        if not any(hidden for _, hidden in self.stack):
            self.all.append(data)
            if any(tag in {"main", "article"} for tag, _ in self.stack):
                self.main.append(data)
            # Real infobox tables can precede <main>; never lose their literal
            # cell text when a page also provides an article/main element.
            if self._infobox_depth is not None and len(self.stack) >= self._infobox_depth:
                self.infobox.append(data)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if (tag == "table" and self._infobox_depth is None and
                re.search(r"(?i)\binfobox\b", str(attributes.get("class") or "") +
                          " " + str(attributes.get("id") or ""))):
            self._infobox_depth = len(self.stack) + 1
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
        if self._infobox_depth is not None and len(self.stack) < self._infobox_depth:
            self._infobox_depth = None

    def handle_data(self, data):
        self._add(data)

    def text(self, query):
        # The extracted main text is still the canonical page body. Prepend
        # literal infobox text solely when a separate <main> would omit it.
        parts = (self.infobox + self.main) if self.main and self.infobox else (self.main or self.all)
        lines = list(dict.fromkeys(
            line for part in "".join(parts).splitlines()
            if (line := " ".join(part.split()))))
        full = "\n".join(lines)
        if len(full) > MAX_TEXT:
            # A whole HTML <p> can be longer than the entire excerpt.
            # Retain intact literal sentences so a date at its end remains
            # eligible for selection instead of clipping the paragraph head.
            expanded = []
            for line in lines:
                if len(line) > 600 and re.search(r"[.!?]\s+", line):
                    expanded.extend(part for part in
                                    re.split(r"(?<=[.!?])\s+", line) if part)
                else:
                    expanded.append(line)
            lines = list(dict.fromkeys(expanded))
            full = "\n".join(lines)
        if len(full) <= MAX_TEXT:
            self.year_clip_code = _bounded_year_clip_code(query, full, full)
            return full
        terms = set(re.findall(r"\b[^\W\d_]{3,}\b", query.casefold())) - {
            "der", "die", "das", "den", "dem", "bitte", "aktuell", "nach", "und", "von", "was"}
        topic = re.compile(r"\b(?:rendering|browser|layout|engine|entwickler|"
                           r"schöpfer|creator|developed|created|designed|"
                           r"developer|kernel|infobox)\b", re.I)
        history = re.compile(r"\b(?:initially|formerly|previously|originally|"
                             r"historically|used to|früher|ursprünglich)\b", re.I)
        def priority(line, index):
            lower = line.casefold()
            matches = sum(term in lower for term in terms)
            measured = bool(re.search(r"\d\s*(?:[%€$£]|(?:[a-z]{2,3})\b)", lower))
            tabular = lower.startswith("|") and any(ch.isdigit() for ch in lower)
            lead = 4 if index < 24 and matches else 0
            subject = 5 if topic.search(line) else 0
            historical_penalty = 7 if history.search(line) else 0
            return matches*2 + lead + subject + (4 if measured or tabular else 0) - historical_penalty

        # Preserve literal neighboring event/year HTML lines as one atomic
        # candidate. Ranking separated lines independently may discard the
        # year even though both sentences are present on the read page.
        year_pairs = {}
        for i in range(len(lines)-1):
            joined = lines[i] + "\n" + lines[i+1]
            lower = joined.casefold()
            if (len(joined) <= 600 and
                    re.search(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)", joined) and
                    re.search(r"(?i)\b(?:land\w*|mondland\w*|moon\s+landing)\b", joined) and
                    any(term in lower for term in terms)):
                year_pairs[i] = (i, i+1)
                year_pairs[i+1] = (i, i+1)
        # Keep an entire three-line event/context/date passage together.
        # Without this, the ranking can retain the event and the date while
        # silently dropping the intervening sentence needed for attribution.
        for i in range(len(lines)-2):
            joined = "\n".join(lines[i:i+3])
            first = lines[i]
            mission = re.findall(r"(?i)\bapollo\s+\d{1,3}\b", first)
            missions = set(re.findall(r"(?i)\bapollo\s+\d{1,3}\b", joined))
            if (len(joined) <= 600 and len(mission) == 1 and
                    {name.casefold() for name in missions} == {mission[0].casefold()} and
                    re.search(r"(?i)\b(?:land\w*|moon\s+landing)\b", first) and
                    re.search(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)", lines[i+2]) and
                    any(term in joined.casefold() for term in terms)):
                trio = (i, i+1, i+2)
                for j in trio:
                    year_pairs[j] = trio
        ranked = sorted(range(len(lines)),
                        key=lambda i: (-priority(lines[i], i) -
                                       (20 if i in year_pairs else 0), i))
        # Reserve a bounded amount of the excerpt for one or more *literal*
        # event/year windows before ranking undated keyword-heavy paragraphs.
        # The later source verifier still decides whether the window proves
        # the requested event; extraction itself makes no historical claim.
        dated_groups = []
        target = re.search(r"(?i)\bapollo\s+(\d{1,3})\b", query)
        requested_mission = ("apollo " + str(int(target.group(1)))) if target else None
        year_pattern = r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)"
        event_pattern = re.compile(
            r"(?iu)\b(?:land\w*|mondland\w*|moon\s+landing|"
            r"touchdown|founded|founded\s+in|gegründet|"
            r"launched|gestartet|ereignis\w*)\b")
        for i in range(len(lines)):
            for width in (1, 2, 3):
                indices = tuple(range(i, min(i + width, len(lines))))
                if len(indices) != width:
                    continue
                joined = "\n".join(lines[j] for j in indices)
                if len(joined) > 750 or not re.search(year_pattern, joined):
                    continue
                if not event_pattern.search(joined):
                    continue
                names = set("apollo " + str(int(x)) for x in
                            re.findall(r"(?i)\bapollo\s+(\d{1,3})\b", joined))
                if requested_mission:
                    if names != {requested_mission}:
                        continue
                elif sum(term in joined.casefold() for term in terms) < 2:
                    continue
                dated_groups.append(indices)
                break
        selected = set()
        remaining = MAX_TEXT
        reserved = min(1200, MAX_TEXT // 3)
        priority_order = []
        for indices in dated_groups:
            missing = [j for j in indices if j not in selected]
            space = sum(len(lines[j]) + 1 for j in missing)
            if space <= remaining and space <= reserved:
                selected.update(missing)
                priority_order.extend(missing)
                remaining -= space
                reserved -= space
            if reserved < 80:
                break
        # A query's named Apollo mission + landing + year in one literal line
        # is more useful than dozens of high-keyword, undated table-of-contents
        # lines. Promote the real source span without creating or rewriting it.
        # Reject mixed missions: another Apollo mission cannot provide its year.
        requested = re.search(r"(?i)\bapollo\s+(\d{1,3})\b", query)
        preferred = []
        if requested and re.search(r"(?i)\b(?:year|jahr)\b", query):
            subject = "apollo " + str(int(requested.group(1)))
            for i, line in enumerate(lines):
                named = set(name.casefold() for name in re.findall(
                    r"(?i)\bapollo\s+\d{1,3}\b", line))
                if (named == {subject} and
                        re.search(r"(?i)\b(?:land\w*|mondland\w*|moon\s+landing)\b", line) and
                        re.search(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?!\d)", line) and
                        len(line) <= 600):
                    preferred.append(i)
        for i in preferred[:3]:
            if i in selected:
                continue
            space = len(lines[i]) + 1
            if space <= remaining:
                selected.add(i)
                priority_order.append(i)
                remaining -= space
        # Preserve a literal year-bearing line when no event/year window
        # was found. This is not a source citation: the downstream verifier
        # will reject unrelated years and non-contiguous events.
        if (not priority_order and re.search(r"(?i)\b(?:year|jahr)\b", query)):
            for i, line in enumerate(lines):
                if len(line) <= 600 and re.search(year_pattern, line):
                    space = len(line) + 1
                    if space <= remaining:
                        selected.add(i)
                        priority_order.append(i)
                        remaining -= space
                    break
        for i in ranked:
            indices = year_pairs.get(i, (i,))
            missing = [j for j in indices if j not in selected]
            space = sum(len(lines[j]) + 1 for j in missing)
            if space <= remaining:
                selected.update(missing)
                remaining -= space
            if remaining < 80:
                break
        # A single long relevant paragraph still provides an excerpt.
        if not selected:
            excerpt = _complete_text_clip(lines[ranked[0]], MAX_TEXT)
        else:
            # Place quoted, dated source lines first, then retain source order
            # for remaining context. All characters still come from the page.
            first = list(dict.fromkeys(i for i in priority_order if i in selected))
            others = [i for i in sorted(selected) if i not in first]
            excerpt = "\n".join(lines[i] for i in first + others)
        self.year_clip_code = _bounded_year_clip_code(query, full, excerpt)
        return excerpt




def _complete_text_clip(value, limit=MAX_TEXT):
    """Never terminate a source excerpt in the middle of a proper name."""
    if not isinstance(value, str):
        return ""
    if len(value) <= limit:
        return value
    chunk = value[:limit]
    endpoints = [i+1 for i,ch in enumerate(chunk) if ch in ".!?\n" and i>=limit//3]
    return chunk[:endpoints[-1]].strip() if endpoints else ""




def select_web_paragraphs(page_text, entity, topic, *, limit=3):
    """Select 2-3 coherent literal page paragraphs, not isolated regex words.

    The page text is provided only by the bounded HTTPS reader; no search
    snippets, page titles, model predictions, or invented text can enter this
    context. Oversized paragraphs are split only at complete sentence ends.
    """
    if (not isinstance(page_text,str) or not isinstance(entity,str) or
            not isinstance(topic,str) or not page_text.strip() or not entity.strip()
            or len(page_text)>120000 or not isinstance(limit,int)
            or not 1<=limit<=3):
        return []
    initial=[part.strip() for part in page_text.splitlines() if part.strip()]
    units=[]
    for p in initial:
        if len(p)<=600:
            units.append(p)
            continue
        sentences=re.split(r"(?<=[.!?])\s+",p)
        group=""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(sentence)>600:
                if group:
                    units.append(group);group=""
                continue
            if group and len(group)+len(sentence)+1>600:
                units.append(group);group=sentence
            else:
                group=(group+" "+sentence).strip()
        if group:
            units.append(group)
    if not units:
        return []
    words=[word.casefold() for word in re.findall(r"[\w-]{4,}",topic)
           if word.casefold() not in {"browser","wiki","wikipedia","official",
                                     "technologie","webseite","quelle","pages",
                                     "documentation","nach","vergleiche","vergleich"}]
    name=entity.casefold()
    current=("currently","today","now","uses","is","verwendet","nutzt","heute")
    historical=("initially","formerly","originally","earlier","historically",
                "ursprünglich","früher","vorläufer","forked from")
    def rank(i):
        low=units[i].casefold()
        entity_match=name in low
        topical=sum(1 for word in words if word in low)
        temporal=1 if any(w in low for w in current) else 0
        old=1 if any(w in low for w in historical) else 0
        return (int(entity_match)*4+topical*2+temporal-old*4,
                -i)
    ordered=sorted(range(len(units)),key=rank,reverse=True)
    selected=[]
    for i in ordered:
        p=units[i]
        if len(p)<35 or (name not in p.casefold() and
                        not any(word in p.casefold() for word in words)):
            continue
        if p not in selected:
            selected.append(p)
        if len(selected)>=limit:
            break
    return selected

def evidence_sentence_windows(page_text, entity, topic, *, max_windows=10):
    """Literal source sentences with adjacent context; no dangling final text."""
    if (not isinstance(page_text,str) or not page_text.strip() or
            len(page_text)>120000 or not isinstance(entity,str) or
            not entity.strip() or not isinstance(topic,str)):
        return []
    spans = []
    position = 0
    boundaries = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
    for match in boundaries.finditer(page_text):
        sentence=page_text[position:match.start()].strip()
        if sentence and (sentence.endswith((".","!","?")) or "\n" in match.group(0)):
            spans.append(sentence)
        position=match.end()
    last=page_text[position:].strip()
    if last.endswith((".","!","?")):
        spans.append(last)
    subject=re.compile(r"(?<!\w)"+re.escape(entity.strip())+r"(?!\w)",re.I)
    topic_regex={
        "engine":r"\b(?:browser|layout|rendering|engine|engineered)\b",
        "creator":r"\b(?:created|developed|designed|invented|creator|developer|schöpfer|entwickelt)\b",
        "kernel":r"\b(?:kernel|betriebssystem)\b",
    }
    t=topic.casefold()
    kind=("engine" if "engine" in t else
          "creator" if any(x in t for x in ("creator","entwickler","schöpfer","develop"))
          else "kernel" if "kernel" in t else "")
    prop=re.compile(topic_regex[kind],re.I) if kind else None
    rows=[]
    for i,sentence in enumerate(spans):
        if not subject.search(sentence):
            continue
        if prop and not prop.search(sentence):
            if i+1>=len(spans) or not prop.search(spans[i+1]):
                continue
        surrounding=" ".join(spans[max(0,i-1):min(len(spans),i+2)])
        historical=bool(re.search(
            r"\b(?:initially|formerly|previously|originally|historically|früher|ursprünglich)\b",
            sentence,re.I))
        current=bool(re.search(
            r"\b(?:now|currently|today|since|uses|nutzt|verwendet|heute)\b",sentence,re.I))
        rows.append({"sentence":sentence,"context":surrounding[:1000],
                     "priority":(3 if current else 0)-(6 if historical else 0),
                     "historical":historical})
    return sorted(rows,key=lambda x:-x["priority"])[:max_windows]


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


def fetch_and_extract(url, query="", *, include_markup=False):
    """Read one public HTTPS page, rechecking and pinning every redirect."""
    deadline = time.monotonic() + (6.0 if include_markup else TIMEOUT)
    cap = MAX_MARKUP_BYTES if include_markup else MAX_BYTES
    visited=set()
    for _ in range(6 if include_markup else 3):
        url = safe_result_url(url)
        if url is None or url in visited:
            return None
        visited.add(url)
        target = urlsplit(url)
        if target.hostname=="consent.google.com":
            return None  # no automated consent or retry loop
        connection = None
        try:
            address = _public_address(target.hostname)
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            connection = _PinnedConnection(target.hostname, address, min(CONNECT_TIMEOUT, left))
            path = (target.path or "/") + ("?" + target.query if target.query else "")
            connection.request("GET", path, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NERAL-PageReader/3.0)",
                "Accept": ("text/html, application/xhtml+xml, application/rss+xml, application/atom+xml, application/xml, text/xml, text/plain" if include_markup else "text/html, application/xhtml+xml, text/plain"),
                "Accept-Encoding": "gzip, identity", "Accept-Language": "de-DE,de;q=0.9,en;q=0.6",
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
                encoding=response.getheader("Content-Encoding","identity").lower().strip()
                allowed_media={"text/html","application/xhtml+xml","text/plain"}
                if include_markup:
                    allowed_media.update({"application/rss+xml","application/atom+xml","application/xml","text/xml"})
                if (response.status != 200 or media not in allowed_media or
                        encoding not in ("identity","gzip")):
                    return None
                data = bytearray()
                while len(data) <= cap:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    if sock is not None:
                        sock.settimeout(left)
                    block = response.read1(min(32768, cap+1-len(data)))
                    if not block:
                        break
                    data.extend(block)
                if len(data)>cap:
                    return None
                if encoding=="gzip":
                    inflater=zlib.decompressobj(16+zlib.MAX_WBITS)
                    raw=inflater.decompress(bytes(data),cap+1)
                    if len(raw)>cap or not inflater.eof or inflater.unconsumed_tail:
                        return None
                    data=bytearray(raw)
                markup = data.decode(response.headers.get_content_charset() or "utf-8", "replace")
            parser = MainText()
            if media not in ("text/html","application/xhtml+xml"):
                parser.handle_data(markup)
            else:
                parser.feed(markup)
                parser.close()
            text = parser.text(query)
            result = {"url": url, "text": text, "year_clip_code": parser.year_clip_code}
            if include_markup:
                result["markup"] = markup
            return result if text.strip() or include_markup else None
        except (OSError, ValueError, LookupError, http.client.HTTPException, zlib.error):
            return None
        finally:
            if connection is not None:
                connection.close()
    return None


def read_top_pages(hits, query, *, fetch=fetch_and_extract,
                   wiki_fallback=None):
    """Read at most three selected HTTPS pages; filter ambiguous names first.

    In focused numbered-name searches, metadata selects *which pages to
    open*, but only the actual HTTPS body can establish relevance or a quote.
    If fewer than two relevant pages survive, consult the genuine Wikipedia
    search provider. Search snippets never enter page_text or the verifier.
    """
    subject = numbered_query_subject(query)
    focused = bool(subject and search_topic_terms(query))
    airport_focus = bool(airport_entity_codes(query))
    selected_hits = ranked_topic_hits(hits, query) if (focused or airport_focus) else (
        hits if isinstance(hits, list) else [])
    seen = set()
    tried = 0

    def fetch_batch(batch):
        nonlocal tried
        sources = {}
        for hit in batch:
            if not isinstance(hit, dict):
                continue
            url = safe_result_url(hit.get("url"))
            if url and url not in seen:
                seen.add(url)
                sources[url] = hit
            if len(sources) >= 3 - tried:
                break
        tried += len(sources)
        tasks = {url: _executor.submit(fetch, url, query) for url in sources}
        done, pending = wait(tasks.values(), timeout=TIMEOUT)
        for task in pending:
            task.cancel()
        rows = []
        for url, task in tasks.items():
            try:
                page = task.result() if task in done else None
            except (OSError, ValueError, TypeError, RuntimeError):
                page = None
            hit = sources[url]
            if focused:
                # A themed-looking *search result* can still redirect to an
                # unrelated page. Require actual read text to match both the
                # complete numbered subject and an independent context signal.
                if not page or not topical_search_hit({
                    "title":str(page.get("text", ""))[:160],
                    "snippet":str(page.get("text", ""))[160:1700],
                    "url":page.get("url") or url,
                }, query):
                    continue
            if airport_focus and (not page or not airport_text_matches(query, page.get("text", ""))):
                continue
            rows.append({
                "url": page["url"] if page else url,
                **({"discovered_url":url} if hit.get("published_at") else {}),
                "title": str(hit.get("title", ""))[:160],
                "snippet": str(hit.get("snippet", ""))[:350],
                "page_text": (
                    (_complete_text_clip(page["text"]) or page["text"][:MAX_TEXT])
                    if page else ""),
                "year_clip_code": (
                    page.get("year_clip_code") if page and
                    page.get("year_clip_code") in _YEAR_CLIP_CODES
                    else "not_available"),
                "page_read": page is not None,
                "reachable": page is not None,
                "discovered": True,
            })
        return rows

    result = fetch_batch(selected_hits)
    if not focused or len(result) >= 2 or tried >= 3:
        return result

    # The existing real DE/EN Wikipedia HTTPS search is a discovery step,
    # not a source of asserted answers. Stay within the three-page read cap.
    # Tests may inject a deterministic fake search, never fake page evidence.
    def wikipedia_candidates():
        if wiki_fallback is not None:
            return wiki_fallback(subject) if callable(wiki_fallback) else []
        from neral_web_search import WebSearch, WebSearchError
        engine = WebSearch()
        found = []
        for language in ("de", "en"):
            try:
                found.extend(engine.search_wikipedia(
                    subject, language=language))
            except (WebSearchError, OSError, ValueError):
                continue
            if len(ranked_topic_hits(found, query)) >= 2:
                break
        return found

    wiki_hits = ranked_topic_hits(wikipedia_candidates(), query)
    fresh = [hit for hit in wiki_hits
             if safe_result_url(hit.get("url")) not in seen]
    if fresh:
        result.extend(fetch_batch(fresh[:3-tried]))
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
                      max_pages=2, exclude_urls=(), predicate=None):
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
        if len(candidates) == 16:
            break
    if predicate is not None:
        # Prefer published reference articles and technical documentation
        # for retrieval, but accept factual claims ONLY from page_read text.
        def source_rank(hit):
            url = safe_result_url(hit.get("url")) or ""
            host = (urlsplit(url).hostname or "").casefold()
            if host=="developer.mozilla.org":
                return 0
            if host=="en.wikipedia.org":
                return 1
            if host=="wikipedia.org" or host.endswith(".wikipedia.org"):
                return 2
            if host.startswith(("docs.","developer.","documentation.")):
                return 3
            return 3
        candidates.sort(key=source_rank)
    for offset in range(0, len(candidates), 3):
        batch = reader(candidates[offset:offset+3], query)
        for page in batch if isinstance(batch, list) else ():
            if not isinstance(page, dict) or page.get("page_read") is not True:
                continue
            if not safe_result_url(page.get("url")):
                continue
            if entity_evidence_excerpt(page.get("page_text"), entity) is None:
                continue
            # V3.3 topic-specific read: keep scanning up to nine actual
            # fetched pages until the requested property has literal evidence.
            if predicate is not None and not predicate(page):
                continue
            found.append(page)
            if len(found) == max_pages:
                return found
        # A relevant read page is sufficient; do not burn retries on more sites.
        if found:
            break
    return found
