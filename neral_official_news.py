"""Optional public official news indices; discovery only, never article evidence."""
from datetime import datetime, timezone
from html.parser import HTMLParser
from itertools import zip_longest
import re
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree
from neral_web_search import (safe_result_url, direct_news_url, airport_entity_codes,
                              live_topic_hits, recent_publication)

# Public publisher directories, not cached answers or fabricated article links.
OFFICIAL_NEWS_INDEXES = {
    "BER": "https://corporate.berlin-airport.de/de/unternehmen-presse/presseportal/pressemitteilungen.html",
    "MUC": "https://www.munich-airport.de/pressemitteilungen-88963",
}
# Complement the operator's own press page with the published feed of the
# responsible transport authority. These are indices, not stored articles.
OFFICIAL_NEWS_FEEDS = {
    "MUC": "https://www.stmb.bayern.de/rss/pm/atom.xml",
}
_ATOM = "{http://www.w3.org/2005/Atom}"
MONTHS = {name: i for i, names in enumerate(
    ("januar january", "februar february", "märz maerz march", "april",
     "mai may", "juni june", "juli july", "august", "september",
     "oktober october", "november", "dezember december"),1)
    for name in names.split()}


_ARTICLE_ITEMTYPES = {scheme + "://schema.org/" + kind
                      for scheme in ("https", "http") for kind in ("Article", "NewsArticle")}


class _ArticleCard:
    """Bounded fields of one explicitly typed, non-nested publication card."""
    def __init__(self, tag, start_row, outer_item):
        self.tag = tag
        self.depth = 1
        self.start_row = start_row
        self.outer_item = outer_item
        self.outer_item_lengths = tuple(map(len, outer_item)) if outer_item else ()
        self.published = None
        self.link = None
        self.permalink = None
        self.ambiguous = False

    def date(self, value):
        value = value or ""
        if self.published is not None and self.published != value:
            self.ambiguous = True
        self.published = value

    def anchor(self, href, text, marked):
        if href:
            if self.link is not None and self.link != href:
                self.ambiguous = True
            self.link = href
        if marked and href and text:
            self.permalink = (href, text)


class _Anchors(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = None
        self.rows = []
        self.item = None
        self.card = None

    def _row(self, url, text, published=None):
        if len(self.rows) < 1500:
            self.rows.append((url, text, published))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        properties = (attrs.get("itemprop") or "").split()
        if self.card is None:
            if (tag in {"article", "div", "li"} and "itemscope" in attrs and
                    attrs.get("itemtype") in _ARTICLE_ITEMTYPES and "hidden" not in attrs):
                self.card = _ArticleCard(tag, len(self.rows), self.item)
        else:
            self.card.depth += tag == self.card.tag
            if "itemscope" in attrs:
                self.card.ambiguous = True
        if self.card is not None and "datePublished" in properties:
            machine_attr = {"time": "datetime", "meta": "content"}.get(tag)
            self.card.date(attrs.get(machine_attr) if machine_attr else None)
        if tag == "li" and self.item is None:
            self.item = [[], []]
        if tag == "a":
            self.active = [attrs.get("href", ""), [], "url" in properties]

    def handle_data(self, data):
        if self.active is not None:
            self.active[1].append(data)
        if self.item is not None:
            self.item[0].append(data)

    def handle_endtag(self, tag):
        card = self.card
        if tag == "a" and self.active is not None:
            url, parts, marked = self.active
            text = " ".join(" ".join(parts).split())[:1000]
            self._row(url, text)
            if self.item is not None:
                self.item[1].append((url,text))
            if card is not None:
                card.anchor(url, text, marked)
            self.active = None
        if tag == "li" and self.item is not None:
            parts, links = self.item
            unique = set(url for url,_ in links if url)
            if len(unique)==1:
                self._row(next(iter(unique)), " ".join(" ".join(parts).split())[:1000])
            self.item = None
        if card is not None and tag == card.tag:
            card.depth -= 1
            if not card.depth:
                self._finish_card()

    def _finish_card(self, *, complete=True):
        card = self.card
        self.card = None
        if card.published is None:
            return
        # An explicit date in the card controls its own candidates. Never
        # backfill an invalid or conflicting publication with a rendered date.
        del self.rows[card.start_row:]
        self.item = card.outer_item
        for parts, length in zip(self.item or (), card.outer_item_lengths):
            del parts[length:]
        if complete and not card.ambiguous and self.active is None and card.permalink:
            self._row(*card.permalink, card.published)
        self.active = None

    def close(self):
        super().close()
        if self.card is not None:
            self._finish_card(complete=False)


def _machine_publication(value):
    if re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", value):
        return _date_parts(*value.split("-"))
    if not re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"
                        r"(?::[0-9]{2}(?:\.[0-9]{1,6})?)?(?:Z|[+-][0-9]{2}:[0-5][0-9])", value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _date_parts(year,month,day):
    try:
        return datetime(int(year),int(month),int(day),tzinfo=timezone.utc)
    except (ValueError,TypeError):
        return None


def _first_date(text):
    match = re.search(r"(?<!\d)(\d{1,2})\.\s*(\d{1,2})\.\s*(20\d{2})(?!\d)",text)
    if match:
        d,m,y=match.groups()
        return _date_parts(y,m,d)
    match = re.search(r"(?i)(?<!\d)(\d{1,2})\.?\s+([a-zäöü]+)\s+(20\d{2})(?!\d)",text)
    if match:
        d,m,y=match.groups()
        return _date_parts(y,MONTHS.get(m.casefold()),d)
    return None


def parse_official_news_index(markup, base_url, *, now=None, query=None, stage_counter=None):
    """Keep at most ten recent same-host candidates, with topic filtering first.

    Optional row counts describe parser boundaries before deduplication and
    caps; they are not counts of confirmed articles or transport errors.
    """
    if not isinstance(markup,str) or len(markup)>800000 or not safe_result_url(base_url):
        return []
    parser = _Anchors()
    parser.feed(markup)
    parser.close()
    host = urlsplit(base_url).hostname
    rows = {}
    for href,text,declared in parser.rows:
        if not href or not text:
            continue
        try:
            url = safe_result_url(urljoin(base_url,href))
        except ValueError:
            continue
        if not url or url==base_url:
            continue
        p=urlsplit(url)
        if p.hostname!=host or p.query or p.fragment or p.path==urlsplit(base_url).path:
            continue
        if re.search(r"(?i)\.(?:pdf|png|jpg|zip|jpeg|svg)$",p.path):
            continue
        if stage_counter is not None:
            stage_counter("official_html_link_rows")
        match=re.search(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)",p.path)
        path_date=_date_parts(*match.groups()) if match else None
        visible=_first_date(text[:400])
        if match and not path_date:
            continue
        if path_date and visible and path_date!=visible:
            continue
        if declared is not None:
            stamp = _machine_publication(declared)
            if not stamp or (path_date and path_date.date() != stamp.date()):
                continue
        else:
            stamp=path_date or visible
        if not stamp:
            continue
        if stage_counter is not None:
            stage_counter("official_html_dated_rows")
        if not recent_publication(stamp.isoformat(),now=now):
            continue
        if stage_counter is not None:
            stage_counter("official_html_fresh_rows")
        row={"url":url,"title":text[:180],"snippet":text[:350],"published_at":stamp.isoformat(),"date_origin":"official_index"}
        # Out-of-topic releases must not consume the publisher's ten slots.
        # Check individually because the shared relevance filter also caps its input.
        if query is not None and not live_topic_hits([row], query):
            continue
        if stage_counter is not None:
            stage_counter("official_html_topic_rows")
        if url not in rows or len(row["title"])>len(rows[url]["title"]):
            rows[url]=row
    return sorted(rows.values(),key=lambda row:(row["published_at"],row["url"]),reverse=True)[:10]


def parse_official_atom(markup, feed_url, query, *, now=None):
    """Discover explicitly dated same-host public articles from an official Atom feed."""
    if (not isinstance(markup, str) or len(markup) > 800000 or
            not safe_result_url(feed_url) or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", markup, re.I)):
        return []
    try:
        root = ElementTree.fromstring(markup)
    except (ElementTree.ParseError, ValueError):
        return []
    if root.tag != _ATOM + "feed":
        return []

    host = urlsplit(feed_url).hostname
    rows = []
    seen = set()
    for entry in root.findall(_ATOM + "entry")[:40]:
        title = " ".join(entry.findtext(_ATOM + "title", "").split())[:180]
        published = entry.findtext(_ATOM + "published", "").strip()
        if not title or not recent_publication(published, now=now):
            continue
        for link in entry.findall(_ATOM + "link")[:8]:
            if link.get("rel", "alternate") != "alternate":
                continue
            href = link.get("href", "").strip()
            if not href:
                continue
            try:
                parts = urlsplit(urljoin(feed_url, href))
            except ValueError:
                continue
            if parts.hostname != host or parts.scheme not in ("http", "https"):
                continue
            # Some authoritative feeds publish legacy http: links. Request
            # only HTTPS on the feed's own authority; never fetch plaintext.
            url = direct_news_url(parts._replace(scheme="https").geturl())
            if not url or url == feed_url or url in seen:
                continue
            row = {"url": url, "title": title, "snippet": title,
                   "published_at": published, "date_origin": "official_feed"}
            if not live_topic_hits([row], query):
                continue
            rows.append(row)
            seen.add(url)
            break
        if len(rows) >= 10:
            break
    return rows


def discover_official_news(query, *, fetch=None, now=None, stage_counter=None):
    codes=airport_entity_codes(query)
    indices = [(OFFICIAL_NEWS_INDEXES[code], False) for code in codes if code in OFFICIAL_NEWS_INDEXES]
    indices.extend((OFFICIAL_NEWS_FEEDS[code], True) for code in codes if code in OFFICIAL_NEWS_FEEDS)
    if not indices:
        return []
    if fetch is None:
        from neral_page_reader import fetch_and_extract
        def fetch(url):
            options = ({"on_failure": lambda category: stage_counter("official_fetch_" + category)}
                       if stage_counter is not None else {})
            return fetch_and_extract(url, "", include_markup=True, **options)
    by_index = []
    seen_urls = set()
    for index, atom in indices[:2]:
        kind = "official_atom" if atom else "official_html"
        if stage_counter is not None:
            stage_counter(kind + "_attempts")
        try:
            page=fetch(index)
        except (OSError,ValueError,TypeError,RuntimeError):
            continue
        if not isinstance(page,dict) or not safe_result_url(page.get("url","")):
            continue
        if urlsplit(page["url"]).hostname!=urlsplit(index).hostname:
            continue
        markup = page.get("markup", "")
        if stage_counter is not None and isinstance(markup, str) and markup.strip():
            stage_counter(kind + "_bodies")
        if atom:
            candidates = parse_official_atom(markup, page["url"], query, now=now)
        else:
            candidates = parse_official_news_index(markup, page["url"], now=now, query=query,
                                                   stage_counter=stage_counter)
        unique = []
        for row in candidates:
            # Keep the first configured index's provenance for shared URLs.
            if row["url"] not in seen_urls:
                seen_urls.add(row["url"])
                unique.append(row)
        by_index.append(unique)
    # Keep each index's ranking while preventing the first index from taking
    # all ten slots before the second can contribute a distinct candidate.
    rows = [row for group in zip_longest(*by_index) for row in group if row is not None][:10]
    if stage_counter is not None:
        stage_counter("official_candidates", len(live_topic_hits(rows, query)))
    return rows
