"""Independent public news discovery; the index itself is never evidence.

GDELT DOC 2.0 returns public article URLs and crawl timestamps.
https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
Publication dates are read from the publisher, not inferred from indexing.
No credentials, proxy, paywall bypass, private URL, or benchmark integration.
"""
from datetime import datetime,timezone
from http.client import HTTPException
from html.parser import HTMLParser
import json
import re
import threading
import time
from urllib.parse import quote_plus,urlsplit
from urllib.request import Request
from neral_web_search import (
    direct_news_url,recent_publication,live_topic_matches,live_topic_hits,
    live_topic_requirements,airport_entity_codes,_AIRPORT_ALIASES,
    rewrite_search_query,news_subject,MAX_RESPONSE_BYTES,
)
from neral_news_articles import article_text,discover_publisher_articles

ENDPOINT="https://api.gdeltproject.org/api/v2/doc/doc"
_INDEX_ROWS = 15
_ARTICLE_CANDIDATES = 10
_LOCK=threading.Lock()
_LAST=None

def parse_gdelt_candidates(raw, *, now=None, query=None):
    if not isinstance(raw,str) or len(raw)>MAX_RESPONSE_BYTES:
        return []
    try:
        value=json.loads(raw)
    except (ValueError,TypeError):
        return []
    items=value.get("articles") if isinstance(value,dict) else None
    if not isinstance(items,list):
        return []
    rows=[]
    seen=set()
    for item in items[:_INDEX_ROWS]:
        if not isinstance(item,dict):
            continue
        url=direct_news_url(item.get("url"))
        title=item.get("title")
        if not url or url in seen or not isinstance(title,str) or not title.strip():
            continue
        try:
            stamp=datetime.strptime(item.get("seendate",""),"%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        except (ValueError,TypeError):
            continue
        if not recent_publication(stamp,now=now):
            continue
        row = {"url": url, "title": " ".join(title.split())[:240],
               "indexed_at": stamp, "provider": "GDELT DOC"}
        # Newer but off-topic index hits must not use the candidate budget.
        # The shared topic filter also caps list input, so check each row.
        if query is not None and not live_topic_hits([row], query):
            continue
        seen.add(url)
        rows.append(row)
        if len(rows) == _ARTICLE_CANDIDATES:
            break
    return rows

_ARTICLE_TYPES = frozenset((
    "Article", "NewsArticle", "ReportageNewsArticle", "AnalysisNewsArticle",
    "OpinionNewsArticle", "ReviewNewsArticle", "BackgroundNewsArticle",
    "Report", "BlogPosting", "TechArticle",
))
_JSONLD_LIMIT = 65536


def _article_dates_from_jsonld(raw):
    """Read only declared publication metadata; never dereference the graph."""
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError, TypeError):
        return []
    dates = []
    queue = [data]
    for _ in range(40):
        if not queue:
            break
        node = queue.pop(0)
        if isinstance(node, list):
            queue.extend(node[:20])
        elif isinstance(node, dict):
            kind = node.get("@type")
            if isinstance(kind, str):
                kind = [kind]
            if isinstance(kind, list) and any(
                isinstance(value, str) and
                value.removeprefix("https://schema.org/").removeprefix("http://schema.org/") in _ARTICLE_TYPES
                for value in kind[:10]
            ):
                stamp = node.get("datePublished")
                if isinstance(stamp, str) and len(stamp) <= 64:
                    dates.append(stamp)
            graph = node.get("@graph")
            if isinstance(graph, list):
                queue.extend(graph[:20])
    return dates


class _PublicationMeta(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.values=[]
        self.scripts=[]
        self.ld_active=False
    def handle_starttag(self,tag,attrs):
        a={key:value or "" for key,value in attrs}
        key=(a.get("property") or a.get("name") or a.get("itemprop") or "").casefold()
        if tag == "script":
            self.ld_active = (len(self.scripts) < 4 and
                              a.get("type", "").strip().casefold() == "application/ld+json")
        if tag=="meta" and key in ("article:published_time","datepublished","pubdate","publication_date","dc.date.issued","parsely-pub-date"):
            self.values.append(a.get("content",""))
        if tag=="time" and key=="datepublished":
            self.values.append(a.get("datetime",""))

    def handle_data(self, data):
        if self.ld_active:
            if len(data) <= _JSONLD_LIMIT:
                self.scripts.append(data)
            self.ld_active = False

    def handle_endtag(self, tag):
        if tag == "script":
            self.ld_active = False

def article_publication_provenance(markup):
    """Return (publisher date, conflict), distinguishing absent from contradictory data."""
    if not isinstance(markup,str) or not 1<=len(markup)<=1600000:
        return None, False
    p=_PublicationMeta()
    try:
        p.feed(markup)
        p.close()
    except (ValueError,TypeError,AttributeError,AssertionError,RecursionError):
        return None, False
    dates=[]
    fields = p.values[:12]
    for script in p.scripts:
        fields.extend(_article_dates_from_jsonld(script)[:12])
    for value in fields:
        try:
            value=value.strip()
            date=datetime.fromisoformat(value.replace("Z","+00:00"))
            if date.tzinfo is None:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}",value):
                    continue
                date=date.replace(tzinfo=timezone.utc)
            dates.append(date.astimezone(timezone.utc))
        except (ValueError,TypeError,AttributeError):
            continue
    if not dates:
        return None, False
    if (max(dates)-min(dates)).total_seconds()>86400:
        return None, True
    return min(dates).isoformat(), False

def article_publication_date(markup):
    return article_publication_provenance(markup)[0]


def _host(url):
    return (urlsplit(url).hostname or "").removeprefix("www.")

def read_gdelt_candidate(hit,query,*,fetch=None):
    if not isinstance(hit,dict):
        return None
    url=direct_news_url(hit.get("url"))
    if not url or not recent_publication(hit.get("indexed_at")):
        return None
    if fetch is None:
        from neral_page_reader import fetch_and_extract
        def fetch(address):
            return fetch_and_extract(address,query,include_markup=True)
    result=fetch(url)
    if not isinstance(result,dict):
        return None
    final=direct_news_url(result.get("url"))
    if not final or _host(final)!=_host(url) or urlsplit(final).path in ("","/"):
        return None
    markup=result.get("markup","")
    published=article_publication_date(markup)
    article=article_text(markup)
    if not article or not recent_publication(published):
        return None
    if not live_topic_matches(query,article["text"][:8000]):
        return None
    if not live_topic_hits([{"url":final,"title":article["title"],"snippet":article["text"][:500]}],query):
        return None
    return {"url":final,"title":article["title"],"snippet":article["text"][:350],
            "published_at":published,"date_origin":"publisher_metadata",
            "indexed_at":hit["indexed_at"],"provider":"GDELT DOC","page_read":True,
            "page_text":article["text"],"article_chars":article["article_chars"],
            "article_truncated":article["article_truncated"]}

def _query(query):
    codes=airport_entity_codes(query)
    if codes:
        groups=[tuple(_AIRPORT_ALIASES.get(c,()))+(c,) for c in codes]
    else:
        groups=list(live_topic_requirements(query))
    if not groups:
        groups=[tuple([w]) for w in re.findall(r"(?u)[^\W_]+",news_subject(rewrite_search_query(query)))[:4] if len(w)>=3]
    parts=[]
    for group in groups[:4]:
        phrases=[]
        for term in group:
            clean=" ".join(re.findall(r"(?u)[^\W_]+",term))
            if len(clean)>=3 and clean not in phrases:
                phrases.append(clean)
        if not phrases:
            continue
        # GDELT searches translated English text; retain original vocabulary
        # and include available language equivalents instead of guessed URLs.
        chosen=phrases[:2]
        for term in phrases:
            if term.casefold() in ("artificial intelligence","munich airport","german parliament","berlin airport"):
                if term not in chosen:
                    chosen.append(term)
                break
        quoted=['"'+x+'"' for x in chosen[:3]]
        part="("+ " OR ".join(quoted)+")" if len(quoted)>1 else quoted[0]
        if len(" ".join(parts+[part]))>145:
            break
        parts.append(part)
    return " ".join(parts)

def discover_public_news(query,*,opener,on_provider=None):
    """At most one fixed-endpoint request, then bounded independent public reads."""
    global _LAST
    term=_query(query)
    if not term:
        return []
    with _LOCK:
        now=time.monotonic()
        if _LAST is not None and now-_LAST<5.0:
            return []
        _LAST=now
    if callable(on_provider):
        on_provider("GDELT DOC")
    url=ENDPOINT+"?query="+quote_plus(term)+f"&mode=ArtList&format=json&maxrecords={_INDEX_ROWS}&timespan=7d&sort=DateDesc"
    try:
        request=Request(url,headers={"Accept":"application/json","User-Agent":"NERAL-Public-News/4.3"},method="GET")
        with opener.open(request,timeout=4.0) as response:
            if getattr(response,"status",200) not in (None,200):
                return []
            if "json" not in str(response.headers.get("Content-Type","")).casefold():
                return []
            raw=response.read(MAX_RESPONSE_BYTES+1)
        if len(raw)>MAX_RESPONSE_BYTES:
            return []
        candidates = parse_gdelt_candidates(raw.decode("utf-8", errors="replace"), query=query)
    except (OSError,HTTPException,ValueError,TypeError,RuntimeError):
        return []
    return discover_publisher_articles(candidates,query,resolver=read_gdelt_candidate)
