"""Read visible public publisher articles; never decode opaque news IDs or bypass access controls."""
from concurrent.futures import ThreadPoolExecutor, wait
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit
from neral_web_search import safe_result_url, airport_text_matches, recent_publication

_STOP = frozenset("der die das den dem des ein eine und oder im in am zu zum zur auf mit für von als ist sind es an bei nach vor über aus -".split())
_HIDDEN = {"script","style","nav","footer","aside","form","button","noscript","iframe","svg","template","head"}
_VOID = {"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}
_BLOCKED = re.compile(r"(?i)\b(?:nur für abonnenten|jetzt abonnieren um|weiterlesen mit abonnement|subscribe to continue|sign in to continue|access denied|verify you are human)\b")

def _words(text):
    return set(re.findall(r"(?u)[^\W_]+",text.casefold()))-_STOP

def _headline(text):
    return " ".join(str(text).rsplit(" - ",1)[0].split())

def _similar(title,expected):
    wanted=_words(_headline(expected))
    actual=_words(title)
    return len(wanted)>=3 and len(wanted & actual)/len(wanted)>=0.72

def _host(url):
    return (urlsplit(url).hostname or "").removeprefix("www.")

class _VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[]
        self.links=[]
        self.anchor=None
        self.paragraph=None
        self.paragraphs=[]
        self.main_paragraphs=[]
        self.title_parts=[]
        self.heading_parts=[]
        self.meta_title=""
        self.paywall=False

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        hidden=(tag in _HIDDEN or "hidden" in a or a.get("aria-hidden")=="true"
                or "display:none" in a.get("style","").replace(" ","").casefold()
                or a.get("role") in {"navigation","banner","contentinfo"})
        article=(tag in {"article","main"} or a.get("itemprop")=="articleBody"
                 or a.get("role")=="main")
        if tag=="meta":
            key=(a.get("property") or a.get("name") or "").casefold()
            if key in ("og:title","twitter:title") and not self.meta_title:
                self.meta_title=a.get("content","")[:500]
        if tag not in _VOID:
            self.stack.append((tag,hidden,article))
        if tag=="a":
            self.anchor=[a.get("href",""),[]]
        if tag=="p":
            self.paragraph=[[],any(x[2] for x in self.stack)]

    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_data(self,data):
        if any(x[0]=="title" for x in self.stack):
            self.title_parts.append(data)
        if any(x[1] for x in self.stack):
            return
        if any(x[0]=="h1" for x in self.stack):
            self.heading_parts.append(data)
        if self.anchor is not None:
            self.anchor[1].append(data)
        if self.paragraph is not None:
            self.paragraph[0].append(data)
        if _BLOCKED.search(data):
            self.paywall=True

    def handle_endtag(self,tag):
        if tag=="a" and self.anchor is not None:
            url,parts=self.anchor
            if len(self.links)<2000:
                self.links.append((url," ".join(" ".join(parts).split())[:600]))
            self.anchor=None
        if tag=="p" and self.paragraph is not None:
            parts,main=self.paragraph
            p=" ".join(" ".join(parts).split())
            if len(p)>=45:
                self.paragraphs.append(p)
                if main:
                    self.main_paragraphs.append(p)
            self.paragraph=None
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:
                del self.stack[i:]
                break

def _parse(markup):
    if not isinstance(markup,str) or not 1<=len(markup)<=1600000:
        return None
    p=_VisibleHTML()
    p.feed(markup)
    p.close()
    return p

def publisher_links(markup,home,title):
    if not safe_result_url(home):
        return []
    p=_parse(markup)
    if not p:
        return []
    seen=set()
    rows=[]
    for href,label in p.links:
        url=safe_result_url(urljoin(home,href)) if href else None
        if not url or _host(url)!=_host(home) or url in seen or urlsplit(url).path in ("","/"):
            continue
        if _similar(label,title):
            seen.add(url)
            rows.append(url)
        if len(rows)>=3:
            break
    return rows

def article_text(markup):
    p=_parse(markup)
    if not p or p.paywall:
        return None
    title=" ".join(" ".join(p.heading_parts).split()) or p.meta_title or " ".join(" ".join(p.title_parts).split())
    paragraphs=list(dict.fromkeys(p.main_paragraphs or p.paragraphs))
    # Repeated boilerplate, a headline or a teaser alone is not an article.
    body="\n\n".join(paragraphs)
    if len(paragraphs)<2 or len(body)<240 or not title.strip():
        return None
    return {"title":title[:500],"text":body[:24000],"article_chars":len(body),
            "article_truncated":len(body)>24000}

def _read_candidate(page,hit,query,home):
    if not isinstance(page,dict):
        return None
    url=safe_result_url(page.get("url",""))
    if not url or _host(url)!=_host(home) or urlsplit(url).path in ("","/"):
        return None
    body=article_text(page.get("markup",""))
    if not body or not _similar(body["title"],hit["title"]) or not airport_text_matches(query,body["text"]):
        return None
    return {**hit,"url":url,"title":body["title"][:180],"page_text":body["text"],
            "article_chars":body["article_chars"],"article_truncated":body["article_truncated"],
            "page_read":True}

def find_publisher_article(hit,query,*,fetch=None):
    if not isinstance(hit,dict) or not recent_publication(hit.get("published_at")):
        return None
    home=safe_result_url(hit.get("publisher_url",""))
    if not home or not isinstance(hit.get("title"),str):
        return None
    if fetch is None:
        from neral_page_reader import fetch_and_extract
        def fetch(url):
            return fetch_and_extract(url,query,include_markup=True)
    index=fetch(home)
    if isinstance(index,dict) and safe_result_url(index.get("url","")) and _host(index["url"])==_host(home):
        for url in publisher_links(index.get("markup",""),index["url"],hit["title"])[:2]:
            found=_read_candidate(fetch(url),hit,query,home)
            if found:
                return found
    # Follow only ordinary public HTTPS redirects. Do not submit consent
    # forms, set consent cookies, decode opaque IDs or bypass access walls.
    link=safe_result_url(hit.get("url",""))
    if link and urlsplit(link).hostname=="news.google.com":
        return _read_candidate(fetch(link),hit,query,home)
    return None

def discover_publisher_articles(hits,query,*,resolver=find_publisher_article):
    """Try up to ten candidates, four at a time, until three articles survive."""
    import time
    deadline=time.monotonic()+10.0
    pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix="news-publisher")
    rows=[]
    seen=set()
    try:
        for offset in range(0,min(len(hits),10),4):
            left=deadline-time.monotonic()
            if left<=0 or len(rows)>=3:
                break
            futures=[pool.submit(resolver,hit,query) for hit in hits[offset:offset+4]]
            done,_=wait(futures,timeout=min(5.0,left))
            for future in futures:
                if future not in done:
                    future.cancel()
                    continue
                try:
                    row=future.result()
                except (OSError,ValueError,TypeError,RuntimeError):
                    continue
                if isinstance(row,dict) and row.get("url") not in seen:
                    seen.add(row["url"])
                    rows.append(row)
    finally:
        pool.shutdown(wait=False,cancel_futures=True)
    return rows[:3]
