"""Optional public official news indices; discovery only, never article evidence."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit
from neral_web_search import safe_result_url, airport_entity_codes, recent_publication

# Public publisher directories, not cached answers or fabricated article links.
OFFICIAL_NEWS_INDEXES = {
    "BER": "https://corporate.berlin-airport.de/de/unternehmen-presse/presseportal/pressemitteilungen.html",
    "MUC": "https://www.munich-airport.de/pressemitteilungen-88963",
}
MONTHS = {name: i for i, names in enumerate(
    ("januar january", "februar february", "märz maerz march", "april",
     "mai may", "juni june", "juli july", "august", "september",
     "oktober october", "november", "dezember december"),1)
    for name in names.split()}


class _Anchors(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = None
        self.rows = []
        self.item = None

    def handle_starttag(self, tag, attrs):
        if tag == "li" and self.item is None:
            self.item = [[], []]
        if tag == "a":
            self.active = [dict(attrs).get("href",""), []]

    def handle_data(self, data):
        if self.active is not None:
            self.active[1].append(data)
        if self.item is not None:
            self.item[0].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.active is not None:
            url, parts = self.active
            text = " ".join(" ".join(parts).split())[:1000]
            if len(self.rows)<1500:
                self.rows.append((url,text))
            if self.item is not None:
                self.item[1].append((url,text))
            self.active = None
        if tag == "li" and self.item is not None:
            parts, links = self.item
            unique = set(url for url,_ in links if url)
            if len(unique)==1 and len(self.rows)<1500:
                self.rows.append((next(iter(unique))," ".join(" ".join(parts).split())[:1000]))
            self.item = None


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


def parse_official_news_index(markup, base_url, *, now=None):
    if not isinstance(markup,str) or len(markup)>800000 or not safe_result_url(base_url):
        return []
    parser = _Anchors()
    parser.feed(markup)
    parser.close()
    host = urlsplit(base_url).hostname
    rows = {}
    for href,text in parser.rows:
        if not href or not text:
            continue
        url = safe_result_url(urljoin(base_url,href))
        if not url or url==base_url:
            continue
        p=urlsplit(url)
        if p.hostname!=host or p.query or p.fragment or p.path==urlsplit(base_url).path:
            continue
        if re.search(r"(?i)\.(?:pdf|png|jpg|zip|jpeg|svg)$",p.path):
            continue
        match=re.search(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)",p.path)
        path_date=_date_parts(*match.groups()) if match else None
        visible=_first_date(text[:400])
        if match and not path_date:
            continue
        if path_date and visible and path_date!=visible:
            continue
        stamp=path_date or visible
        if not stamp or not recent_publication(stamp.isoformat(),now=now):
            continue
        row={"url":url,"title":text[:180],"snippet":text[:350],"published_at":stamp.isoformat(),"date_origin":"official_index"}
        if url not in rows or len(row["title"])>len(rows[url]["title"]):
            rows[url]=row
    return sorted(rows.values(),key=lambda row:(row["published_at"],row["url"]),reverse=True)[:10]


def discover_official_news(query, *, fetch=None, now=None):
    codes=airport_entity_codes(query)
    indices=[OFFICIAL_NEWS_INDEXES[code] for code in codes if code in OFFICIAL_NEWS_INDEXES]
    if not indices:
        return []
    if fetch is None:
        from neral_page_reader import fetch_and_extract
        def fetch(url):
            return fetch_and_extract(url, "", include_markup=True)
    rows=[]
    for index in indices[:2]:
        try:
            page=fetch(index)
        except (OSError,ValueError,TypeError,RuntimeError):
            continue
        if not isinstance(page,dict) or not safe_result_url(page.get("url","")):
            continue
        if urlsplit(page["url"]).hostname!=urlsplit(index).hostname:
            continue
        rows.extend(parse_official_news_index(page.get("markup",""),page["url"],now=now))
    return rows[:10]
