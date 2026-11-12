"""Public provider diagnostics using a fixed nonpersonal airport query."""
from __future__ import annotations
from urllib.parse import urlencode
import json
from neral_web_search import fixed_search_opener, WebSearchError
from xml.etree import ElementTree

QUERY="Flughafen Frankfurt"
def inspect(name,url,accept):
    from urllib.request import Request
    opener=fixed_search_opener()
    try:
        req=Request(url,headers={"User-Agent":"NERAL public search diagnostic/1.0",
             "Accept":accept,"Accept-Language":"de,en;q=0.8"},method="GET")
        with opener.open(req,timeout=8) as response:
            mime=response.headers.get("Content-Type","").split(";")[0].strip().lower()
            raw=response.read(400_000)
            status=response.status
        detail={"name":name,"http":status,"mime":mime,"bytes":len(raw)}
        try:
            if mime in ("application/json","application/json; charset=utf-8"):
                obj=json.loads(raw)
                hits=obj.get("query",{}).get("search",[]) if "query" in obj else obj.get("pages",[])
                detail.update(count=len(hits),titles=[h.get("title","")[:90] for h in hits[:3]])
            elif "xml" in mime or "rss" in mime:
                tree=ElementTree.fromstring(raw)
                hits=tree.findall(".//item")
                detail.update(count=len(hits),titles=[n.findtext("title","")[:90] for n in hits[:3]])
            else:
                detail.update(first_80=raw[:80].decode("utf8",errors="replace"))
        except (ValueError,ElementTree.ParseError,TypeError):
            detail["parse_error"]=True
        return detail
    except Exception as exc:
        return {"name":name,"error":type(exc).__name__}
q=urlencode({"q":QUERY})
cases=[
 ("bing_de","https://www.bing.com/search?"+q+"&format=rss&setlang=de&cc=de","application/rss+xml, application/xml"),
 ("bing_en","https://www.bing.com/search?"+q+"&format=rss&setlang=en&cc=us","application/rss+xml, application/xml"),
 ("wikipedia_de","https://de.wikipedia.org/w/api.php?"+urlencode(
  {"action":"query","list":"search","srsearch":QUERY,"format":"json","srlimit":5}),"application/json"),
 ("wikipedia_en","https://en.wikipedia.org/w/api.php?"+urlencode(
  {"action":"query","list":"search","srsearch":QUERY,"format":"json","srlimit":5}),"application/json"),
 ("google_news","https://news.google.com/rss/search?"+q+"&hl=de&gl=DE&ceid=DE:de","application/rss+xml"),
]
if __name__=="__main__":
    for name,url,mime in cases: print(json.dumps(inspect(name,url,mime),ensure_ascii=False),flush=True)


# Inspect only fixed public icon metadata, never user-derived sites here.
from urllib.request import Request
from html.parser import HTMLParser
class IconLinks(HTMLParser):
    def __init__(self):
        super().__init__();self.icons=[]
    def handle_starttag(self,tag,attributes):
        if tag!="link":return
        a=dict(attributes)
        if "icon" in (a.get("rel","").lower().split()):
            self.icons.append({"href":a.get("href","")[:180],"rel":a.get("rel","")[:50]})
try:
    op=fixed_search_opener()
    with op.open(Request("https://www.frankfurt-airport.com/",headers={
        "User-Agent":"Mozilla/5.0 (compatible; NERAL-Icon-Diagnostic/1.0)",
        "Accept":"text/html"},method="GET"),timeout=8) as response:
        mime=response.headers.get("Content-Type","")
        raw=response.read(120_000)
    parser=IconLinks();parser.feed(raw.decode("utf-8",errors="replace"))
    print(json.dumps({"name":"frankfurt_html_icon_links","mime":mime,
         "sample_icons":parser.icons[:8],"bytes":len(raw)},ensure_ascii=False),flush=True)
except Exception as e:
    print(json.dumps({"name":"frankfurt_html_icon_links","error":type(e).__name__}),flush=True)
