"""Opt-in provider adapter. No live requests in CI; untrusted snippets never become instructions."""
from __future__ import annotations
from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
import json
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,urlopen


class _Text(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[];self.block=0
    def handle_starttag(self,tag,attrs):
        if tag in ("script","style"): self.block+=1
    def handle_endtag(self,tag):
        if tag in ("script","style") and self.block:self.block-=1
    def handle_data(self,data):
        if not self.block:self.parts.append(data)


def clean_html(value:str)->str:
    parser=_Text()
    parser.feed(value[:30000])
    return " ".join(" ".join(parser.parts).split())[:1500]


def safe_public_url(value:str)->bool:
    try:
        url=urlsplit(value)
        if url.scheme!="https" or not url.hostname or url.username or url.password or url.port not in (None,443): return False
        host=url.hostname.rstrip(".")
        if not host or host=="localhost" or "." not in host: return False
        try: return ipaddress.ip_address(host).is_global
        except ValueError: return True  # links only; URLs are never fetched by this adapter
    except (ValueError,TypeError): return False


@dataclass(frozen=True)
class Result:
    title:str
    url:str
    snippet:str
    provider:str
    citation:str


class SearchClient:
    def __init__(self, provider:str, endpoint:str, *, api_key:str|None=None, timeout:float=5, transport=None):
        if provider not in ("searxng","tavily","duckduckgo"): raise ValueError("provider")
        u=urlsplit(endpoint)
        if u.scheme!="https" and not (u.scheme=="http" and u.hostname in ("localhost","127.0.0.1","::1")):
            raise ValueError("endpoint must be HTTPS or loopback")
        if not 0.1<=timeout<=15: raise ValueError("timeout")
        self.provider,self.endpoint,self.api_key,self.timeout,self.transport=provider,endpoint,api_key,timeout,transport

    def _request(self,url:str,body:bytes|None=None)->dict:
        if self.transport:
            obj=self.transport(url,body,self.timeout)
        else:
            request=Request(url,data=body,headers={"Content-Type":"application/json","User-Agent":"NERAL-CPU-Foundation/1.2"})
            with urlopen(request,timeout=self.timeout) as response:
                raw=response.read(1_000_001)
            if len(raw)>1_000_000: raise ValueError("provider response too large")
            obj=json.loads(raw)
        if not isinstance(obj,dict): raise ValueError("invalid provider response")
        return obj

    def search(self,query:str,limit:int=5)->list[Result]:
        if not isinstance(query,str) or not 1<=len(query)<=500 or not 1<=limit<=10: raise ValueError("query")
        if self.provider=="searxng":
            sep="&" if "?" in self.endpoint else "?"
            raw=self._request(self.endpoint+sep+urlencode({"q":query,"format":"json"}))
            data=raw.get("results",[])
        elif self.provider=="tavily":
            if not self.api_key: raise ValueError("Tavily key missing")
            body=json.dumps({"api_key":self.api_key,"query":query,"max_results":limit}).encode()
            raw=self._request(self.endpoint,body)
            data=raw.get("results",[])
        else:
            sep="&" if "?" in self.endpoint else "?"
            raw=self._request(self.endpoint+sep+urlencode({"q":query,"format":"json","no_html":1}))
            data=raw.get("RelatedTopics",[])
            # DuckDuckGo Instant Answer isn't a full web result index.
        out=[];seen=set()
        for item in data:
            if not isinstance(item,dict): continue
            url=item.get("url") or item.get("FirstURL")
            if not isinstance(url,str) or url in seen or not safe_public_url(url): continue
            seen.add(url)
            title=clean_html(str(item.get("title") or item.get("Text") or "Quelle"))
            snippet=clean_html(str(item.get("content") or item.get("snippet") or item.get("Text") or ""))
            out.append(Result(title[:200],url,snippet,self.provider,f"[S{len(out)+1}]"))
            if len(out)>=limit:break
        return out
