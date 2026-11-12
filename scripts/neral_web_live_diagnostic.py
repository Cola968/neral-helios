"""Opt-in live matrix of actual public search providers and favicon retrieval.

Uses only fixed, nonpersonal queries. No Ollama calls, user conversations,
models, tokens, passwords, or local files are sent to third parties.
This is a non-gating network diagnostic: providers may be unavailable.
"""
from __future__ import annotations
import json
from urllib.parse import urlsplit
from flask import Flask
from neral_web_api import register_web_search_api
from neral_web_search import WebSearch, fetch_public_favicon

QUERIES = ("Flughafen Frankfurt", "NASA Artemis",
           "Python Dokumentation", "Wetter Berlin")
class TracedSearch(WebSearch):
    def __init__(self):
        super().__init__()
        self.provider_raw = {}
    def search(self, query, on_provider=None):
        result = super().search(query, on_provider=on_provider)
        self.provider_raw.setdefault(query, []).append(
            {"provider":result.get("provider"),"count":len(result.get("results",[])),
             "first_titles":[x.get("title","")[:70] for x in result.get("results",[])[:3]]})
        return result
    def search_bing(self, query, on_provider=None):
        hits = super().search_bing(query, on_provider=on_provider)
        self.provider_raw.setdefault(query, []).append(
            {"provider":"Bing RSS retry","count":len(hits),
             "first_titles":[x.get("title","")[:70] for x in hits[:3]]})
        return hits

def run():
    service=TracedSearch()
    app=Flask(__name__)
    register_web_search_api(app,service)
    rows=[]
    for query in QUERIES:
        response=app.test_client().post("/api/web/search/stream",
            json={"query":query,"verify_links":True})
        events=[json.loads(line) for line in response.response]
        result=next((event["result"] for event in events if event.get("type")=="result"),{})
        previews=result.get("previews",[])
        checked=result.get("results",[])
        rows.append({
            "query":query,"http_status":response.status_code,
            "providers_contacted":[e["name"] for e in events if e.get("type")=="provider"],
            "raw":service.provider_raw.get(query,[]),
            "error":[e.get("message") for e in events if e.get("type")=="error"],
            "candidate_events":sum(e.get("type")=="candidate" for e in events),
            "verified_events":sum(e.get("type")=="source" for e in events),
            "previews":len(previews),"verified":len(checked),
            "sample_titles":[s.get("title","")[:100] for s in (checked or previews)[:3]],
            "sample_urls":[s.get("url","")[:170] for s in (checked or previews)[:3]],
        })
    hosts=("de.wikipedia.org","www.frankfurt-airport.com","www.nasa.gov","docs.python.org")
    icons={host:bool(fetch_public_favicon(host)) for host in hosts}
    print(json.dumps({"multi_query_search":rows,"favicon_probe":icons},
                     ensure_ascii=False,indent=2),flush=True)
    return 0 if sum(bool(x["previews"] or x["verified"]) for x in rows)>=3 else 1

if __name__=="__main__":
    raise SystemExit(run())
