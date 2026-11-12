"""Opt-in, fixed-query LIVE diagnostic: no model call, secrets or user data.

Intentionally does not assume DuckDuckGo/Bing are available: provider health
must be tested against the actual public network, not just mocked unit tests.
"""
from __future__ import annotations
import json
import sys
from neral_web_search import WebSearch, WebSearchError, fetch_public_favicon, verify_public_https_url

QUERY = "Flughafen Berlin Brandenburg offizielle Website"
def run():
    report = {"query": QUERY, "provider_ok": False, "providers_contacted": [],
              "parsed_hits": 0, "verified_hits": 0, "favicon": {}}
    service = WebSearch()
    try:
        first = service.search(QUERY)
        report["provider_ok"] = True
        report["providers_contacted"] = first.get("providers_contacted", [])
        report["parsed_hits"] = len(first.get("results", []))
        for hit in first.get("results", [])[:4]:
            if verify_public_https_url(hit.get("url")):
                report["verified_hits"] += 1
        if not report["verified_hits"] and "Bing RSS" not in report["providers_contacted"]:
            try:
                hits = service.search_bing(QUERY)
                report["providers_contacted"].append("Bing RSS")
                report["bing_fallback_hits"] = len(hits)
                for hit in hits[:4]:
                    if verify_public_https_url(hit.get("url")):
                        report["verified_hits"] += 1
            except WebSearchError as exc:
                report["bing_fallback_error"] = type(exc).__name__
    except WebSearchError as exc:
        report["provider_error"] = type(exc).__name__
    for host in ("github.com", "www.nasa.gov"):
        data = fetch_public_favicon(host)
        report["favicon"][host] = {"loaded": bool(data),
                                    "mime": data[1] if data else None}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["verified_hits"] and any(x["loaded"] for x in report["favicon"].values()) else 1

if __name__ == "__main__":
    sys.exit(run())
