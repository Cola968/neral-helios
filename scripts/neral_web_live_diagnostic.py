"""Non-gating live diagnostic for the exact user-reported Grok search case.

Runs the same /api/web/search/stream route with fixed public test query.
No user history, credentials, local model, external AI API, or secrets.
"""
from __future__ import annotations

import json
from flask import Flask
from neral_web_api import register_web_search_api
from neral_web_search import fetch_public_favicon, FAVICON_PATHS, _request_favicon, sanitize_favicon_svg
from ipaddress import ip_address
import socket

QUERY = "grok ai"

def run():
    app = Flask(__name__)
    register_web_search_api(app)
    response = app.test_client().post(
        "/api/web/search/stream",
        json={"query": QUERY, "verify_links": True},
    )
    events = [json.loads(line) for line in response.response]
    result = next((event["result"] for event in events if event.get("type") == "result"), {})
    hits = result.get("results", [])
    favicon_hosts = ("grok.com", "x.ai", "github.com", "www.nasa.gov")
    icons = {host: bool(fetch_public_favicon(host)) for host in favicon_hosts}
    report = {
        "query": QUERY,
        "http_status": response.status_code,
        "providers_contacted": [e["name"] for e in events if e.get("type") == "provider"],
        "error": [e["message"] for e in events if e.get("type") == "error"],
        "verified_relevant_hits": len(hits),
        "titles": [h.get("title", "")[:100] for h in hits],
        "urls": [h.get("url", "")[:200] for h in hits],
        "icons_loaded": icons,
    }
    if not icons.get("grok.com"):
        # Fixed public host only; never accept a user-provided URL in diagnostics.
        details = []
        try:
            ips={v[4][0] for v in socket.getaddrinfo(
                "grok.com",443,type=socket.SOCK_STREAM,proto=socket.IPPROTO_TCP)}
            if ips and all(ip_address(ip).is_global for ip in ips):
                ip=sorted(ips,key=lambda x:(":" in x,x))[0]
                for path in FAVICON_PATHS:
                    try:
                        content, mime=_request_favicon("grok.com",ip,path)
                        details.append({"path":path,"length":len(content),
                                        "mime":mime,
                                        "svg_accepted":bool(sanitize_favicon_svg(content)) if path.endswith(".svg") else None})
                    except Exception as e:
                        details.append({"path":path,"error":type(e).__name__})
            else:
                details.append({"error":"nonpublic_dns"})
        except Exception as e:
            details.append({"error":type(e).__name__})
        report["grok_icon_diagnostic"]=details
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if hits and any(icons.values()) else 1

if __name__ == "__main__":
    raise SystemExit(run())
