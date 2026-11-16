"""Live, read-only NERAL Web Search acceptance on a Windows self-hosted runner.

Only two fixed public queries leave the host. This does not inspect browser
history, local documents, Ollama, or account information; nothing is saved.
Records bounded scalar counts and provider labels, never page bodies or URLs.
"""
from __future__ import annotations

import json
import platform
import sys
from flask import Flask

from neral_web_api import register_web_search_api
from neral_web_search import WebSearch, WebSearchError

QUERIES = ("Flughafen Frankfurt", "Wetter Berlin")
PROVIDERS = {"DuckDuckGo HTML", "Bing RSS"}


def one_probe(query: str) -> dict:
    service = WebSearch(max_per_minute=6)
    contacted = []
    try:
        raw = service.search(query, on_provider=contacted.append)
        first_provider = str(raw.get("provider", "unknown"))[:35]
        raw_count = len(raw.get("results") or [])
        search_error = None
    except WebSearchError:
        first_provider, raw_count, search_error = "failed", 0, "WebSearchError"

    # Independent real Bing call. A 200 and hits are not topicality proof.
    try:
        bing_count = len(service.search_bing(query))
        bing_error = None
    except WebSearchError:
        bing_count, bing_error = 0, "WebSearchError"

    # The real NERAL API, actual provider network + topical filter + HTTPS
    # link probes. No mocks, no Ollama, no end-user prompt.
    app = Flask(__name__)
    register_web_search_api(app, service=WebSearch(max_per_minute=6))
    response = app.test_client().post(
        "/api/web/search", json={"query": query, "verify_links": True}
    )
    payload = response.get_json(silent=True) or {}
    verified = payload.get("results") or []
    previews = payload.get("previews") or []
    source_providers = sorted({
        str(hit.get("provider", ""))[:35]
        for hit in [*verified, *previews]
        if isinstance(hit, dict)
    })
    non_reference = any(p in PROVIDERS for p in source_providers)
    return {
        "query": query,
        "first_provider": first_provider,
        "first_raw_count": raw_count,
        "first_error": search_error,
        "contacted": list(dict.fromkeys(contacted)),
        "bing_raw_count": bing_count,
        "bing_error": bing_error,
        "api_http": response.status_code,
        "api_provider": str(payload.get("provider", ""))[:35],
        "api_checked": len(verified),
        "api_previews": len(previews),
        "source_providers": source_providers,
        "non_reference_topical_source": non_reference,
    }


def main() -> int:
    results = [one_probe(query) for query in QUERIES]
    healthy = any(
        row["api_http"] == 200
        and row["non_reference_topical_source"]
        and (row["api_checked"] > 0 or row["api_previews"] > 0)
        for row in results
    )
    print(json.dumps({
        "name": "NERAL_WEB_PC_REAL_NO_MOCKS",
        "host_os": platform.system(),
        "live_network_test": True,
        "model_called": False,
        "user_data_sent": False,
        "status": "PASS_WITH_REAL_NONREFERENCE_RESULTS" if healthy
                  else "FAIL_NO_TOPICAL_DDG_OR_BING",
        "results": results,
    }, ensure_ascii=False), flush=True)
    return 0 if healthy else 2


if __name__ == "__main__":
    sys.exit(main())
