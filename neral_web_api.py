"""Same-origin, session-guarded opt-in web search for every NERAL plan."""
from __future__ import annotations

from flask import jsonify, request
from concurrent.futures import ThreadPoolExecutor
from neral_web_search import WebSearch, WebSearchError, WebSearchInputError, WebSearchRateLimitError, verify_public_https_url


def register_web_search_api(app, service=None):
    """Parent Flask app's existing localhost, session and Origin guards apply."""
    search = service if service is not None else WebSearch()

    @app.get("/api/web/search/status")
    def web_search_status():
        return jsonify(configured=True, live_verified=False, free_for_all=True,
                       requires_account=False, requires_api_key=False,
                       provider="DuckDuckGo HTML / Bing RSS", model_provider="none",
                       disclosure="Nur der Suchbegriff wird an DuckDuckGo, bei Bedarf auch an Bing gesendet.")

    @app.post("/api/web/search")
    def web_search_submit():
        if not request.is_json:
            return jsonify(error="Nur JSON erlaubt."), 415
        data = request.get_json(silent=True)
        if (not isinstance(data, dict) or "query" not in data or
                not set(data).issubset({"query", "verify_links"}) or
                ("verify_links" in data and data["verify_links"] is not True)):
            return jsonify(error="Nur ein expliziter Suchbegriff ist erlaubt."), 400
        try:
            result = search.search(data["query"])
            if data.get("verify_links") is True:
                candidates = result.get("results", [])[:5]
                # Bound both worker count and per-link socket timeout.
                def verify(hit):
                    if not isinstance(hit, dict):
                        return None
                    checked_url = verify_public_https_url(hit.get("url"))
                    return {**hit, "url": checked_url, "reachable": True} if checked_url else None
                with ThreadPoolExecutor(max_workers=3) as pool:
                    checked = list(pool.map(verify, candidates))
                result = {**result, "results": [hit for hit in checked if hit],
                          "link_check": "public_https_tls_and_http_only"}
            return jsonify(result)
        except WebSearchInputError as exc:
            return jsonify(error=str(exc)), 400
        except WebSearchRateLimitError as exc:
            return jsonify(error=str(exc)), 429
        except WebSearchError:
            return jsonify(error="Websuche nicht verfügbar. Bitte Verbindung und Suchanbieter prüfen."), 503
        except Exception:
            return jsonify(error="Websuche nicht verfügbar."), 503

    return search
