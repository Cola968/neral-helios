"""Same-origin, session-guarded opt-in web search for every NERAL plan."""
from __future__ import annotations

from flask import jsonify, request
from neral_web_search import WebSearch, WebSearchError, WebSearchInputError, WebSearchRateLimitError


def register_web_search_api(app, service=None):
    """Parent Flask app's existing localhost, session and Origin guards apply."""
    search = service if service is not None else WebSearch()

    @app.get("/api/web/search/status")
    def web_search_status():
        return jsonify(configured=True, live_verified=False, free_for_all=True,
                       requires_account=False, requires_api_key=False,
                       provider="DuckDuckGo HTML / Bing RSS", model_provider="none",
                       disclosure="Nur der eingegebene Suchbegriff wird an DuckDuckGo gesendet.")

    @app.post("/api/web/search")
    def web_search_submit():
        if not request.is_json:
            return jsonify(error="Nur JSON erlaubt."), 415
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"query"}:
            return jsonify(error="Nur ein expliziter Suchbegriff ist erlaubt."), 400
        try:
            return jsonify(search.search(data["query"]))
        except WebSearchInputError as exc:
            return jsonify(error=str(exc)), 400
        except WebSearchRateLimitError as exc:
            return jsonify(error=str(exc)), 429
        except WebSearchError:
            return jsonify(error="Websuche nicht verfügbar. Bitte Verbindung und Suchanbieter prüfen."), 503
        except Exception:
            return jsonify(error="Websuche nicht verfügbar."), 503

    return search
