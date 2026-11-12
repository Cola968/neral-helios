"""Same-origin, session-guarded opt-in web search for every NERAL plan."""
from __future__ import annotations

from flask import jsonify, request, Response
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import json
from neral_web_search import WebSearch, WebSearchError, WebSearchInputError, WebSearchRateLimitError, verify_public_https_url, fetch_public_favicon


def register_web_search_api(app, service=None):
    """Parent Flask app's existing localhost, session and Origin guards apply."""
    search = service if service is not None else WebSearch()

    def verified_results(result: dict, on_source=None) -> dict:
        """Check public HTTPS endpoints; emit only reachable hits, no page contents."""
        candidates = result.get("results", [])[:10]
        def verify(index, hit):
            if not isinstance(hit, dict):
                return index, None
            checked_url = verify_public_https_url(hit.get("url"))
            return index, ({**hit, "url": checked_url, "reachable": True}
                           if checked_url else None)
        accepted = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(verify, i, hit) for i, hit in enumerate(candidates)]
            for task in as_completed(tasks):
                index, checked = task.result()
                if checked is not None and len(accepted) < 5:
                    accepted[index] = checked
                    if on_source is not None:
                        on_source(checked)
        return {**result, "results": [accepted[i] for i in sorted(accepted)],
                "link_check": "public_https_tls_and_http_only"}

    def verified_with_fallback(result: dict, query: str, on_provider=None, on_source=None) -> dict:
        """Use Bing only when DDG had hits but zero reachable sources.

        This is a continuation of the single user-approved search. It reuses
        the exact query and does not charge a second local search-rate slot.
        """
        verified = verified_results(result, on_source=on_source)
        contacted = result.get("providers_contacted", [])
        retry = getattr(search, "search_bing", None)
        if (verified["results"] or "Bing RSS" in contacted
                or result.get("provider") != "DuckDuckGo HTML"
                or not callable(retry)):
            return verified
        try:
            bing_hits = retry(query, on_provider=on_provider)
        except WebSearchError:
            return verified
        second = {**result, "results": bing_hits, "provider": "Bing RSS",
                  "providers_contacted": [*contacted, "Bing RSS"],
                  "notice": "Nur dieser Suchbegriff wurde an DuckDuckGo und Bing übertragen. Suchtreffer sind ungeprüft."}
        return verified_results(second, on_source=on_source)

    @app.post("/api/web/search/stream")
    def web_search_stream():
        """NDJSON stream of real contacted providers and verified source hits.

        An event is not a claim that page content was read. This endpoint
        inherits the same-origin session/CSRF guards of the hosting Flask app.
        """
        if not request.is_json:
            return jsonify(error="Nur JSON erlaubt."), 415
        data = request.get_json(silent=True)
        if (not isinstance(data, dict) or set(data) != {"query", "verify_links"}
            or data.get("verify_links") is not True
            or not isinstance(data.get("query"), str)
            or not 2 <= len(data["query"].strip()) <= 160
            or any(ord(char) < 32 or ord(char) == 127 for char in data["query"])):
            return jsonify(error="Ungültige Suchanfrage."), 400
        query = data["query"]
        events = Queue()
        def emit(item):
            events.put(item)
        def worker():
            try:
                result = search.search(query,
                    on_provider=lambda name: emit({"type": "provider", "name": name}))
                verified = verified_with_fallback(result, query,
                    on_provider=lambda name: emit({"type": "provider", "name": name}),
                    on_source=lambda source: emit({"type": "source", "source": source}))
                emit({"type": "result", "result": verified})
            except (WebSearchInputError, WebSearchRateLimitError):
                emit({"type": "error", "message": "Websuche konnte nicht ausgeführt werden."})
            except Exception:
                # Never leak provider/network internals into the browser stream.
                emit({"type": "error", "message": "Websuche nicht verfügbar."})
            finally:
                emit({"type": "end"})
        Thread(target=worker, daemon=True).start()
        def generate():
            while True:
                event = events.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] == "end":
                    break
        return Response(generate(), mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-store, no-transform",
                     "X-Content-Type-Options": "nosniff",
                     "X-Accel-Buffering": "no"})

    @app.get("/api/web/favicon")
    def favicon_image():
        """Same-origin favicon for a validated source hostname, never a URL proxy."""
        if set(request.args) != {"host"}:
            return jsonify(error="Nur ein Hostname erlaubt."), 400
        host = request.args.get("host", "")
        if not isinstance(host, str) or len(host) > 253:
            return jsonify(error="Ungültiger Hostname."), 400
        result = fetch_public_favicon(host)
        if result is None:
            return Response(status=404)
        data, mime = result
        return Response(data, mimetype=mime, headers={
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"
        })

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
                result = verified_with_fallback(result, data["query"])
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
