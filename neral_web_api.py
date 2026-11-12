"""Same-origin, session-guarded opt-in web search for every NERAL plan."""
from __future__ import annotations

from flask import jsonify, request, Response
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import json
import re
import unicodedata
from urllib.parse import urlsplit
from neral_web_search import WebSearch, WebSearchError, WebSearchInputError, WebSearchRateLimitError, verify_public_https_url, fetch_public_favicon, safe_result_url


def register_web_search_api(app, service=None):
    """Parent Flask app's existing localhost, session and Origin guards apply."""
    search = service if service is not None else WebSearch()


    def topic_terms(text: str) -> set[str]:
        """Conservative topical keywords, including cross-language common labels.

        A public search result need not repeat a whole German sentence. Treat
        provider snippets as untrusted metadata, never as read article content.
        """
        plain = unicodedata.normalize("NFKD", str(text).lower())
        plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
        aliases = {
            "airport":"flughafen", "airports":"flughafen",
            "weather":"wetter", "forecast":"wetter", "forecasts":"wetter",
            "documentation":"dokumentation", "docs":"dokumentation",
            "flights":"fluge", "flug":"fluge", "fluege":"fluge",
        }
        noise = {"suche","such","nach","bitte","finde","recherchiere","im",
                 "dem","der","den","die","das","und","mit","ist","eine","einer",
                 "einem","was","wie","wer","von","bei","fur","auf","zum",
                 "zur","online","internet","web","aktuell","heute","neueste",
                 "neuesten","quelle","quellen","links","news","information",
                 "informationen","uber","ueber","zeige","mehr","deine",
                 "site","www","https","http","2026","2025","2027","2028",
                 "official","offiziell","offizielle","offiziellen"}
        words = re.findall(r"[a-z0-9]{3,}", plain)
        result = set()
        for word in words:
            if word in noise:
                continue
            if word.endswith("er") and len(word) >= 7 and word not in aliases:
                word = word[:-2]
            result.add(aliases.get(word, word))
        return result

    def topical_score(query: str, hit: dict) -> tuple[int, int] | None:
        """A *search* relevance score, not verification that a page was read.

        Reject unrelated results; don't require every word of a multilingual
        query, which previously hid real results like 'Frankfurt Airport'.
        """
        if not isinstance(hit, dict) or safe_result_url(hit.get("url")) is None:
            return None
        terms = topic_terms(query)
        if not terms:
            return (0, 0)
        host = urlsplit(hit["url"]).hostname or ""
        heading = topic_terms(str(hit.get("title", "")) + " " + host.replace("-", " "))
        snippet = topic_terms(str(hit.get("snippet", "")))
        matched = terms.intersection(heading | snippet)
        if not matched:
            return None
        categories = {"flughafen","wetter"}
        category = terms.intersection(categories)
        # With multiple query terms an unrelated result at the right location
        # is not enough (e.g. TV Frankfurt is not Frankfurt Airport).
        if category and len(terms)>1 and not matched.intersection(category):
            return None
        if not matched.intersection(heading) and len(matched)<2:
            return None
        return (len(matched), len(matched.intersection(heading)))

    def verified_results(result: dict, on_source=None, on_candidate=None, query: str = "") -> dict:
        """Publish provider-discovered safe links independently of URL probes.

        'previews' are merely relevant search-engine snippets, not verified
        reachability; 'results' retain the old fully checked contract.
        """
        selected = []
        for index, hit in enumerate(result.get("results", [])[:10]):
            score = topical_score(query or result.get("query", ""), hit)
            if score is not None:
                selected.append((score, index, hit))
        selected.sort(key=lambda item: (-item[0][0], -item[0][1], item[1]))
        previews = []
        for score, _, hit in selected[:5]:
            url = safe_result_url(hit.get("url"))
            if url is None:
                continue
            item = {**hit, "url":url, "reachable":False,
                    "discovered":True,"topically_relevant":True,
                    "provider":result.get("provider","Suchanbieter")}
            previews.append(item)
            if on_candidate is not None:
                on_candidate(item)
        # Probe at most five candidate links and cap concurrency. A failing HEAD
        # must never erase the real search-engine result or its site icon.
        def verify(index, item):
            checked_url = verify_public_https_url(item["url"])
            return index, ({**item, "url":checked_url,"reachable":True}
                           if checked_url else None)
        accepted = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(verify, i, hit) for i, hit in enumerate(previews)]
            for task in as_completed(tasks):
                index, checked = task.result()
                if checked is not None:
                    accepted[index] = checked
        verified = [accepted[i] for i in sorted(accepted)][:5]
        if on_source is not None:
            for hit in verified:
                on_source(hit)
        return {**result, "results": verified, "previews": previews,
                "link_check": "public_https_tls_and_http_only"}

    def verified_with_fallback(result: dict, query: str, on_provider=None,
                               on_source=None, on_candidate=None) -> dict:
        """Use Bing when there is no VERIFIED DDG result; retain safe previews."""
        first = verified_results(result, on_source=on_source,
                                 on_candidate=on_candidate, query=query)
        contacted = list(result.get("providers_contacted", []))
        retry = getattr(search, "search_bing", None)
        if (first["results"] or "Bing RSS" in contacted
                or result.get("provider") != "DuckDuckGo HTML"
                or not callable(retry)):
            return first
        try:
            bing_hits = retry(query, on_provider=on_provider)
        except WebSearchError:
            return {**first,"providers_contacted":[*contacted, "Bing RSS"],
                    "notice":"Bing wurde kontaktiert, konnte aber keine weiteren Treffer bereitstellen."}
        second = {**result, "results": bing_hits, "provider": "Bing RSS",
                  "providers_contacted": [*contacted, "Bing RSS"],
                  "notice": "Nur der Suchbegriff wurde an DuckDuckGo und Bing übertragen. Suchtreffer sind ungeprüft."}
        checked = verified_results(second, on_source=on_source,
                                   on_candidate=on_candidate, query=query)
        if checked["results"]:
            return checked
        combined=[]
        seen=set()
        for item in [*checked["previews"],*first["previews"]]:
            if item["url"] not in seen:
                combined.append(item)
                seen.add(item["url"])
        return {**checked,"previews":combined[:5]}

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
                    on_source=lambda source: emit({"type": "source", "source": source}),
                    on_candidate=lambda source: emit({"type": "candidate", "source": source}))
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
