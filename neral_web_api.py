"""Same-origin, session-guarded opt-in web search for every NERAL plan."""
from __future__ import annotations

from flask import jsonify, request, Response
from collections import OrderedDict
from threading import Lock
from time import monotonic
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
import json
import re
import unicodedata
from urllib.parse import urlsplit
from neral_web_search import WebSearch, WebSearchError, WebSearchInputError, WebSearchRateLimitError, verify_public_https_url, fetch_public_favicon, safe_result_url, is_fresh_news_query


def register_web_search_api(app, service=None, page_reader=None):
    """Parent Flask app's existing localhost, session and Origin guards apply."""
    search = service if service is not None else WebSearch()
    from neral_v3_web_reader import fetch_and_extract, read_top_pages
    fetch_page = page_reader

    def enrich_with_read_pages(result):
        """Fetch real article HTML, never claim snippets were opened."""
        candidates = result.get("previews") or result.get("results") or []
        pages = read_top_pages(candidates, fetcher=fetch_page, max_pages=3,
                               query=result.get('query', ''))
        return {**result, "pages": pages,
                "page_read": "main_text_https" if pages else "not_available"}

    # Cache only public, already-sanitized icon bytes. The UI displays the same
    # site icon in its live status, source stack and source cards.
    favicon_cache = OrderedDict()
    favicon_cache_lock = Lock()
    favicon_cache_limit = 96
    favicon_positive_ttl = 600
    favicon_negative_ttl = 90



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
            "game":"spiel", "games":"spiel", "videogame":"spiel",
            "videospiel":"spiel", "computerspiel":"spiel",
            "save":"speichern", "saved":"speichern",
            "saves":"speichern", "saving":"speichern",
            "gespeichert":"speichern",
        }
        noise = {"suche","such","nach","bitte","finde","recherchiere","im",
                 "dem","der","den","die","das","und","mit","ist","eine","einer",
                 "einem","was","wie","wer","ich","man","kann","konnen",
                 "ob","mir","von","bei","fur","auf","zum",
                 "zur","online","internet","web","aktuell","aktuelle",
                 "aktuellen","aktueller","aktuelles","heute","neueste",
                 "neuesten","neuester","neuestes","neu","neue","neuen",
                 "neuer","neues","quelle","quellen","links","news",
                 "nachricht","nachrichten","meldung","meldungen","berichte",
                 "bericht","letzte","letzten","letzter","letztes",
                 "information","informationen","uber","ueber","zeige","mehr","deine",
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
        # A generic category alone must not pass for a named destination.
        # Example: Bremen tram stop "Flughafen" is not Frankfurt Airport.
        if len(terms) >= 2 and len(matched) < 2:
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
        fresh = is_fresh_news_query(query or result.get("query", ""))
        # For fresh searches, link verification adds substantial HTTPS latency.
        # Provider-discovered HTTPS URLs remain usable, but are labelled as
        # metadata-only instead of claiming their destinations were contacted.
        if not fresh:
            with ThreadPoolExecutor(max_workers=3) as pool:
                tasks = [pool.submit(verify, i, hit) for i, hit in enumerate(previews[:3])]
                for task in as_completed(tasks):
                    index, checked = task.result()
                    if checked is not None:
                        accepted[index] = checked
        verified = [accepted[i] for i in sorted(accepted)][:5]
        if on_source is not None:
            for hit in verified:
                on_source(hit)
        return {**result, "results": verified, "previews": previews,
                "link_check": ("provider_metadata_only" if fresh else
                               "public_https_tls_and_http_only")}

    def verified_with_fallback(result: dict, query: str, on_provider=None,
                               on_source=None, on_candidate=None) -> dict:
        """Preserve all safe, relevant discovery hits across every provider.

        HTTPS reachability failure is NOT an empty search; a result preview
        remains visible with reachable=False. News previews are preferable to
        stale encyclopedia entries for explicitly fresh search intent.
        """
        first=verified_results(result,on_source=on_source,
                               on_candidate=on_candidate,query=query)
        if first["results"] or first.get("previews"):
            # Real, topic-matched search metadata is enough for a fast response.
            # Do not wait for Bing/Google/Wikipedia just because a remote site
            # refuses a separate HTTPS reachability probe.
            return first
        contacted=list(result.get("providers_contacted",[]))
        current=first
        retained=list(first.get("previews",[]))
        retry=getattr(search,"search_bing",None)
        if ("Bing RSS" not in contacted and result.get("provider")=="DuckDuckGo HTML"
                and callable(retry)):
            contacted.append("Bing RSS")
            try:
                hits=retry(query,on_provider=on_provider)
                secondary={**result,"results":hits,"provider":"Bing RSS",
                           "providers_contacted":list(contacted)}
                current=verified_results(secondary,on_source=on_source,
                                         on_candidate=on_candidate,query=query)
                retained.extend(current.get("previews",[]))
            except WebSearchError:
                current={**first,"providers_contacted":list(contacted)}
        if current["results"]:
            return current

        news=getattr(search,"search_news",None)
        if (is_fresh_news_query(query) and "Google News RSS" not in contacted
                and callable(news)):
            contacted.append("Google News RSS")
            try:
                hits=news(query,on_provider=on_provider)
                news_result={**result,"results":hits,"provider":"Google News RSS",
                             "providers_contacted":list(contacted)}
                current=verified_results(news_result,on_source=on_source,
                                         on_candidate=on_candidate,query=query)
                retained.extend(current.get("previews",[]))
            except WebSearchError:
                current={**current,"providers_contacted":list(contacted)}
        # A genuine, topic-matched News RSS hit stays visible even if the
        # publisher disallows HEAD/GET checks. Do not erase it by replacing
        # the result with a stale Wikipedia reference.
        if current["results"] or (is_fresh_news_query(query)
                and current.get("provider")=="Google News RSS"
                and current.get("previews")):
            return current

        wiki=getattr(search,"search_wikipedia",None)
        if ("Wikipedia DE" not in contacted and
                result.get("provider")!="Wikipedia DE" and callable(wiki)):
            contacted.append("Wikipedia DE")
            try:
                hits=wiki(query,on_provider=on_provider)
                reference={**result,"results":hits,"provider":"Wikipedia DE",
                           "providers_contacted":list(contacted)}
                current=verified_results(reference,on_source=on_source,
                                         on_candidate=on_candidate,query=query)
                retained.extend(current.get("previews",[]))
            except WebSearchError:
                current={**current,"providers_contacted":list(contacted)}

        seen=set()
        discovered=[]
        for hit in [*current.get("previews",[]),*retained]:
            url=hit.get("url")
            if url and url not in seen:
                discovered.append(hit)
                seen.add(url)
            if len(discovered)>=5:
                break
        result_provider=(current.get("provider") if current.get("results") else
                         discovered[0].get("provider",current.get("provider"))
                         if discovered else current.get("provider"))
        return {**current,"previews":discovered,
                "provider":result_provider,
                "providers_contacted":list(contacted)}

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
        # Existing NDJSON clients remain compatible; new browser clients request
        # true SSE with an immediate status frame (before provider I/O).
        wants_sse = "text/event-stream" in request.headers.get("Accept", "").lower()
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
                emit({"type": "status", "state": "reading_pages"})
                enriched = enrich_with_read_pages(verified)
                for page in enriched["pages"]:
                    emit({"type": "page", "page": page})
                emit({"type": "result", "result": enriched})
            except (WebSearchInputError, WebSearchRateLimitError):
                emit({"type": "error", "message": "Websuche konnte nicht ausgeführt werden."})
            except Exception:
                # Never leak provider/network internals into the browser stream.
                emit({"type": "error", "message": "Websuche nicht verfügbar."})
            finally:
                emit({"type": "end"})
        Thread(target=worker, daemon=True).start()
        def generate():
            def encode(event):
                payload = json.dumps(event, ensure_ascii=False)
                return ("data: " + payload + "\n\n") if wants_sse else (payload + "\n")
            yield encode({"type": "status", "state": "searching"})
            while True:
                event = events.get()
                yield encode(event)
                if event["type"] == "end":
                    break
        return Response(generate(), mimetype=(
            "text/event-stream" if wants_sse else "application/x-ndjson"),
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
        now = monotonic()
        with favicon_cache_lock:
            entry = favicon_cache.get(host)
            if entry and entry[0] > now:
                favicon_cache.move_to_end(host)
                result = entry[1]
                cached = True
            else:
                favicon_cache.pop(host, None)
                cached = False
                result = None
        if not cached:
            result = fetch_public_favicon(host)
            with favicon_cache_lock:
                favicon_cache[host] = (
                    monotonic() + (favicon_positive_ttl if result is not None
                                   else favicon_negative_ttl), result)
                favicon_cache.move_to_end(host)
                while len(favicon_cache) > favicon_cache_limit:
                    favicon_cache.popitem(last=False)
        if result is None:
            return Response(status=404)
        data, mime = result
        return Response(data, mimetype=mime, headers={
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
            "Cache-Control": "private, max-age=600"
        })

    @app.get("/api/web/search/status")
    def web_search_status():
        return jsonify(configured=True, live_verified=False, page_reader=True, free_for_all=True,
                       requires_account=False, requires_api_key=False,
                       provider="DuckDuckGo HTML / Bing RSS / Google News RSS (aktuell) / Wikipedia DE", model_provider="none",
                       disclosure="Nur der Suchbegriff wird an DuckDuckGo, bei Bedarf an Bing, Google News (aktuelle Nachrichten) oder Wikipedia gesendet.")

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
                result = enrich_with_read_pages(result)
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
