"""Single-shot, source-attributed NERAL Orbit public research, no LLM or shell.

A human must approve AND manually launch a research task via Orbit. The ONLY
remote endpoint is Wikimedia's German Wikipedia MediaWiki API. One request,
maximum three encyclopedic excerpts, hard bytes/timeout, no redirects/proxies,
no arbitrary URLs, no user code execution, no periodic/background activity.

Source excerpts are UNTRUSTED DATA, not commands or factual verification.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

_HOST = "de.wikipedia.org"
_API = "https://de.wikipedia.org/w/api.php"
_USER_AGENT = "NERALOrbitResearchBot/0.1 (https://github.com/Cola968) Python-urllib"
_MAX_RESPONSE_BYTES = 96 * 1024
_MAX_QUERY_CHARS = 200
_MAX_SOURCES = 3


class ResearchError(ValueError):
    """Research did not complete; never present a fabricated result."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_wikipedia(url: str) -> dict:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != _HOST
            or parsed.path != "/w/api.php" or parsed.fragment):
        raise ResearchError("Die Recherchequelle ist nicht erlaubt.")
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=6) as response:
            if response.status != 200:
                raise ResearchError("Wikipedia antwortete nicht erfolgreich.")
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" not in content_type:
                raise ResearchError("Wikipedia meldet kein JSON.")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ResearchError("Wikipedia-Antwort überschreitet das Limit.")
        return json.loads(raw.decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, ResearchError):
            raise
        raise ResearchError("Wikipedia ist nicht erreichbar oder lieferte ungültige Daten.") from exc


def run_public_research(goal: str, *, transport=None) -> dict:
    """Look up a *bounded* encyclopedic query and persistable source excerpts.

    transport is test-injectable, so CI never calls real external services.
    """
    if (not isinstance(goal, str) or not 3 <= len(goal.strip()) <= _MAX_QUERY_CHARS
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in goal)):
        raise ResearchError("Recherchefrage muss 3 bis 200 Zeichen ohne Steuerzeichen haben.")
    query = goal.strip()
    options = {
        "action": "query", "generator": "search",
        "gsrsearch": query, "gsrnamespace": "0", "gsrlimit": str(_MAX_SOURCES),
        "prop": "extracts", "exintro": "1", "explaintext": "1",
        "exsentences": "2", "format": "json", "formatversion": "2",
    }
    url = _API + "?" + urllib.parse.urlencode(options)
    try:
        payload = (transport or _fetch_wikipedia)(url)
    except ResearchError:
        raise
    except Exception as exc:
        raise ResearchError("Öffentliche Recherchequelle ist nicht erreichbar.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), dict):
        raise ResearchError("Wikipedia lieferte kein gültiges Suchergebnis.")
    pages = payload["query"].get("pages")
    if not isinstance(pages, list) or not 1 <= len(pages) <= _MAX_SOURCES:
        raise ResearchError("Keine auswertbaren öffentlichen Treffer gefunden.")
    results = []
    for row in pages:
        if not isinstance(row, dict):
            raise ResearchError("Ungültiger Quellen-Eintrag.")
        page_id, title, excerpt = row.get("pageid"), row.get("title"), row.get("extract")
        if (type(page_id) is not int or not 1 <= page_id < 2**53
                or not isinstance(title, str) or not 1 <= len(title) <= 180
                or not isinstance(excerpt, str) or not excerpt.strip()
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in title)):
            raise ResearchError("Ungültiger Quellen-Eintrag.")
        # No user-supplied destination. Page ID links are built on a pinned host.
        results.append({
            "title": title,
            "url": f"https://de.wikipedia.org/?curid={page_id}",
            "excerpt": re.sub(r"\s+", " ", excerpt).strip()[:700],
        })
    return {
        "provider": "Wikipedia (de)",
        "query": query,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": results, "source_count": len(results),
        "model_used": False, "requires_review": True,
        "limitations": "Nur enzyklopädische Einleitungen; keine Live-Nachrichten, "
                       "keine verifizierte Primärrecherche und keine KI-Synthese.",
    }
