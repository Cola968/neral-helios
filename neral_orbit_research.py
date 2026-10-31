"""Single-shot, source-attributed NERAL Orbit public research, no LLM or shell.

A human must approve AND manually launch a research task via Orbit. The ONLY
remote endpoint is Wikimedia's German Wikipedia MediaWiki API. One request,
maximum three encyclopedic excerpts, hard bytes/timeout, no redirects/proxies,
no arbitrary URLs, no user code execution, no periodic/background activity.

Source excerpts are UNTRUSTED DATA, not commands or factual verification.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
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
_PRIVATE_QUERY_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|github_pat_[A-Za-z0-9_-]{16,}|"
    r"gh[pousr]_[A-Za-z0-9_-]{16,}|xox[baprs]-[A-Za-z0-9_-]{16,}|"
    r"-----BEGIN[^\r\n]{0,60}PRIVATE KEY|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})", re.IGNORECASE)


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


def validate_public_query(goal: str) -> str:
    """Fail closed before disclosure of private/credential-like search terms."""
    if (not isinstance(goal, str) or not 3 <= len(goal.strip()) <= _MAX_QUERY_CHARS
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in goal)):
        raise ResearchError("Recherchefrage muss 3 bis 200 Zeichen ohne Steuerzeichen haben.")
    query = goal.strip()
    if _PRIVATE_QUERY_RE.search(query):
        raise ResearchError("Recherchefrage enthält mögliche Zugangsdaten oder private Kontaktinformationen.")
    return query


def run_public_research(goal: str, *, transport=None) -> dict:
    """Look up bounded encyclopedic text; never send secrets or credentials."""
    query = validate_public_query(goal)
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


def main(argv=None):
    """Mobile/Termux standalone mode; defaults to read-only, NEVER background."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Public Wikipedia research query; no secrets")
    parser.add_argument("--out", type=Path, help="NEW JSON report file; will not overwrite")
    parser.add_argument("--confirm-public", action="store_true",
                        help="Explicitly allow sending the search query to Wikipedia once")
    parser.add_argument("--self-test", action="store_true",
                        help="Offline synthetic worker test: no network, file or GPU")
    a=parser.parse_args(argv)
    if a.self_test:
        if a.query is not None or a.out is not None or a.confirm_public:
            parser.error("--self-test cannot be combined with public queries or output paths")
        example={"query":{"pages":[
            {"pageid":42,"title":"NERAL offline probe",
             "extract":"Synthetischer Text für die Offline-Funktionsprüfung."}
        ]}}
        report=run_public_research(
            "NERAL offline probe",transport=lambda url: example)
        if (report.get("source_count")!=1 or
                report["sources"][0]["url"]!="https://de.wikipedia.org/?curid=42" or
                report["model_used"] is not False):
            raise ResearchError("Offline-Quellenvertrag fehlgeschlagen")
        print(json.dumps({
            "state":"PASS_OFFLINE",
            "source_count":report["source_count"],
            "network_used":False, "gpu_used":False,
            "model_used":False, "filesystem_written":False,
        },ensure_ascii=False,indent=2))
        return 0
    if not a.query and not a.out and not a.confirm_public:
        print(json.dumps({
            "state":"NO_NETWORK",
            "action":"Specify --query, --out and --confirm-public together",
            "model_used":False, "pc_gpu_required":False,
            "no_background_execution":True,
        },ensure_ascii=False,indent=2))
        return 0
    if not a.confirm_public or not a.query or a.out is None:
        parser.error("All of --query, --out and --confirm-public are required")
    goal=validate_public_query(a.query)
    output=a.out.expanduser()
    # Check obvious mistakes BEFORE the public request; O_EXCL still protects
    # against concurrent writes between this check and the completed lookup.
    if output.exists() or output.is_symlink():
        raise FileExistsError("Research report already exists; choose a new path")
    if not output.parent.is_dir():
        raise ValueError("Output parent must be an existing directory")
    report=run_public_research(goal)
    encoded=(json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    if len(encoded)>12_000:
        raise ResearchError("Research report exceeds bounded output size")
    fd=os.open(output,os.O_WRONLY | os.O_CREAT | os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as file:
        file.write(encoded)
    print(json.dumps({
        "status":"REVIEW_READY","report":str(output),
        "source_count":report["source_count"],
        "requires_review":True, "model_used":False,
        "no_background_execution":True,
    },ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
