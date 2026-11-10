"""Offline source-linked Helios reference mode, deliberately NOT model-knowledge training.

Answers ONLY exact curated question aliases from a tiny source-linked JSON file.
Unknown or ambiguous questions abstain; nothing is generated or downloaded.
The original own-model preview and raw quality benchmark are left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from helios350_preview import MAX_INPUT_BYTES, parse_request, permitted_origin

MAX_CORPUS_BYTES = 65_536
ALLOWED_HOSTS = frozenset({"science.nasa.gov", "www.nist.gov"})
ABSTAIN = "Keine passende geprüfte Referenzfrage vorhanden. / No matching curated reference question is available."
HTML = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Helios · Referenzwissen</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101625;color:#e9edf6}body{max-width:780px;margin:36px auto;padding:0 16px}p{line-height:1.5;color:#b5c2d5}textarea{width:100%;box-sizing:border-box;padding:14px;min-height:90px;border:1px solid #52627f;border-radius:12px;background:#19253a;color:inherit;font:inherit}button{padding:11px 19px;border:0;border-radius:10px;background:#77b9ff;color:#091222;font-weight:700;cursor:pointer}button:disabled{opacity:.6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#19253a;border:1px solid #384b6b;border-radius:12px;padding:16px;min-height:50px;font:inherit}a{color:#9ccaff}small{color:#aebcd0}</style></head><body>
<h1>Helios · lokales Referenzwissen</h1><p id="status">Prüfe Referenzdatei …</p><p>Dieser Modus gibt ausschließlich hinterlegte Fakten zu exakt erkannten Fragen zurück. Kein generatives Modell, kein Internetzugriff, kein allgemeines Weltwissen. Unbekannte Fragen werden ausdrücklich nicht beantwortet. Quellen können nach dem Prüfdatum geändert worden sein.</p><label for="prompt">Deine Frage</label><textarea id="prompt" maxlength="1024" placeholder="Wie viele Planeten hat unser Sonnensystem?"></textarea><p><button id="send">Referenz prüfen</button></p><h2>Antwort</h2><pre id="reply" aria-live="polite"></pre><p><a id="source" href="#" rel="noopener noreferrer" target="_blank" hidden>Quelle öffnen</a></p><small>Offline, ohne Modelltraining. Antworten sind redaktionelle Kurzfassungen mit verlinkter Quelle.</small>
<script>const $=id=>document.getElementById(id);fetch('/health').then(r=>r.json()).then(d=>{$('status').textContent=`${d.facts} Referenzeinträge · Quellenstand ${d.verified_on} · ${d.mode}`}).catch(()=>{$('status').textContent='Lokaler Referenzserver nicht erreichbar.'});$('send').onclick=async()=>{const p=$('prompt').value.trim();if(!p)return;$('send').disabled=true;$('source').hidden=true;$('source').removeAttribute('href');$('reply').textContent='Prüfe …';try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,max_new_tokens:64})});const d=await r.json();$('reply').textContent=r.ok?d.answer:(d.error||'Anfrage fehlgeschlagen');if(r.ok&&d.source&&d.source.url){$('source').href=d.source.url;$('source').textContent=d.source.title;$('source').hidden=false}}catch(e){$('reply').textContent='Lokale Verbindung fehlgeschlagen.'}finally{$('send').disabled=false}};</script></body></html>"""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def normalized(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("question must be text")
    value = unicodedata.normalize("NFKC", question).casefold().strip()
    value = re.sub(r"[?!\.]+$", "", value).strip()
    return " ".join(value.split())


def _source(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return (parts.scheme == "https" and parts.hostname in ALLOWED_HOSTS
                and parts.port is None and not parts.username and not parts.password
                and parts.path.startswith("/") and not parts.query and not parts.fragment)
    except (TypeError, ValueError):
        return False


class ReferenceStore:
    def __init__(self, path: Path):
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CORPUS_BYTES:
            raise ValueError("missing, symlinked or oversized source-linked corpus")
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        corpus = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(corpus, dict)
                or set(corpus) != {"schema", "classification", "verified_on", "facts"}
                or type(corpus["schema"]) is not int or corpus["schema"] != 1
                or corpus["classification"] != "public_paraphrased_reference_facts_not_training_data"
                or not re.fullmatch(r"20\d\d-\d\d-\d\d", str(corpus["verified_on"]))):
            raise ValueError("unexpected reference corpus schema")
        entries = corpus["facts"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 128:
            raise ValueError("invalid reference count")
        self.aliases: dict[str, tuple[str, str, str, str]] = {}
        self.ids = set()
        self.verified_on = corpus["verified_on"]
        for row in entries:
            if not isinstance(row, dict) or set(row) != {"id", "questions", "answers", "source_title", "source_url"}:
                raise ValueError("invalid reference entry")
            ident = row["id"]
            if not isinstance(ident, str) or not re.fullmatch(r"[a-z0-9_]{3,64}", ident) or ident in self.ids:
                raise ValueError("invalid/duplicate reference ID")
            self.ids.add(ident)
            questions, answers = row["questions"], row["answers"]
            if not isinstance(questions, dict) or not isinstance(answers, dict) or set(questions) != {"de", "en"} or set(answers) != {"de", "en"}:
                raise ValueError("bilingual reference fields required")
            title, url = row["source_title"], row["source_url"]
            if not isinstance(title, str) or not 4 <= len(title) <= 128 or not isinstance(url, str) or not _source(url):
                raise ValueError("source must be an exact allowed HTTPS authority")
            for lang in ("de", "en"):
                answer = answers[lang]
                if not isinstance(answer, str) or not 3 <= len(answer.strip()) <= 400 or "\n" in answer:
                    raise ValueError("invalid reference answer")
                variants = questions[lang]
                if not isinstance(variants, list) or not 1 <= len(variants) <= 4:
                    raise ValueError("invalid aliases")
                for question in variants:
                    key = normalized(question)
                    if (not isinstance(question, str) or not 10 <= len(question) <= 200
                            or not key or key in self.aliases or "\n" in question):
                        raise ValueError("invalid/ambiguous reference alias")
                    self.aliases[key] = (answer.strip(), title, url, ident)
        self.fact_count = len(entries)

    def lookup(self, question: str) -> dict:
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1024:
            raise ValueError("invalid question")
        matched = self.aliases.get(normalized(question))
        if matched is None:
            return {"answer": ABSTAIN, "status": "abstained_no_exact_reference",
                    "source": None, "model_generated": False, "verified_reference_available": False}
        answer, title, url, ident = matched
        return {"answer": answer, "status": "exact_reference_match",
                "source": {"title": title, "url": url}, "reference_id": ident,
                "model_generated": False, "verified_reference_available": True}


def make_handler(store: ReferenceStore, port: int):
    class KnowledgeHandler(BaseHTTPRequestHandler):
        def _send(self, status, raw, content_type):
            self.send_response(status)
            for name, value in (("Content-Type", content_type), ("Content-Length", str(len(raw))),
                                ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                                ("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _allowed(self):
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"} and permitted_origin(self.headers.get("Origin"), port)

        def do_GET(self):
            if not self._allowed():
                return self._json(403, {"error": "localhost only"})
            if self.path == "/":
                return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/health":
                return self._json(200, {"mode": "exact offline reference, not model", "facts": store.fact_count,
                                         "verified_on": store.verified_on, "corpus_sha256": store.sha256})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._allowed():
                return self._json(403, {"error": "localhost only"})
            if self.path != "/generate":
                return self._json(404, {"error": "not found"})
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                return self._json(415, {"error": "JSON required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "invalid request length"})
            if not 1 <= length <= MAX_INPUT_BYTES:
                return self._json(413, {"error": "request too large"})
            try:
                prompt, _ = parse_request(self.rfile.read(length))
                result = store.lookup(prompt)
                result.update({"generated_tokens": 0, "decoder": "reference_lookup_only",
                               "corpus_sha256": store.sha256})
                return self._json(200, result)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception:
                return self._json(500, {"error": "reference lookup failed"})

        def log_message(self, *_):
            pass  # No user prompts in local access logs.
    return KnowledgeHandler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--facts", type=Path, required=True)
    ap.add_argument("--port", type=int, default=9683)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.execute or not 1024 <= args.port <= 65535:
        ap.error("explicit execution and valid port required")
    store = ReferenceStore(args.facts)
    if args.smoke:
        positive = store.lookup("Wie viele Planeten hat unser Sonnensystem?")
        negative = store.lookup("Wie hoch ist die Temperatur in meiner Stadt heute?")
        if (positive["status"] != "exact_reference_match" or positive["answer"] != "Unser Sonnensystem hat acht Planeten."
                or negative["status"] != "abstained_no_exact_reference"):
            raise ValueError("knowledge reference smoke failed")
        print(json.dumps({"event": "reference_smoke_passed", "facts": store.fact_count,
                          "corpus_sha256": store.sha256, "model_loaded": False,
                          "training_started": False, "network_accessed": False}), flush=True)
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, args.port))
    print(f"Reference-only localhost: http://127.0.0.1:{args.port}/ ; Ctrl+C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
