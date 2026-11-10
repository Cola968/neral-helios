"""Optional Wikipedia-grounded Helios preview using own pinned Neral weights.

Knowledge mode is explicit: when enabled, the user's query is sent to Wikipedia's
MediaWiki API to retrieve a short DE/EN extract. No third-party model/weights are
used. Raw/guarded local modes remain separate. No training or checkpoint writes.
"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import helios350_preview as base
from helios_guarded_preview import guarded_generate, load_continuity_cpu

MAX_WIKI_EXTRACT_CHARS = 900
WIKI_TIMEOUT_SECONDS = 6
USER_AGENT = "Neral-Helios/0.3 knowledge-preview (local research chatbot)"
GERMAN_HINT = re.compile(r"\b(?:was|wer|wie|wo|warum|welche|welcher|welches|ist|sind|hat|haben|der|die|das|ein|eine)\b|[äöüß]", re.I)


def detect_language(text: str) -> str:
    return "de" if GERMAN_HINT.search(text) else "en"


def wikipedia_lookup(query: str, language: str | None = None) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > base.MAX_PROMPT_CHARS:
        raise ValueError("invalid knowledge query")
    language = language or detect_language(query)
    if language not in {"de", "en"}:
        raise ValueError("knowledge language must be de or en")
    params = urlencode({
        "action": "query", "generator": "search", "gsrsearch": query.strip(),
        "gsrlimit": 1, "prop": "extracts|info", "exintro": 1, "explaintext": 1,
        "inprop": "url", "format": "json", "utf8": 1,
    })
    url = f"https://{language}.wikipedia.org/w/api.php?{params}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=WIKI_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise ValueError("Wikipedia request failed")
            raw = response.read(1_000_001)
    except Exception as exc:
        raise ValueError("Wikipedia lookup unavailable") from exc
    if len(raw) > 1_000_000:
        raise ValueError("Wikipedia response too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid Wikipedia response") from exc
    pages = data.get("query", {}).get("pages", {}) if isinstance(data, dict) else {}
    if not isinstance(pages, dict) or not pages:
        raise ValueError("no Wikipedia result")
    page = next(iter(pages.values()))
    title = page.get("title")
    extract = page.get("extract")
    if not isinstance(title, str) or not isinstance(extract, str) or not extract.strip():
        raise ValueError("Wikipedia result has no usable extract")
    extract = re.sub(r"\s+", " ", extract).strip()[:MAX_WIKI_EXTRACT_CHARS]
    page_url = f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    return {"language": language, "title": title, "extract": extract, "url": page_url}


def _compose_grounded(question: str, source: dict, compact: bool = False) -> str:
    lang = source["language"]
    if lang == "de":
        instruction = ("Nutze den Wikipedia-Auszug als Faktenquelle. Antworte nur aus dem Kontext; "
                       "wenn er nicht reicht, sage das kurz.\n" if compact else
                       "Nutze den folgenden Wikipedia-Auszug als Faktenquelle. Wenn die Antwort darin nicht steht, "
                       "sage kurz, dass der Kontext nicht reicht. Erfinde keine zusätzlichen Fakten.\n")
        return instruction + f"Quelle: {source['title']}\nKontext: {source['extract']}\nFrage: {question}\nAntwort:"
    instruction = ("Use the Wikipedia extract as the factual source. Answer only from context; say briefly if it is insufficient.\n"
                   if compact else
                   "Use the following Wikipedia extract as the factual source. If it does not contain the answer, "
                   "briefly say the context is insufficient. Do not invent extra facts.\n")
    return instruction + f"Source: {source['title']}\nContext: {source['extract']}\nQuestion: {question}\nAnswer:"


def grounded_prompt(question: str, source: dict, tokenizer, bos: int) -> str:
    """Fit retrieved grounding under BOTH guarded char and model-token ceilings."""
    if not isinstance(question, str) or not question.strip() or len(question) > base.MAX_PROMPT_CHARS:
        raise ValueError("invalid grounded question")
    source = dict(source)
    compact = False
    while True:
        text = _compose_grounded(question.strip(), source, compact)
        encoded = [bos] + tokenizer.encode("User: " + text + "\nAssistant:").ids
        if len(text) <= base.MAX_PROMPT_CHARS and len(encoded) <= base.MAX_INPUT_TOKENS:
            return text
        current = source.get("extract", "")
        if not isinstance(current, str) or len(current) < 180:
            raise ValueError("retrieved context cannot fit model input limits")
        compact = True
        source["extract"] = current[: max(160, int(len(current) * 0.78))].rstrip()


def parse_knowledge_request(raw: bytes) -> tuple[str, int, bool]:
    if not 0 < len(raw) <= base.MAX_INPUT_BYTES:
        raise ValueError("invalid request size")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"prompt", "max_new_tokens", "knowledge"}:
        raise ValueError("expected prompt, max_new_tokens and knowledge")
    prompt, count, knowledge = obj["prompt"], obj["max_new_tokens"], obj["knowledge"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > base.MAX_PROMPT_CHARS:
        raise ValueError("prompt must contain 1–1024 characters")
    if type(count) is not int or not 1 <= count <= base.MAX_NEW_TOKENS:
        raise ValueError("max_new_tokens must be between 1 and 64")
    if type(knowledge) is not bool:
        raise ValueError("knowledge must be boolean")
    return prompt.strip(), count, knowledge


def make_handler(model, tokenizer, bos, eos, torch, processed, port):
    mutex = threading.Lock()
    parent = base.make_handler(model, tokenizer, bos, eos, torch, processed, "continuity-2b", "cpu", port)
    html = """<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Helios Knowledge</title><style>body{font-family:system-ui;max-width:800px;margin:35px auto;padding:0 16px;background:#101625;color:#e9edf6}textarea{width:100%;min-height:110px;background:#19253a;color:inherit;border:1px solid #52627f;border-radius:10px;padding:12px;box-sizing:border-box}button{padding:10px 18px}pre{white-space:pre-wrap;background:#19253a;padding:14px;border-radius:10px}.muted{color:#aebcd0}</style></head><body><h1>Helios Knowledge · experimentell</h1><p class=\"muted\">Eigene Neral-Gewichte. Knowledge Mode nutzt optional Wikipedia; dabei wird deine Frage an Wikipedia gesendet. Ohne Haken bleibt die Anfrage lokal.</p><textarea id=\"p\" maxlength=\"1024\" placeholder=\"Frage...\"></textarea><p><label><input id=\"k\" type=\"checkbox\"> Knowledge Mode (Wikipedia)</label></p><button id=\"b\">Senden</button><h2>Antwort</h2><pre id=\"a\"></pre><p id=\"s\" class=\"muted\"></p><script>b.onclick=async()=>{const q=p.value.trim();if(!q)return;b.disabled=true;a.textContent='Generiere …';s.textContent='';try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:q,max_new_tokens:64,knowledge:k.checked})});const d=await r.json();a.textContent=r.ok?d.answer:(d.error||'Fehler');if(d.source){const x=document.createElement('a');x.href=d.source.url;x.target='_blank';x.rel='noopener noreferrer';x.textContent='Quelle: '+d.source.title;s.replaceChildren(x);}}catch(e){a.textContent='Verbindung fehlgeschlagen.'}finally{b.disabled=false;}};</script></body></html>"""

    class KnowledgeHandler(parent):
        def do_GET(self):
            if not self._allowed():
                return self._json(403, {"error": "localhost access only"})
            if self.path == "/":
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return super().do_GET()

        def do_POST(self):
            if not self._allowed():
                return self._json(403, {"error": "localhost access only"})
            if self.path != "/generate":
                return self._json(404, {"error": "not found"})
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                return self._json(415, {"error": "application/json required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "invalid length"})
            if not 1 <= length <= base.MAX_INPUT_BYTES:
                return self._json(413, {"error": "request too large"})
            try:
                prompt, count, knowledge = parse_knowledge_request(self.rfile.read(length))
                source = None
                model_prompt = prompt
                if knowledge:
                    source = wikipedia_lookup(prompt)
                    model_prompt = grounded_prompt(prompt, source, tokenizer, bos)
                with mutex:
                    answer, used, reason = guarded_generate(model, tokenizer, bos, eos, torch, model_prompt, count)
                payload = {"answer": answer, "generated_tokens": used, "finish_reason": reason,
                           "model": "continuity-2b", "decoder": "guarded", "knowledge_mode": knowledge}
                if source:
                    payload["source"] = {"title": source["title"], "url": source["url"], "language": source["language"]}
                return self._json(200, payload)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception:
                return self._json(500, {"error": "knowledge preview failed"})
    return KnowledgeHandler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--trusted-local-checkpoint", action="store_true")
    ap.add_argument("--assets-root", default=os.getenv("NERAL_ASSETS_ROOT"))
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--port", type=int, default=9683)
    args = ap.parse_args()
    if not args.execute or not args.trusted_local_checkpoint or not args.assets_root:
        ap.error("explicit execution, own checkpoint trust and assets root required")
    if not 1024 <= args.port <= 65535:
        ap.error("invalid port")
    model, tok, bos, eos, processed, torch = load_continuity_cpu(Path(args.assets_root), args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, tok, bos, eos, torch, processed, args.port))
    print(f"Helios Knowledge preview: http://127.0.0.1:{args.port}/ ; Ctrl+C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
