from __future__ import annotations

"""Read-only localhost preview of SHA-pinned, own Neral 350M weights.

No training, checkpoint writes, downloads, cloud services or production claims.
Only use local checkpoints whose exact SHA256 was recorded independently.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

SAFE_SHA = "ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c"
CONTINUITY_SHA = "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
TOKENIZER_SHA = "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
SAFE_RELATIVE = Path("runs/neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected/resume.pt")
MAX_INPUT_BYTES = 4096
MAX_PROMPT_CHARS = 1024
MAX_INPUT_TOKENS = 384  # Trained seq length 512; reserve space for response.
MAX_NEW_TOKENS = 64

HTML = """<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Helios 350M · lokale Vorschau</title><style>
:root{font-family:system-ui,sans-serif;color-scheme:dark;background:#101625;color:#e9edf6}body{max-width:780px;margin:36px auto;padding:0 16px}h1{font-size:1.8rem}p{line-height:1.5;color:#b5c2d5}textarea{width:100%;box-sizing:border-box;padding:14px;min-height:108px;border:1px solid #52627f;border-radius:12px;background:#19253a;color:inherit;font:inherit}button{padding:11px 19px;border:0;border-radius:10px;background:#77b9ff;color:#091222;font-weight:700;cursor:pointer}button:disabled{opacity:.6;cursor:wait}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#19253a;border:1px solid #384b6b;border-radius:12px;padding:16px;min-height:80px;font:inherit}small{color:#aebcd0}</style></head><body>
<h1>Helios · echte Neral-350M-Vorschau</h1><p id="status">Lade Modellstatus …</p><p>Experimentelle eigene Gewichte. Antworten können fehlerhaft sein, sich wiederholen oder nicht enden. Dies ist kein Qualitäts- oder Sicherheitstest.</p><label for="prompt">Deine Nachricht</label><textarea id="prompt" maxlength="1024" placeholder="Hallo, wer bist du?"></textarea><p><button id="send">Mit Neral testen</button></p><small>Maximal 64 neue Tokens. Die Gewichte verbleiben auf diesem PC.</small><h2>Modellantwort</h2><pre id="reply" aria-live="polite"></pre>
<script>const $=id=>document.getElementById(id);fetch('/health').then(r=>r.json()).then(x=>{$('status').textContent=`${x.model} · ${x.processed_tokens.toLocaleString('de-DE')} Tokens · ${x.device} · ${x.status}`}).catch(()=>{$('status').textContent='Keine Verbindung zum lokalen Modell.'});$('send').onclick=async()=>{const p=$('prompt').value.trim();if(!p)return;$('send').disabled=true;$('reply').textContent='Generiere …';try{const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,max_new_tokens:64})});const d=await r.json();$('reply').textContent=r.ok?d.answer:(d.error||'Anfrage fehlgeschlagen');}catch(e){$('reply').textContent='Lokale Verbindung fehlgeschlagen.'}finally{$('send').disabled=false;}};</script></body></html>"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trusted_asset(root: Path, path: Path, expected_sha: str) -> Path:
    root = root.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked local asset")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("asset must be inside local Neral assets root")
    if sha256_file(resolved) != expected_sha:
        raise ValueError("own asset SHA256 mismatch: refuse to load")
    return resolved


def parse_request(raw: bytes) -> tuple[str, int]:
    if not 0 < len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("invalid request size")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    if not isinstance(obj, dict) or set(obj) != {"prompt", "max_new_tokens"}:
        raise ValueError("expected prompt and max_new_tokens")
    prompt, count = obj["prompt"], obj["max_new_tokens"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("prompt must contain 1–1024 characters")
    if type(count) is not int or not 1 <= count <= MAX_NEW_TOKENS:
        raise ValueError("max_new_tokens must be between 1 and 64")
    return prompt.strip(), count


def permitted_origin(origin: str | None, port: int) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        return (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
                and parsed.port == port and parsed.path in {"", "/"}
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment)
    except ValueError:  # malformed URL/port must return 403, not crash the server.
        return False


def load_verified_model(root: Path, checkpoint: Path, model_kind: str, device_name: str):
    # Unsafe Torch pickle is decoded only AFTER verifying a pinned known-local SHA.
    expected = SAFE_SHA if model_kind == "safe" else CONTINUITY_SHA
    checkpoint = trusted_asset(root, checkpoint, expected)
    tok_path = trusted_asset(root, root / "frontier350_assets" / "tokenizer.json", TOKENIZER_SHA)
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350, Neral350M
    if device_name == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA BF16 unavailable; use --device cpu")
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size() != FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    if bos is None or eos is None:
        raise ValueError("own tokenizer BOS/EOS missing")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("config") != FRONTIER_350.to_dict() or state.get("tokenizer_sha256") != TOKENIZER_SHA:
        raise ValueError("checkpoint architecture/tokenizer mismatch")
    processed = state.get("processed_tokens")
    if type(processed) is not int or (model_kind == "safe" and processed != 1_010_003_584) or (model_kind != "safe" and not 2_000_000_000 < processed < 2_200_000_000):
        raise ValueError("unexpected checkpoint token count")
    if not isinstance(state.get("model"), dict):
        raise ValueError("missing model weights")
    model = Neral350M()
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.to(device=torch.device(device_name), dtype=torch.bfloat16 if device_name == "cuda" else torch.float32).eval()
    return model, tok, int(bos), int(eos), processed, torch


def generate(model, tok, bos: int, eos: int, torch, prompt: str, count: int) -> tuple[str, int]:
    ids = [bos] + tok.encode("User: " + prompt + "\nAssistant:").ids
    if len(ids) > MAX_INPUT_TOKENS:
        raise ValueError("message exceeds 384 encoded tokens")
    result = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for _ in range(count):
            logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
            token = int(logits[0, -1].float().argmax().item())
            if token == eos:
                break
            result.append(token)
            ids.append(token)
    return tok.decode(result, skip_special_tokens=True), len(result)


def make_handler(model, tok, bos: int, eos: int, torch, processed: int, model_kind: str, device: str, port: int):
    mutex = threading.Lock()
    class PreviewHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, data: bytes, content_type: str):
            self.send_response(status)
            for name, value in (("Content-Type", content_type), ("Content-Length", str(len(data))),
                                ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                                ("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")):
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)
        def _json(self, status, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        def _allowed(self):
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"} and permitted_origin(self.headers.get("Origin"), port)
        def do_GET(self):
            if not self._allowed():
                return self._json(403, {"error": "localhost access only"})
            if self.path == "/":
                return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/health":
                return self._json(200, {"status": "experimental; not approved", "model": "Neral 350M SAFE" if model_kind == "safe" else "Neral 350M 2B research; NOT SAFE", "processed_tokens": processed, "device": device})
            self._json(404, {"error": "not found"})
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
            if length < 1 or length > MAX_INPUT_BYTES:
                return self._json(413, {"error": "request too large"})
            try:
                prompt, count = parse_request(self.rfile.read(length))
                with mutex:
                    answer, used = generate(model, tok, bos, eos, torch, prompt, count)
                self._json(200, {"answer": answer, "generated_tokens": used, "model": model_kind})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "local inference failed; check the terminal"})
        def log_message(self, *_):
            pass  # No prompts in logs.
    return PreviewHandler


def main():
    ap = argparse.ArgumentParser(description="Read-only localhost preview of own Neral 350M weights")
    ap.add_argument("--assets-root", default=os.getenv("NERAL_ASSETS_ROOT"), help="Existing private Neral asset directory (or NERAL_ASSETS_ROOT)")
    ap.add_argument("--checkpoint", help="Explicit own checkpoint path; mandatory for continuity-2b")
    ap.add_argument("--model", choices=("safe", "continuity-2b"), default="safe")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--port", type=int, default=9681)
    ap.add_argument("--smoke", action="store_true", help="One-token actual checkpoint generation; no server")
    args = ap.parse_args()
    if not args.assets_root:
        ap.error("--assets-root or NERAL_ASSETS_ROOT is required")
    if args.model == "continuity-2b" and not args.checkpoint:
        ap.error("--checkpoint is required for the unapproved continuity-2b candidate")
    if not 1024 <= args.port <= 65535:
        ap.error("port must be 1024..65535")
    root = Path(args.assets_root)
    checkpoint = Path(args.checkpoint) if args.checkpoint else root / SAFE_RELATIVE
    model, tok, bos, eos, processed, torch = load_verified_model(root, checkpoint, args.model, args.device)
    print(f"Verified own Neral 350M {args.model}: {processed:,} tokens; device={args.device}; read-only", flush=True)
    if args.smoke:
        answer, used = generate(model, tok, bos, eos, torch, "Hallo", 1)
        print(json.dumps({"smoke": "real_350m_inference_completed", "generated_tokens": used, "answer": answer, "model": args.model, "training_started": False}, ensure_ascii=False), flush=True)
        return
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, tok, bos, eos, torch, processed, args.model, args.device, args.port))
    print(f"Open http://127.0.0.1:{args.port}/ on THIS computer. Ctrl+C stops preview.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
