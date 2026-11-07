"""NERAL Mini: private, offline, single-user Windows chat with verified local LoRA.

This small app deliberately does NOT use the legacy 20M/50M Helios server.
The user's Qwen2.5-Coder-7B snapshot is only the foundation; responses use
a separately SHA-verified, explicitly selected NERAL LoRA adapter. There is
NO cloud-model fallback, download, training, code execution or file access
tool in the chat. Only localhost HTTP is exposed.
"""
from __future__ import annotations

import argparse
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
import webbrowser

BASE_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
# Proven-bad experimental weights may still be evaluated in isolated research jobs,
# but must never be loaded through normal sealed runtime adapter validation.
QUARANTINED_ADAPTER_SHA256 = frozenset({
    "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd",
})
MAX_PROMPT_BYTES = 4096
MAX_HISTORY = 12
MAX_BODY = 16000
MAX_TOKENS = 224
TOKEN_LIMIT = 3072
# Keep this policy self-contained: the downloadable Mini ZIP deliberately ships
# without root-level Python modules. Technical origins remain in status metadata.
SYSTEM = (
    "Du bist NERAL AI, ein experimenteller, lokal ausgefuehrter Assistent. "
    "Bei Fragen wie 'Wer bist du?' stelle dich als NERAL AI vor und erwaehne "
    "Modellhersteller oder Foundation nicht unaufgefordert. "
    "Erklaere ehrlich deine tatsaechlichen Aufgaben, ohne Funktionen zu erfinden. "
    "Nur wenn jemand explizit nach Modellbasis, Herkunft oder Training fragt, "
    "erklaere: Qwen2.5-Coder-7B ist die Foundation; ein eigener experimenteller "
    "NERAL-LoRA-Adapter ist aktiv. Es ist kein von Grund auf selbst trainiertes "
    "Basismodell; ein Qualitaetsgewinn ist nicht automatisch bewiesen. "
    "Gib Unsicherheit zu, erfinde keine Quellen oder ausgefuehrten Aktionen. "
    "In diesem Mini-Chat gibt es keine Websuche und keinen Computerzugriff."
)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}{% endif %}"
)


class MiniError(ValueError):
    pass


class MiniCancelled(MiniError):
    """User-requested generation stop; no model answer is published."""
    pass


def offline() -> None:
    # Set BEFORE importing any HF code. Local weights, no remote model access.
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1",
    })


def _bounded_json(path: Path, limit: int = 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise MiniError("Fehlende oder ungueltige lokale Modelldaten.")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MiniError("Doppelter Metadatenschluessel.")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise MiniError("Metadaten muessen ein JSON-Objekt sein.")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _local_dir(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise MiniError("Lokaler Modellordner fehlt oder ist ein Link.")
    return path.resolve(strict=True)


def verify_assets(base: Path, adapter: Path) -> dict:
    """Verify the sealed local Qwen inventory and compatible NERAL LoRA."""
    base, adapter = _local_dir(base), _local_dir(adapter)
    manifest = _bounded_json(base / "helios-manifest.json")
    if (set(manifest) != {"base_model", "revision", "files"}
        or manifest["base_model"] != BASE_ID
        or manifest["revision"] != BASE_REVISION
        or not isinstance(manifest["files"], dict)):
        raise MiniError("Qwen-Snapshot ist nicht die erwartete verifizierte Version.")
    files = manifest["files"]
    if not 5 <= len(files) <= 64:
        raise MiniError("Ungueltiges Qwen-Dateiverzeichnis.")
    found = {p.name for p in base.iterdir()}
    if found != set(files) | {"helios-manifest.json"}:
        raise MiniError("Lokale Qwen-Dateien stimmen nicht mit dem Manifest ueberein.")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(files):
        raise MiniError("Qwen-Konfiguration oder Tokenizer fehlt.")
    for filename, expected in files.items():
        if (not isinstance(filename, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", filename)
            or filename.startswith(".")
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise MiniError("Ungueltiger Qwen-Manifest-Eintrag.")
        file = base / filename
        if file.is_symlink() or not file.is_file() or _sha256(file) != expected:
            raise MiniError("Qwen-Datei ist unvollstaendig oder nicht verifiziert.")
    conf = _bounded_json(base / "config.json")
    if (conf.get("model_type") != "qwen2"
        or conf.get("hidden_size") != 3584
        or conf.get("num_hidden_layers") != 28
        or conf.get("vocab_size") != 152064
        or "auto_map" in conf):
        raise MiniError("Unbekannte oder nicht freigegebene Qwen-Architektur.")
    identity = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seal = _bounded_json(adapter / "neral-adapter.json")
    if (seal.get("schema") != 1 or seal.get("product") != "NERAL"
        or seal.get("base_model") != BASE_ID
        or seal.get("base_revision") != BASE_REVISION
        or seal.get("base_snapshot_integrity") != identity
        or not isinstance(seal.get("release_validated"), bool)
        or not isinstance(seal.get("files"), dict)
        or set(seal["files"]) != {"adapter_config.json", "adapter_model.safetensors"}):
        raise MiniError("NERAL-Adapter besitzt keine passende Herkunftsversiegelung.")
    allowed = {"neral-adapter.json", "adapter_config.json", "adapter_model.safetensors",
               "neral-training-manifest.json", "training-report.json", "README.md"}
    if any(f.name not in allowed or f.is_symlink() or not f.is_file() for f in adapter.iterdir()):
        raise MiniError("Unerwartete Datei im Adapterordner.")
    for filename, expected in seal["files"].items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise MiniError("Ungueltiger NERAL-Adapter-Hash.")
        file = adapter / filename
        if not file.is_file() or file.is_symlink() or file.stat().st_size > 512*1024**2:
            raise MiniError("Adaptergewicht fehlt oder ist zu gross.")
        if _sha256(file) != expected:
            raise MiniError("Adapter-Integritaet fehlgeschlagen.")
    if seal["files"]["adapter_model.safetensors"] in QUARANTINED_ADAPTER_SHA256:
        raise MiniError("NERAL-Adapter wegen gemessener Regression gesperrt.")
    config = _bounded_json(adapter / "adapter_config.json")
    rank = config.get("r")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
        or type(rank) is not int or rank not in (16, 32)
        or config.get("bias") != "none"
        or config.get("use_dora", False)
        or config.get("modules_to_save")
        or set(config.get("target_modules", [])) !=
          {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}):
        raise MiniError("Adapter ist keine freigegebene NERAL-LoRA-Konfiguration.")
    return {
        "rank": rank,
        "adapter_sha256": seal["files"]["adapter_model.safetensors"],
        "release_validated": seal["release_validated"],
        "base_revision": BASE_REVISION,
    }


def parse_chat(data: object) -> list[dict[str, str]]:
    if not isinstance(data, dict) or set(data) not in ({"prompt"}, {"prompt", "history"}):
        raise MiniError("Ungueltiger Chat-Auftrag.")
    prompt = data["prompt"]
    history = data.get("history", [])
    if (not isinstance(prompt, str) or not prompt.strip()
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
        or not isinstance(history, list) or len(history) > MAX_HISTORY
        or len(history) % 2):
        raise MiniError("Text oder Chatverlauf ist zu lang oder ungueltig.")
    result = [{"role": "system", "content": SYSTEM}]
    for i, entry in enumerate(history):
        if not isinstance(entry, dict) or set(entry) != {"role", "content"}:
            raise MiniError("Unerlaubter Verlaufseintrag.")
        expected_role = "user" if i % 2 == 0 else "assistant"
        value = entry["content"]
        if (entry["role"] != expected_role or not isinstance(value, str)
            or not value.strip() or len(value.encode("utf-8")) > 1800):
            raise MiniError("Unerlaubte Rolle oder zu grosser Verlauf.")
        result.append({"role": entry["role"], "content": value})
    result.append({"role": "user", "content": prompt.strip()})
    return result


class NeralModel:
    def __init__(self, base: Path, adapter: Path):
        self.base = Path(base)
        self.adapter = Path(adapter)
        self.model = None
        self.tokenizer = None
        self.identity = None
        self.error = None
        self._cancel_event = threading.Event()

    def set_cancel_event(self, event: threading.Event) -> None:
        self._cancel_event = event

    def _cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def status(self) -> dict:
        present = (self.base / "helios-manifest.json").is_file() and (
            self.adapter / "neral-adapter.json").is_file()
        return {
            "name": "NERAL Mini",
            "foundation": BASE_ID,
            "adapter_configured": bool(present),
            "adapter_loaded": self.model is not None,
            "adapter_sha256": (self.identity or {}).get("adapter_sha256") if self.model is not None else None,
            "rank": (self.identity or {}).get("rank") if self.model is not None else None,
            "local_only": True,
            "experimental": True,
            "quality_improvement_proven": False,
            "inference_available": bool(present and self.error is None),
        }

    def _load(self):
        if self.model is not None:
            return
        offline()
        identity = verify_assets(self.base, self.adapter)
        import torch
        from transformers import BitsAndBytesConfig, Qwen2ForCausalLM, Qwen2TokenizerFast
        from peft import PeftModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise MiniError("NERAL benoetigt deine CUDA-GPU mit BF16.")
        free, total = torch.cuda.mem_get_info(0)
        if total < 10*1024**3 or free < 9*1024**3:
            raise MiniError("Zu wenig freier GPU-Speicher. GPU-Training und andere Modelle zuerst beenden.")
        tok = Qwen2TokenizerFast.from_pretrained(
            str(self.base), local_files_only=True, trust_remote_code=False, token=False
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2ForCausalLM.from_pretrained(
            str(self.base), local_files_only=True, trust_remote_code=False,
            token=False, use_safetensors=True, device_map={"": 0},
            attn_implementation="sdpa", dtype=torch.bfloat16, quantization_config=quant,
        )
        model = PeftModel.from_pretrained(
            model, str(self.adapter), is_trainable=False, local_files_only=True
        )
        model.eval()
        self.identity = identity
        self.tokenizer = tok
        self.model = model

    def respond(self, messages: list[dict[str, str]]) -> str:
        offline()
        if self._cancel_requested():
            raise MiniCancelled("Abgebrochen.")
        self._load()
        if self._cancel_requested():
            raise MiniCancelled("Abgebrochen.")
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        stop_event = self._cancel_event
        class StopOnRequest(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop_event.is_set()
        tokens = self.tokenizer.apply_chat_template(
            messages, chat_template=CHAT_TEMPLATE, tokenize=True,
            add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")
        if tokens.shape[-1] + MAX_TOKENS > TOKEN_LIMIT:
            raise MiniError("Chatkontext zu lang. Bitte neuen Chat beginnen.")
        if self._cancel_requested():
            raise MiniCancelled("Abgebrochen.")
        with torch.inference_mode():
            generated = self.model.generate(
                tokens, attention_mask=torch.ones_like(tokens),
                do_sample=False, max_new_tokens=MAX_TOKENS,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnRequest()]),
            )
        if self._cancel_requested():
            raise MiniCancelled("Abgebrochen.")
        answer = self.tokenizer.decode(generated[0, tokens.shape[-1]:], skip_special_tokens=True)
        if not answer.strip():
            raise MiniError("Das Modell hat keine Antwort erzeugt.")
        return answer


class MiniServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, backend, html: str):
        self.backend = backend
        self.html = html
        self.token = secrets.token_urlsafe(36)
        self.busy = threading.Lock()
        self.cancel_event = threading.Event()
        if callable(getattr(backend, "set_cancel_event", None)):
            backend.set_cancel_event(self.cancel_event)
        super().__init__(address, MiniHandler)

    def request_cancel(self) -> bool:
        # Same event is consumed by the desktop and mobile GPU inference.
        if not self.busy.locked():
            return False
        self.cancel_event.set()
        return True


class MiniHandler(BaseHTTPRequestHandler):
    server: MiniServer

    def log_message(self, fmt, *args):
        # Never log user prompts, local file paths or model answers.
        pass

    def _origin(self):
        return "http://127.0.0.1:" + str(self.server.server_address[1])

    def _host_ok(self):
        return self.headers.get("Host", "") == "127.0.0.1:" + str(self.server.server_address[1])

    def _send(self, status: int, data: bytes, *, mime: str, cookie: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'self'")
        self.send_header("X-Frame-Options", "DENY")
        if cookie:
            self.send_header("Set-Cookie", "neral_session=" + self.server.token +
                             "; Path=/; HttpOnly; SameSite=Strict")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, body: dict):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self._send(status, raw, mime="application/json; charset=utf-8")

    def do_GET(self):
        if not self._host_ok():
            self._json(403, {"error": "Nur lokaler Zugriff erlaubt."})
        elif self.path == "/":
            raw = self.server.html.replace("%%NERAL_SESSION%%", self.server.token)
            self._send(200, raw.encode("utf-8"), mime="text/html; charset=utf-8", cookie=True)
        elif self.path == "/api/status":
            self._json(200, self.server.backend.status())
        else:
            self._json(404, {"error": "Nicht gefunden."})

    def do_POST(self):
        if not self._host_ok() or self.path not in ("/api/chat", "/api/cancel"):
            self._json(403, {"error": "Lokaler Aufruf verweigert."})
            return
        if (self.headers.get("Origin") != self._origin()
            or self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none")):
            self._json(403, {"error": "Fremde Webseite blockiert."})
            return
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
            session = jar.get("neral_session")
            valid_cookie = session is not None and secrets.compare_digest(
                session.value, self.server.token)
        except (cookies.CookieError, ValueError):
            valid_cookie = False
        if (not valid_cookie or not secrets.compare_digest(
            self.headers.get("X-Neral-Session", ""), self.server.token)):
            self._json(403, {"error": "Chat-Sitzung fehlt. Seite neu laden."})
            return
        if self.path == "/api/cancel":
            if self.server.request_cancel():
                self._json(202, {"cancel_requested": True})
            else:
                self._json(409, {"error": "Keine laufende Anfrage zum Abbrechen."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._json(415, {"error": "Nur JSON erlaubt."})
            return
        try:
            count = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            count = -1
        if not 1 <= count <= MAX_BODY:
            self._json(413, {"error": "Chat-Anfrage zu gross."})
            return
        try:
            messages = parse_chat(json.loads(self.rfile.read(count)))
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            self._json(400, {"error": "Ungueltige Nachricht oder Verlauf."})
            return
        if not self.server.busy.acquire(blocking=False):
            self._json(429, {"error": "NERAL bearbeitet bereits eine Frage."})
            return
        try:
            self.server.cancel_event.clear()
            try:
                started = time.monotonic()
                response = self.server.backend.respond(messages)
                if not isinstance(response, str):
                    raise MiniError("Keine gueltige Modellantwort.")
                self._json(200, {
                    "text": response, "seconds": round(time.monotonic()-started, 2),
                    **self.server.backend.status(),
                })
            except MiniCancelled:
                self._json(409, {"cancelled": True, "error": "Antwort abgebrochen."})
            except Exception:
                # No filesystem paths, traceback, credentials or raw prompts in HTTP.
                self._json(503, {"error": "Lokale Inferenz fehlgeschlagen. Gewichte, GPU-Speicher und Python-Pakete pruefen."})
        finally:
            self.server.busy.release()


def create_server(backend, *, port: int = 9691, html: str | None = None):
    if not 1024 <= port <= 65535:
        raise MiniError("Ungueltiger lokaler Port.")
    if html is None:
        html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    return MiniServer(("127.0.0.1", port), backend, html)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NERAL Mini: lokaler NERAL-LoRA-Chat")
    parser.add_argument("--port", type=int, default=int(os.environ.get("NERAL_MINI_PORT", "9691")))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    home = Path.home()
    default_base = home / ".cache" / "neral" / "qwen25-smoke" / BASE_REVISION / "installed_models" / "hf-qwen25-coder-7b"
    default_adapter = home / "NERAL-QLoRA-Research" / "actions-r32-42078722575-1" / "adapter"
    backend = NeralModel(
        Path(os.environ.get("NERAL_MINI_BASE_PATH", str(default_base))),
        Path(os.environ.get("NERAL_MINI_ADAPTER_PATH", str(default_adapter))),
    )
    offline()
    server = create_server(backend, port=args.port)
    url = "http://127.0.0.1:" + str(args.port) + "/"
    print("NERAL Mini ist lokal erreichbar: " + url, flush=True)
    if not backend.status()["adapter_configured"]:
        print("Hinweis: Lokaler NERAL-LoRA-Adapter oder Qwen-Snapshot fehlt.", flush=True)
        print("Pfad mit NERAL_MINI_BASE_PATH und NERAL_MINI_ADAPTER_PATH setzen.", flush=True)
    print("Keine Cloud-Inferenz. Beenden mit Strg+C.", flush=True)
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(0.7), webbrowser.open(url)),
                         daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.15)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
