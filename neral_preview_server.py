"""NERAL 1.0 Local Preview: tiny local chat server, NO legacy 50M dependency.

All requests stay on loopback. Session+CSRF and Host/Origin controls protect
the local Ollama backend from hostile web pages. No user content is logged,
stored or sent to GitHub. This preview does NOT promote NERAL adapter weights.
"""
from __future__ import annotations

import argparse
import ast
import hmac
import json
import re
import secrets
import sys
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from werkzeug.exceptions import RequestEntityTooLarge
from neral_preview_attachments import MAX_REQUEST_BYTES

ALLOWED_MODELS = ("qwen3:8b", "qwen3.5:9b")
HOST = "127.0.0.1"
PORT = 9690
ORIGIN = f"http://{HOST}:{PORT}"


def _installed_ollama_models() -> dict[str, str]:
    """Read installed model digests from loopback, never download or send data."""
    from neral_agent_autonomy import local_only_opener
    req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
    with local_only_opener().open(req, timeout=15) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("Lokale Ollama-Modellliste zu groß.")
    data = json.loads(raw)
    rows = data.get("models") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) > 150:
        raise ValueError("Lokale Ollama-Modellliste ungültig.")
    found = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Lokale Modellliste enthält ungültigen Eintrag.")
        name, digest = item.get("name"), item.get("digest")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("Unvollständige lokale Modellkennung.")
        canonical = digest if digest.startswith("sha256:") else "sha256:" + digest
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", canonical):
            raise ValueError("Lokaler Modell-Digest ungültig.")
        if name in found:
            raise ValueError("Doppelte lokale Modellkennung.")
        found[name] = canonical
    return found


def build_experimental_high(low_model: str, enabled: bool):
    """Only explicit opt-in and both models already installed permit HIGH."""
    if not enabled:
        return None
    if low_model != "qwen3:8b":
        raise ValueError("Experimentelles HIGH erfordert Qwen3:8B als LOW.")
    try:
        installed = _installed_ollama_models()
    except Exception as exc:
        raise ValueError("Lokale Ollama-Inventur nicht verfügbar.") from exc
    if not all(re.fullmatch(r"sha256:[0-9a-f]{64}",installed.get(model,""))
               for model in ("qwen3:8b","qwen3.5:9b")):
        raise ValueError("Für HIGH müssen Qwen3:8B und Qwen3.5:9B bereits installiert sein.")
    from neral_agent_autonomy import OllamaLocalBackend
    return OllamaLocalBackend(
        "qwen3.5:9b", timeout=90,
        inference_profile="measured_nonthinking_2048")


def create_preview_app(backend=None, *, secret_key=None, search_service=None, vision_analyzer=None, voice_service=None, experimental_high_backend=None):
    if backend is None:
        from neral_agent_autonomy import OllamaLocalBackend
        backend = OllamaLocalBackend("qwen3:8b")
    name = getattr(backend, "model", None)
    if name not in ALLOWED_MODELS:
        raise ValueError("Only explicitly approved local Qwen foundations allowed")
    if secret_key is None:
        secret_key = secrets.token_bytes(32)
    if not isinstance(secret_key, bytes) or len(secret_key) < 32:
        raise ValueError("Fresh secret key required for local sessions")

    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app = Flask("neral_1_0_preview",template_folder=str(root/"templates"),
                static_folder=str(root/"static"))
    app.secret_key = secret_key
    app.config.update(
        NERAL_MODEL_TAG=name,
        MAX_CONTENT_LENGTH=16384,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=False,  # HTTP is permitted on private loopback only.
    )

    @app.errorhandler(RequestEntityTooLarge)
    def reject_oversized_request(_error):
        # The browser client always parses API responses as JSON; Flask's
        # default HTML 413 would hide the actual upload-limit error.
        return jsonify(error="Anfrage oder Anhang überschreitet das erlaubte Größenlimit."), 413

    @app.before_request
    def reject_external_or_cross_origin():
        if request.path.startswith("/api/neral/account/chats/") and request.method == "PUT":
            request.max_content_length = 130000
        if request.path == "/api/neral/preview/attachment-chat":
            request.max_content_length = MAX_REQUEST_BYTES
        if request.path == "/api/neral/voice/transcribe":
            request.max_content_length = 500_000
        if request.host not in (f"127.0.0.1:{PORT}",):
            return jsonify(error="Nur der lokale NERAL-Dienst ist erlaubt."), 403
        if (request.headers.get("Origin") not in (None, ORIGIN) or
                request.headers.get("Sec-Fetch-Site") in ("cross-site", "none") and
                request.path.startswith("/api/")):
            return jsonify(error="Anfrage von einer fremden Webseite blockiert."), 403
        if request.path.startswith("/api/"):
            token = session.get("csrf")
            if not isinstance(token, str) or len(token) < 24:
                return jsonify(error="Lokale NERAL-Sitzung fehlt."), 403
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                proposed = request.headers.get("X-Neral-CSRF", "")
                if not hmac.compare_digest(proposed, token):
                    return jsonify(error="Lokale Sitzungsprüfung fehlgeschlagen."), 403
        return None

    @app.after_request
    def secure_response(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; media-src 'self' blob:; font-src 'self'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def home():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return render_template("neral_preview.html", model=name)

    @app.get("/api/neral/preview/session")
    def session_info():
        return jsonify(csrf=session["csrf"],model=name,
                       experimental_high_model=(
                           getattr(experimental_high_backend, "model", None)),
                       experimental_high_enabled=experimental_high_backend is not None,
                       model_origin="local_foundation_not_neral_adapter",
                       release_validated=False,adapter_promoted=False)

    from neral_local_chat_api import register_neral_local_chat_api
    chat_busy = register_neral_local_chat_api(
        app,backend,experimental_high_backend=experimental_high_backend)
    from neral_preview_attachments import register_preview_attachment_api
    register_preview_attachment_api(
        app, backend, chat_busy, vision_analyzer=vision_analyzer,
        experimental_high_backend=experimental_high_backend,
        prepare_model=app._neral_prepare_model)

    from neral_local_voice import register_local_voice_api
    local_voice = register_local_voice_api(app, service=voice_service)

    from neral_local_accounts import register_local_accounts
    register_local_accounts(app)

    # Web search stays opt-in: constructing a service or reading status must
    # never send data off the machine. All API routes inherit the same
    # session, CSRF, Host and Origin security checks.
    from neral_web_api import register_web_search_api
    register_web_search_api(app, service=search_service)

    @app.get("/api/neral/preview/plugins")
    def preview_plugins():
        return jsonify(plugins={
            "exact_math": {"available": True, "execution": "local_verified"},
            "think": {"available": True, "execution": "local_foundation_plus_tools"},
            "web_search": {"configured": True, "live_verified": False,
                           "automatic": False, "query_leaves_pc": True,
                           "requires_api_key": False},
            "offline_voice": local_voice.status(),
            "browser_voice": {"configured": True, "live_verified": False,
                              "requires_user_consent": True,
                              "may_use_browser_provider": True},
            "python_syntax_check": {"available": True, "scope": "python_parse_compile_only",
                                    "executes_code": False},
            "selected_file_context": {"available": True,
                                      "requires_user_selection": True,
                                      "supported": ["png","jpg","webp","txt","md","pdf","docx","csv","json","code"],
                                      "vision_requires_local_model": True,
                                      "automatic_model_install": False,
                                      "persistent_filesystem_access": False},
            "local_code_execution": {"available": False},
            "project_files": {"available": False},
        })
    @app.post("/api/neral/preview/codecheck")
    def preview_codecheck():
        # Parse only. This does not run or import user code, write files,
        # invoke a shell, or send anything over a network.
        if not request.is_json:
            return jsonify(error="Nur JSON erlaubt."), 415
        data=request.get_json(silent=True)
        if (not isinstance(data, dict) or set(data)!={"source"} or
                not isinstance(data["source"], str) or not data["source"].strip() or
                len(data["source"].encode("utf-8"))>4096):
            return jsonify(error="Nur Python-Quelltext bis 4096 Bytes erlaubt."), 400
        try:
            # Parsing alone accepts e.g. top-level "return", "break" or
            # "await". Python's compiler rejects them. Compilation creates
            # an in-memory code object only; it does NOT execute, import,
            # write or evaluate the supplied program.
            ast.parse(data["source"], mode="exec")
            compile(data["source"], "<neral-codecheck>", "exec",
                    dont_inherit=True, optimize=0)
        except SyntaxError as exc:
            return jsonify(syntax_ok=False, scope="python_parse_compile_only",
                           code_executed=False, line=exc.lineno or 1,
                           column=exc.offset or 0, message=str(exc.msg)[:120])
        except (ValueError, RecursionError):
            return jsonify(error="Ungültiger Python-Quelltext."), 400
        return jsonify(syntax_ok=True, scope="python_parse_compile_only",
                       code_executed=False)

    return app


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--model",choices=ALLOWED_MODELS,default="qwen3:8b")
    parser.add_argument("--inference-profile", choices=("legacy", "measured_nonthinking_2048"),
                        default="legacy",
                        help="Experimental measured profile; default stays legacy.")
    parser.add_argument("--experimental-high-qwen35",action="store_true",
                        help="Opt-in HIGH with installed Qwen3.5:9b; defaults off.")
    parser.add_argument("--experimental-dual-routing",action="store_true",
                        help="Opt-in pinned QLoRA for TIR and base-Qwen for raw fallback; default off.")
    args=parser.parse_args(argv)
    from neral_agent_autonomy import OllamaLocalBackend
    from waitress import serve
    high=build_experimental_high(args.model,args.experimental_high_qwen35)
    if args.experimental_dual_routing:
        if high is None:
            raise ValueError("Dual-Routing requires explicit HIGH plus two installed pinned models.")
        from scripts.neral_generalization_100_run import require_exact_models
        require_exact_models()  # fail before listening if either digest differs
        import os
        os.environ["NERAL_EXPERIMENTAL_DUAL_ROUTE"]="1"
    app=create_preview_app(OllamaLocalBackend(
        args.model,timeout=90,inference_profile=args.inference_profile),
        experimental_high_backend=high)
    print(f"NERAL 1.0 Local Preview: {ORIGIN} | LOW={args.model} | "
          f"HIGH={getattr(high,'model',args.model)} | "
          f"inference_profile={args.inference_profile} | "
          f"dual_routing_opt_in={args.experimental_dual_routing} | "
          "no default model promotion",flush=True)
    serve(app,host=HOST,port=PORT,threads=4)


if __name__=="__main__":
    main()
