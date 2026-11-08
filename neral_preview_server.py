"""NERAL 1.0 Local Preview: tiny local chat server, NO legacy 50M dependency.

All requests stay on loopback. Session+CSRF and Host/Origin controls protect
the local Ollama backend from hostile web pages. No user content is logged,
stored or sent to GitHub. This preview does NOT promote NERAL adapter weights.
"""
from __future__ import annotations

import argparse
import ast
import hmac
import secrets
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from werkzeug.exceptions import RequestEntityTooLarge
from neral_preview_attachments import MAX_REQUEST_BYTES

ALLOWED_MODELS = ("qwen3:8b", "qwen3.5:9b")
HOST = "127.0.0.1"
PORT = 9690
ORIGIN = f"http://{HOST}:{PORT}"


def create_preview_app(backend=None, *, secret_key=None, search_service=None, vision_analyzer=None):
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
            "connect-src 'self'; img-src 'self'; font-src 'self'; "
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
                       model_origin="local_foundation_not_neral_adapter",
                       release_validated=False,adapter_promoted=False)

    from neral_local_chat_api import register_neral_local_chat_api
    chat_busy = register_neral_local_chat_api(app,backend)
    from neral_preview_attachments import register_preview_attachment_api
    register_preview_attachment_api(app, backend, chat_busy, vision_analyzer=vision_analyzer)

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
    args=parser.parse_args(argv)
    from neral_agent_autonomy import OllamaLocalBackend
    from waitress import serve
    app=create_preview_app(OllamaLocalBackend(args.model,timeout=90))
    print(f"NERAL 1.0 Local Preview: {ORIGIN} | {args.model} | "
          "local foundation only | no promoted LoRA",flush=True)
    serve(app,host=HOST,port=PORT,threads=4)


if __name__=="__main__":
    main()
