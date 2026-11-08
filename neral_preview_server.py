"""NERAL 1.0 Local Preview: tiny local chat server, NO legacy 50M dependency.

All requests stay on loopback. Session+CSRF and Host/Origin controls protect
the local Ollama backend from hostile web pages. No user content is logged,
stored or sent to GitHub. This preview does NOT promote NERAL adapter weights.
"""
from __future__ import annotations

import argparse
import hmac
import secrets
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

ALLOWED_MODELS = ("qwen3:8b", "qwen3.5:9b")
HOST = "127.0.0.1"
PORT = 9690
ORIGIN = f"http://{HOST}:{PORT}"


def create_preview_app(backend=None, *, secret_key=None, search_service=None):
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

    root = Path(__file__).resolve().parent
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

    @app.before_request
    def reject_external_or_cross_origin():
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
    register_neral_local_chat_api(app,backend)

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
            "local_code_execution": {"available": False},
            "project_files": {"available": False},
        })
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
