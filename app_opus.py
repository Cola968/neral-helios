"""Helios launcher with optional Claude Opus 5 via Experiential Labs.

This wraps the existing local app instead of replacing Neral. Local Neral,
SOLVE, code checks, projects, storage and security keep their existing paths.
Claude is an explicitly selectable cloud model and the API key never reaches
the browser.
"""
from __future__ import annotations

import re
import time

from flask import jsonify, request
from waitress import serve

import app as local
from experiential_client import (
    ExperientialError,
    chat as experiential_chat,
    configured as experiential_configured,
    model_slug as experiential_model_slug,
)

app = local.app
_original_index = app.view_functions["index"]
_original_generate = app.view_functions["generate"]


def _cloud_info() -> dict:
    return {
        "id": experiential_model_slug(),
        "label": "Claude Opus 5",
        "provider": "Experiential Labs",
        "available": experiential_configured(),
        "remote": True,
    }


def index_with_opus():
    response = _original_index()
    html = response.get_data(as_text=True)
    script = '<script src="/static/opus-bridge.js" defer></script>'
    if script not in html:
        response.set_data(html.replace("</head>", script + "</head>", 1))
    return response


def info_with_opus():
    payload = local.info_payload()
    payload["cloud_models"] = [_cloud_info()]
    payload["cloud_gateway"] = {
        "provider": "Experiential Labs",
        "configured": experiential_configured(),
        "key_exposed_to_browser": False,
    }
    return jsonify(payload)


def _validated_cloud_history(data: dict, prompt_bytes: int):
    history = data.get("history", [])
    if not isinstance(history, list) or len(history) > 12 or len(history) % 2:
        return None, (jsonify(error="Maximal sechs vollständige Nachrichtenpaare erlaubt."), 400)
    for index, item in enumerate(history):
        expected = "user" if index % 2 == 0 else "assistant"
        if (
            not isinstance(item, dict)
            or item.get("role") != expected
            or not isinstance(item.get("content"), str)
        ):
            return None, (jsonify(error="Ungültiger Chatverlauf."), 400)
    try:
        history_bytes = sum(len(item["content"].encode("utf-8")) for item in history)
    except UnicodeEncodeError:
        return None, (jsonify(error="Ungültiger Unicode-Text im Verlauf."), 400)
    if history_bytes + prompt_bytes > local.MAX_PROMPT_BYTES:
        return None, (jsonify(error="Chatkontext zu lang. Ältere Nachrichten werden beim nächsten Versuch ausgelassen."), 413)
    return history, None


def generate_with_opus():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get("provider", "local") != "experiential":
        return _original_generate()

    if type(data.get("stream", False)) is not bool:
        return jsonify(error="Ungültige Stream-Option."), 400
    if data.get("mode", "chat") != "chat":
        return jsonify(error="Claude Opus 5 ist in Helios nur im Chat-Modus aktiv. SOLVE und Codeprüfung bleiben lokal."), 400

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="Bitte gib einen Text ein."), 400
    try:
        prompt_bytes = len(prompt.encode("utf-8"))
    except UnicodeEncodeError:
        return jsonify(error="Ungültiger Unicode-Text."), 400
    if prompt_bytes > local.MAX_PROMPT_BYTES:
        return jsonify(error="Prompt überschreitet 4096 UTF-8-Bytes."), 413

    requested_model = data.get("model", experiential_model_slug())
    if requested_model != experiential_model_slug():
        return jsonify(error="Unbekanntes Cloud-Modell."), 400
    if not experiential_configured():
        return jsonify(error="Claude Opus 5 ist noch nicht eingerichtet. Setze EXPLABS_API_KEY und starte Helios neu."), 503

    request_id = data.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]{16,64}", request_id)):
        return jsonify(error="Ungültige Anfrage-ID."), 400

    max_new = data.get("max_new_tokens", 256)
    if type(max_new) is not int or not 1 <= max_new <= local.MAX_NEW_TOKENS:
        return jsonify(error="Antwortlänge muss zwischen 1 und 512 liegen."), 400

    history, error = _validated_cloud_history(data, prompt_bytes)
    if error:
        return error
    for text in [item["content"] for item in history if item["role"] == "user"] + [prompt]:
        guard = local.check_prompt(text)
        if not guard.allowed:
            return jsonify(
                mode="guarded-response",
                text=guard.response,
                security_category=guard.category,
                model_generated=False,
            )

    # Local byte-token budgets are intentionally mapped to a useful cloud answer
    # budget. The gateway still caps this integration at 4096 output tokens.
    cloud_max_tokens = min(4096, max(1024, max_new * 4))
    messages = [
        {
            "role": "system",
            "content": (
                "You are Claude Opus 5 inside the Helios AI workspace. "
                "Be precise and production-oriented. For UI and code tasks, return implementable code and concrete design decisions. "
                "Do not claim to have edited local files unless Helios explicitly provided a file-editing tool."
            ),
        },
        *({"role": item["role"], "content": item["content"]} for item in history),
        {"role": "user", "content": prompt.strip()},
    ]

    started = time.perf_counter()
    try:
        result = experiential_chat(messages, max_tokens=cloud_max_tokens)
    except ExperientialError as exc:
        status = 503 if not experiential_configured() else 502
        return jsonify(error=str(exc)), status
    elapsed = time.perf_counter() - started

    text, redacted = local.sanitize_output(result["text"])
    usage = result.get("usage") or {}
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    try:
        output_tokens = int(output_tokens)
    except (TypeError, ValueError):
        output_tokens = 0
    try:
        input_tokens = int(input_tokens)
    except (TypeError, ValueError):
        input_tokens = 0

    return jsonify(
        mode="cloud-generation",
        model_generated=True,
        text=text,
        generated_tokens=output_tokens,
        input_tokens=input_tokens,
        seconds=round(elapsed, 4),
        tokens_per_second=round(output_tokens / max(elapsed, 1e-9), 2) if output_tokens else 0,
        output_secret_redacted=redacted,
        cancelled=False,
        active_checkpoint="claude-opus-5",
        provider="experiential",
        cloud_model=result.get("model") or experiential_model_slug(),
        finish_reason=result.get("finish_reason"),
        usage=usage,
        request_id=request_id,
    )


# Replace only the view functions. All existing Flask rules and guards remain in
# place, including localhost-only access, the session token and rate limiting.
app.view_functions["index"] = index_with_opus
app.view_functions["info"] = info_with_opus
app.view_functions["generate"] = generate_with_opus


if __name__ == "__main__":
    print(
        f"Neral / Helios bereit · Claude Opus 5: {'bereit' if experiential_configured() else 'API-Key fehlt'}",
        flush=True,
    )
    serve(app, host="127.0.0.1", port=local.APP_PORT, threads=4)
