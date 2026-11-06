"""A genuine local HF NERAL adapter or local Ollama backend in conversational mode.

No network provider fallback is implemented here. The backend is the SAME
instance used by the local model-driven agent. Model provenance is explicit;
a Qwen-based experimental adapter is not an independently trained model.
This module is a synchronous, bounded first chat interface, not streaming.
"""
from __future__ import annotations

import re
import threading
import time

from flask import jsonify, request

SYSTEM = ("Du bist NERAL. Du antwortest natürlich, präzise und sachlich auf "
          "Deutsch, soweit der Nutzer nichts anderes möchte. "
          "Du benutzt lokal ausgeführte Modellgewichte, keine Cloud-Inferenz. "
          "Gib Unsicherheit zu und erfinde keine Quellen, Resultate oder "
          "ausgeführten Werkzeuge. Stelle dich als NERAL mit einer "
          "Qwen-Foundation dar, sofern ein verifizierter Adapter aktiv ist. "
          "Ein Adapter ist kein Beleg für überlegene Modellqualität.")
BASE_MODEL = "Qwen2.5-Coder-7B-Instruct"
MAX_PROMPT_BYTES = 4096
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_BYTES = 12000


def _provenance(backend) -> dict:
    is_adapter = isinstance(getattr(backend, "adapter_sha256", None), str)
    verified_sha = getattr(backend, "adapter_sha256", None)
    is_adapter = is_adapter and re.fullmatch(r"[0-9a-f]{64}", verified_sha) is not None
    loaded = bool(is_adapter and getattr(backend, "_model", None) is not None)
    return {
        "model": getattr(backend, "model", "local_model"),
        "foundation": BASE_MODEL if is_adapter else "local_model_see_name",
        "adapter_loaded": loaded,
        "adapter_sha256": verified_sha if loaded else None,
        "model_origin": ("local_neral_lora_experimental" if loaded
                         else "local_adapter_configured_not_loaded" if is_adapter
                         else "local_foundation_not_neral_adapter"),
        "quality_improvement_proven": False,
        "no_remote_qwen_service_configured": True,
        "response_mode": "local_non_streaming",
    }


def _messages(payload: object) -> list[dict]:
    if not isinstance(payload, dict) or not {"prompt"} <= set(payload) or not set(payload) <= {"prompt", "history"}:
        raise ValueError("Nur prompt und history sind zulässig.")
    prompt = payload.get("prompt")
    history = payload.get("history", [])
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("Eingabe muss 1 bis 4096 UTF-8-Bytes enthalten.")
    if not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES:
        raise ValueError("Verlauf zu lang.")
    normalized = []
    for i, turn in enumerate(history):
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise ValueError("Ungültiger Gesprächsverlauf.")
        role, content = turn["role"], turn["content"]
        expected_role = "user" if i % 2 == 0 else "assistant"
        if role != expected_role or not isinstance(content, str) or not content.strip():
            raise ValueError("Verlauf enthält unerlaubte Rollen oder leere Texte.")
        if len(content.encode("utf-8")) > 2048:
            raise ValueError("Verlaufseintrag zu groß.")
        normalized.append({"role": role, "content": content})
    if len(normalized) % 2:
        raise ValueError("Verlauf muss mit einer vollständigen Assistenzantwort enden.")
    if len(str(normalized).encode("utf-8")) > MAX_HISTORY_BYTES:
        raise ValueError("Verlauf überschreitet das Kontextbudget.")
    return [{"role": "system", "content": SYSTEM}] + normalized + [{"role": "user", "content": prompt.strip()}]


def register_neral_local_chat_api(app, backend):
    """Mount onto same Flask app; the main app's session/Origin guard applies."""
    busy = threading.Lock()

    @app.get("/api/neral/chat/status")
    def local_chat_status():
        return jsonify(_provenance(backend))

    @app.post("/api/neral/chat/generate")
    def local_chat_generate():
        if not request.is_json:
            return jsonify(error="Nur JSON-Anfragen erlaubt."), 415
        try:
            data = _messages(request.get_json(silent=True))
        except (TypeError, ValueError, UnicodeError):
            return jsonify(error="Ungültiger oder zu langer Chatinhalt."), 400
        if not busy.acquire(blocking=False):
            return jsonify(error="NERAL berechnet bereits eine Antwort."), 409
        try:
            started = time.monotonic()
            try:
                # Ollama agent requests enforce JSON; normal conversations do
                # not. HF adapter backends use their existing local respond().
                reply = getattr(backend, "respond_chat", None)
                text = reply(data) if callable(reply) else backend.respond(data)
            except Exception:
                # Do not leak local paths, model credentials or private prompts.
                return jsonify(error="Lokale Modell-Inferenz nicht verfügbar oder fehlgeschlagen."), 503
            if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32768:
                return jsonify(error="Lokales Modell lieferte eine ungültige Antwort."), 503
            return jsonify({
                "text": text,
                "model_generated": True,
                "seconds": round(time.monotonic() - started, 3),
                **_provenance(backend),
            })
        finally:
            busy.release()
    return busy
