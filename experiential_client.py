"""Small server-side client for Experiential Labs.

The browser never receives the API key. The only required configuration is
EXPLABS_API_KEY (or the backwards-compatible EXPERIENTIAL_API_KEY alias).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
DEFAULT_MODEL = "claude-opus-5"
MAX_CLOUD_OUTPUT_TOKENS = 4096


class ExperientialError(RuntimeError):
    """A safe, user-displayable gateway error."""


def api_key() -> str:
    return (os.environ.get("EXPLABS_API_KEY") or os.environ.get("EXPERIENTIAL_API_KEY") or "").strip()


def configured() -> bool:
    return bool(api_key())


def model_slug() -> str:
    return (os.environ.get("HELIOS_CLOUD_MODEL") or DEFAULT_MODEL).strip()


def _base_url() -> str:
    return (os.environ.get("EXPLABS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def chat(messages: list[dict[str, str]], *, max_tokens: int = 2048) -> dict:
    """Call the OpenAI-compatible Chat Completions endpoint.

    Claude 5 routes can pin sampling parameters, so this intentionally does not
    send temperature/top_p. Keep the request surface minimal and portable.
    """
    key = api_key()
    if not key:
        raise ExperientialError("Claude Opus 5 ist noch nicht eingerichtet. Setze EXPLABS_API_KEY und starte Helios neu.")
    if not isinstance(messages, list) or not messages:
        raise ExperientialError("Keine Nachrichten für Claude Opus 5 vorhanden.")

    limit = max(1, min(int(max_tokens), MAX_CLOUD_OUTPUT_TOKENS))
    payload = {
        "model": model_slug(),
        "messages": messages,
        "max_tokens": limit,
    }
    request = urllib.request.Request(
        _base_url() + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Neral-Helios/2.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(64 * 1024).decode("utf-8", "replace")
            parsed = json.loads(body)
            detail = parsed.get("error")
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("code")
            if not isinstance(detail, str):
                detail = None
        except Exception:
            detail = None
        if exc.code in {401, 403}:
            raise ExperientialError("Experiential API-Key wurde abgelehnt. Prüfe EXPLABS_API_KEY.") from exc
        if exc.code == 429:
            raise ExperientialError("Experiential hat das Anfrage-Limit erreicht. Bitte kurz warten.") from exc
        raise ExperientialError(detail or f"Experiential-Anfrage fehlgeschlagen (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ExperientialError("Experiential ist gerade nicht erreichbar. Prüfe Internetverbindung und Gateway.") from exc

    if len(raw) > 4 * 1024 * 1024:
        raise ExperientialError("Experiential-Antwort war unerwartet groß.")
    try:
        data = json.loads(raw.decode("utf-8"))
        choice = data["choices"][0]
        message = choice["message"]
        text = message["content"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ExperientialError("Experiential hat keine gültige Chat-Antwort geliefert.") from exc

    if not isinstance(text, str):
        raise ExperientialError("Claude Opus 5 hat keinen Text zurückgegeben.")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "text": text,
        "model": data.get("model") or model_slug(),
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
    }
