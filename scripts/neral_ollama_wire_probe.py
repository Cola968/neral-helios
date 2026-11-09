"""Capture sanitized Ollama wire metadata in the manual THREE-case smoke only.

This module is never enabled in NERAL's normal chat. The existing localhost-
only opener and its redirect protection remain authoritative. The live HTTP
bytes are replayed unchanged to the production OllamaLocalBackend parser.
No prompts, response contents, thinking text or raw HTTP error bodies are
saved, uploaded, printed or placed in diagnostic objects.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.error

from neral_agent_autonomy import _safe_ollama_http_error_type


CHAT_URL = "http://127.0.0.1:11434/api/chat"
KNOWN_REQUEST_KEYS = frozenset({
    "model", "messages", "think", "stream", "options",
    "keep_alive", "format", "tools", "stop",
})
KNOWN_OPTION_KEYS = frozenset({
    "num_ctx", "num_predict", "temperature", "top_k", "top_p",
    "min_p", "presence_penalty", "repeat_penalty", "stop",
})
KNOWN_RESPONSE_ERROR_KEYS = frozenset({"error", "code", "status", "message"})


def _public_counter(value):
    return value if type(value) is int and 0 <= value <= 100_000_000 else None


def _public_done_reason(value):
    return value if value in ("stop", "length", "load", "unload") else (
        "other" if isinstance(value, str) else None)


def summarize_ollama_request(payload: object) -> dict:
    """Schema-only request diff: the model does not receive a modified prompt."""
    if not isinstance(payload, dict):
        return {"request_valid_json_object": False}
    messages = payload.get("messages")
    options = payload.get("options")
    parsed_messages = messages if isinstance(messages, list) else []
    parsed_options = options if isinstance(options, dict) else {}
    roles = [
        row.get("role") if isinstance(row, dict) and
            row.get("role") in ("system", "assistant", "user", "tool")
        else "invalid" for row in parsed_messages
    ]
    empty_count = sum(
        not isinstance(row, dict) or not isinstance(row.get("content"), str)
        or not row["content"].strip()
        for row in parsed_messages
    )
    valid_roles = (
        isinstance(messages, list) and len(messages) >= 1
        and all(isinstance(m, dict)
                and m.get("role") in ("system", "assistant", "user", "tool")
                and isinstance(m.get("content"), str)
                and m["content"].strip() for m in messages)
    )
    return {
        "request_valid_json_object": True,
        "request_keys": sorted(set(payload) & KNOWN_REQUEST_KEYS),
        "unknown_request_key_count": len(set(payload) - KNOWN_REQUEST_KEYS),
        "option_keys": sorted(set(parsed_options) & KNOWN_OPTION_KEYS),
        "unknown_option_key_count": len(set(parsed_options) - KNOWN_OPTION_KEYS),
        "model_is_qwen35_9b": payload.get("model") == "qwen3.5:9b",
        "message_count": len(parsed_messages),
        "message_roles": roles,
        "valid_message_roles": valid_roles,
        "empty_message_count": empty_count,
        "has_assistant_prefill": bool(roles and roles[-1] == "assistant"),
        "has_top_level_stop": "stop" in payload,
        "has_options_stop": "stop" in parsed_options,
        "has_format": "format" in payload,
        "has_tools": "tools" in payload,
        "think": payload.get("think") if type(payload.get("think")) is bool else None,
        "stream": payload.get("stream") if type(payload.get("stream")) is bool else None,
        "num_ctx": _public_counter(parsed_options.get("num_ctx")),
        "num_predict": _public_counter(parsed_options.get("num_predict")),
        "keep_alive_is_5m": payload.get("keep_alive") == "5m",
    }


def summarize_ollama_response(packet: object, *, status: int) -> dict:
    """Non-stream response only: counts identify actual decoding vs zero tokens."""
    msg = packet.get("message") if isinstance(packet, dict) else None
    msg = msg if isinstance(msg, dict) else {}
    content = msg.get("content")
    thinking = msg.get("thinking")
    calls = msg.get("tool_calls")
    return {
        "ollama_http_status": status if type(status) is int else None,
        "done": packet.get("done") if isinstance(packet, dict)
                and type(packet.get("done")) is bool else None,
        "done_reason": _public_done_reason(
            packet.get("done_reason") if isinstance(packet, dict) else None),
        "eval_count": _public_counter(
            packet.get("eval_count") if isinstance(packet, dict) else None),
        "prompt_eval_count": _public_counter(
            packet.get("prompt_eval_count") if isinstance(packet, dict) else None),
        "content_length": len(content) if isinstance(content, str) else None,
        "thinking_length": len(thinking) if isinstance(thinking, str) else None,
        "tool_calls_present": "tool_calls" in msg,
        "tool_calls_count": len(calls) if isinstance(calls, list)
                            and len(calls) <= 100 else None,
    }


def summarize_ollama_error(raw: bytes, *, status: int) -> dict:
    """Never print raw 500 errors: they may embed portions of a user's prompt."""
    if not isinstance(raw, bytes):
        raw = b""
    try:
        packet = json.loads(raw)
        if not isinstance(packet, dict):
            packet = {}
    except (ValueError, TypeError, UnicodeError):
        packet = {}
    err = packet.get("error")
    return {
        "ollama_http_status": status if type(status) is int else None,
        "ollama_error_type": _safe_ollama_http_error_type(io.BytesIO(raw)),
        "ollama_error_json_keys": sorted(set(packet) & KNOWN_RESPONSE_ERROR_KEYS),
        "ollama_error_length": len(err) if isinstance(err, str) else None,
        "ollama_error_body_bytes": len(raw),
        "ollama_error_sha256": hashlib.sha256(raw).hexdigest(),
    }


class _BufferedResponse:
    """Preserve read(), context-manager and status for the existing backend."""
    def __init__(self, raw: bytes, status: int):
        self._buffer = io.BytesIO(raw)
        self.status = status

    def read(self, *args):
        return self._buffer.read(*args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._buffer.close()
        return False


def make_capturing_opener(upstream_factory, events: list):
    """Return a temporary replacement opener factory, not an Ollama proxy.

    This is wired only inside the explicitly authorized three-case local smoke.
    All localhost-only network/redirect behavior is delegated to upstream.
    """
    if not callable(upstream_factory) or not isinstance(events, list):
        raise TypeError("Diagnostic opener requires a factory and event sink")

    class _CapturingOpener:
        def open(self, request, timeout):
            # Never capture unrelated traffic or model unload/load requests.
            if request.full_url != CHAT_URL or not isinstance(request.data, bytes):
                return upstream_factory().open(request, timeout=timeout)
            try:
                packet = json.loads(request.data)
            except (ValueError, TypeError, UnicodeError):
                packet = {}
            if not isinstance(packet, dict) or not packet.get("messages"):
                return upstream_factory().open(request, timeout=timeout)
            safe_request = summarize_ollama_request(packet)
            try:
                live = upstream_factory().open(request, timeout=timeout)
                with live:
                    raw = live.read(128 * 1024 + 1)
                    status = getattr(live, "status", 200)
            except urllib.error.HTTPError as exc:
                # Capture only a bounded local JSON error; replay identical
                # bytes to the original diagnostic classifier for compatibility.
                raw_error = exc.read(4097)
                if len(events) < 24:
                    events.append({
                        **safe_request,
                        **summarize_ollama_error(raw_error, status=exc.code),
                    })
                raise urllib.error.HTTPError(
                    exc.url, exc.code, exc.msg, exc.headers,
                    io.BytesIO(raw_error)) from None
            if len(events) < 24:
                try:
                    response_packet = json.loads(raw)
                except (ValueError, TypeError, UnicodeError):
                    response_packet = {}
                events.append({
                    **safe_request,
                    **summarize_ollama_response(response_packet, status=status),
                })
            return _BufferedResponse(raw, status)

    return _CapturingOpener
