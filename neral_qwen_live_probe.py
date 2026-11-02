"""One-shot, opt-in real Qwen -> NERAL Flask diagnostic (stdlib + existing app deps).

Only installed localhost Ollama is contacted. This command cannot download
models, train weights, call an external API, run shell commands, or publish
private chat/model output. Three synthetic prompts, each with 128 tokens max.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

MODEL = "qwen3.5:9b"
HOST = "http://127.0.0.1:11434"
CASES = (
    {
        "id": "fact_de",
        "prompt": "Antworte auf Deutsch mit einem Satz: Was ist die Hauptstadt von Frankreich?",
        "needle": "paris",
        "max_new_tokens": 96,
        "quality": "standard",
    },
    {
        "id": "math_de",
        "prompt": "Berechne 17 mal 23. Nenne das genaue Ergebnis.",
        "needle": "391",
        "max_new_tokens": 96,
        "quality": "standard",
    },
    {
        "id": "code_de",
        "prompt": "Pruefe folgenden Python-Code auf Sicherheitsprobleme und erklaere kurz: def unsafe(x): return eval(x)",
        "needle": "eval",
        "max_new_tokens": 320,
        "required_tool": "code_check",
        "quality": "standard",
    },
)


def classify_assistant_error(message: object) -> str:
    """An allowlisted diagnostic label, never the raw exception or input."""
    known = {
        "Ungültige Codeprüfparameter.": "invalid_code_check_arguments",
        "Codeprüfung überschreitet das sichere Ergebnislimit.": "code_check_result_limit",
        "Werkzeugaufrufe überschreiten das Limit.": "tool_budget",
        "Unzulässige Anzahl Werkzeugaufrufe.": "too_many_calls_in_one_message",
        "Ungültiger Werkzeugaufruf.": "malformed_tool_call",
        "Ungültiger Python-Prüfaufruf.": "malformed_code_tool",
        "Nicht freigegebenes Codewerkzeug.": "wrong_code_tool",
        "Nicht freigegebenes Werkzeug.": "unknown_tool",
        "Das Modell hat ein nicht freigegebenes Werkzeug angefordert.": "wrong_calculator_tool",
        "Python-Code überschreitet das 4096-Byte-Limit.": "code_input_limit",
        "Python-Code kann nicht statisch geprüft werden.": "static_checker_error",
        "Der Codeprüfer erwartet nichtleeren Python-Quelltext.": "empty_code",
        "Das lokale Modell hat keine verwendbare Textantwort erzeugt.": "missing_text",
        "Nach Sicherheitsprüfung blieb keine Antwort übrig.": "sanitized_to_empty",
        "Das Modell hat keine endgültige Antwort geliefert.": "no_final_answer",
        "Unvollständiger Werkzeugaufruf am Antwortlimit.": "incomplete_tool_call",
        "Unerwartete oder unvollständige Modellantwort.": "model_identity_or_done",
        "Unbekannter Abschlussgrund aus Ollama.": "unknown_finish_reason",
        "Ungültiger Nachrichten-Typ aus Ollama.": "wrong_message_role",
        "Ungültiger Textinhalt im Werkzeugaufruf.": "malformed_tool_content",
    }
    return known.get(message, "other_assistant_error") if isinstance(message, str) else "other_assistant_error"


def sanitized_protocol_event(raw: object) -> dict:
    """Inspect transport *shapes* only, never copy actual code, content or names."""
    data = raw if isinstance(raw, dict) else {}
    msg = data.get("message")
    msg = msg if isinstance(msg, dict) else {}
    calls = msg.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    names, arg_types = [], []
    for call in calls[:3]:
        fn = call.get("function") if isinstance(call, dict) else None
        fn = fn if isinstance(fn, dict) else {}
        name = fn.get("name")
        names.append(name if type(name) is str and name in ("calculator", "code_check") else "other")
        arg_types.append(type(fn.get("arguments")).__name__)
    reason = data.get("done_reason")
    return {
        "done_reason": reason if reason in ("stop", "length") else "unknown_or_missing",
        "tool_count": len(calls),
        "tool_names": names,
        "argument_types": arg_types,
        "assistant_content_type": type(msg.get("content")).__name__,
    }


def verify_installed_model(payload: object) -> bool:
    """Require the exact preinstalled tag; no fallback, download or alias."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("malformed localhost Ollama model listing")
    models = payload["models"]
    if any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in models):
        raise ValueError("malformed model listing entry")
    return any(item["name"] == MODEL for item in models)


def assess(http_status: int, payload: object, case: dict, elapsed: float) -> dict:
    """Expose metadata only. Do not publish raw user/model text or local paths."""
    data = payload if isinstance(payload, dict) else {}
    raw = data.get("text")
    tools = data.get("tools_used")
    valid_tools = (isinstance(tools, list)
                   and len(tools) <= 2
                   and len(set(str(x) for x in tools)) == len(tools)
                   and all(type(x) is str and x in {"calculator", "code_check"} for x in tools))
    valid = (
        http_status == 200
        and data.get("mode") == "neral-assistant"
        and data.get("source") == "external-model"
        and data.get("base_model") == MODEL
        and data.get("model_generated") is True
        and data.get("incomplete") is False
        and data.get("finish_reason") in (None, "stop")
        and isinstance(raw, str)
        and bool(raw.strip())
        and valid_tools
        and (not case.get("required_tool") or case["required_tool"] in tools)
    )
    clean = raw if isinstance(raw, str) else ""
    count = data.get("generated_tokens")
    speed = data.get("tokens_per_second")
    return {
        "case": case["id"],
        "http_status": http_status,
        "valid_provider": bool(valid),
        "answer_match": bool(valid and case["needle"].casefold() in clean.casefold()),
        "incomplete": data.get("incomplete") is True,
        "tool_names": tools if valid_tools else [],
        "text_bytes": len(clean.encode("utf8", errors="replace")),
        "response_sha256": hashlib.sha256(clean.encode("utf8", errors="replace")).hexdigest() if clean else None,
        "latency_seconds": round(elapsed, 2),
        "generated_tokens": count if type(count) is int and 0 <= count <= 1_000_000 else None,
        "tokens_per_second": round(float(speed), 2) if isinstance(speed, (int, float)) and not isinstance(speed, bool) and 0 <= speed < 100_000 else None,
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _localhost_json(path: str) -> dict:
    if path not in {"/api/tags", "/api/ps"}:
        raise ValueError("Only fixed read-only local Ollama endpoints permitted")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(HOST + path, timeout=5) as response:
        data = response.read(65537)
    if len(data) > 65536:
        raise ValueError("Oversize local Ollama response")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Malformed local Ollama response")
    return result


def run_live() -> int:
    # Do not start Ollama, pull a model, or use a cloud fallback.
    try:
        installed = verify_installed_model(_localhost_json("/api/tags"))
    except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        print(json.dumps({"status": "blocked", "reason": "local_ollama_unavailable",
                          "model": MODEL, "download_attempted": False}))
        return 2
    if not installed:
        print(json.dumps({"status": "blocked", "reason": "exact_model_not_installed",
                          "model": MODEL, "download_attempted": False}))
        return 2

    # A temporary local app state only: never read or mutate user's real chats,
    # projects, Arctus corpora, checkpoint files, or personal state.
    with tempfile.TemporaryDirectory(prefix="neral-qwen-readonly-", ignore_cleanup_errors=True) as tmp:
        os.environ["HELIOS_STATE_ROOT"] = tmp
        os.environ["HELIOS_ASSETS_ROOT"] = tmp
        os.environ["HELIOS_PREPARATION_MODE"] = "1"
        os.environ["HELIOS_DEVICE"] = "cpu"
        os.environ["NERAL_ASSISTANT_ENABLED"] = "0"
        os.environ["NERAL_ASSISTANT_MODEL"] = MODEL
        import app as web
        import neral_assistant as provider
        # Capture only protocol shape and a known exception category, never
        # tool arguments, user prompts, full model output or private paths.
        protocol_events = []
        provider_original = provider.local_ollama_request

        def tracked_transport(payload):
            response = provider_original(payload)
            protocol_events.append(sanitized_protocol_event(response))
            return response

        provider.local_ollama_request = tracked_transport
        errors = []
        original_assistant = web.assistant_turn

        def traced_assistant(*args, **kwargs):
            try:
                return original_assistant(*args, **kwargs)
            except web.AssistantError as exc:
                errors.append(classify_assistant_error(str(exc)))
                raise

        web.assistant_turn = traced_assistant
        # Restrict the opt-in to the current Flask test client. Runtime loading
        # was disabled during import; no native model weights have been opened.
        web.PREPARATION_MODE = False
        web.ASSISTANT_ENABLED = True
        web.ASSISTANT_MODEL = MODEL
        web.ASSISTANT_THINK = False
        web.checkpoint_loaded = False
        web.arctus = None
        web._hits.clear()
        client = web.app.test_client()
        client.get("/", base_url="http://localhost")
        summaries = []
        for case in CASES:
            event_start, error_start = len(protocol_events), len(errors)
            started = time.monotonic()
            response = client.post(
                "/api/generate",
                json={"mode": "assistant", "prompt": case["prompt"],
                      "history": [], "stream": False, "quality": case["quality"],
                      "max_new_tokens": case["max_new_tokens"], "temperature": 0.0},
                base_url="http://localhost",
            )
            payload = response.get_json(silent=True)
            summary = assess(response.status_code, payload, case, time.monotonic() - started)
            summary["provider_protocol"] = protocol_events[event_start:]
            summary["assistant_error_code"] = errors[error_start] if len(errors) > error_start else None
            summaries.append(summary)
    vram = None
    try:
        ps = _localhost_json("/api/ps")
        for row in ps.get("models", []):
            if isinstance(row, dict) and row.get("name") == MODEL and type(row.get("size_vram")) is int:
                vram = row["size_vram"]
                break
    except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        pass

    # This report intentionally contains neither model answer text nor secrets.
    valid = sum(1 for x in summaries if x["valid_provider"])
    matched = sum(1 for x in summaries if x["answer_match"])
    print(json.dumps({
        "status": "completed" if valid == len(CASES) else "product_regression",
        "model": MODEL, "layer": "real-local-ollama+flask",
        "valid_responses": valid, "answer_matches": matched, "case_count": len(CASES),
        "vram_bytes_claimed_by_ollama": vram,
        "model_download_attempted": False, "model_weight_change_attempted": False,
        "cloud_api_used": False, "private_data_uploaded": False,
        "cases": summaries,
    }, ensure_ascii=True, sort_keys=True))
    return 0 if valid == len(CASES) and matched == len(CASES) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Pure offline inventory; default")
    mode.add_argument("--execute-local-qwen", action="store_true", help="Explicit one-time local inference opt-in")
    args = parser.parse_args(argv)
    if not args.execute_local_qwen:
        print(json.dumps({
            "status": "PLAN_ONLY_NO_INFERENCE", "model": MODEL,
            "case_count": len(CASES), "model_download_attempted": False,
            "private_data_uploaded": False,
        }))
        return 0
    return run_live()


if __name__ == "__main__":
    sys.exit(main())
