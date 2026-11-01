"""NERAL Assistant: explicitly opted-in local Ollama inference.

NERAL is the product; model weights are Qwen3.5 (Apache-2.0), not Arctus.
All network traffic is loopback-only. No model download, fine-tuning,
cloud endpoint, Python eval/exec or arbitrary function calling.
"""
from __future__ import annotations

import json
import math
import re
import socket
import urllib.error
import urllib.request

from solve import arithmetic
from code_check import check_code, MAX_CODE_BYTES

ALLOWED_MODELS = frozenset({"qwen3.5:9b", "qwen3.5:4b"})
DEFAULT_MODEL = "qwen3.5:9b"
LOOPBACK_ENDPOINT = "http://127.0.0.1:11434/api/chat"
MAX_INPUT_BYTES = 4096
DEEP_INPUT_BYTES = 8192
DEEP_MAX_NEW_TOKENS = 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_CODE_CHECK_RESULT_BYTES = 4096
from neral_assistant_persona import PERSONA_VERSION, SYSTEM_PROMPT

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate a bounded numeric arithmetic expression accurately.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Only numeric arithmetic, such as (12+8)*4."}
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    }
}


CODE_CHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_check",
        "description": (
            "Statically inspect Python source for syntax and selected security or "
            "reliability findings. Never runs, imports or tests submitted code; "
            "this is not a proof of functional correctness."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code, at most 4096 UTF-8 bytes"}
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


class AssistantError(ValueError):
    """Request or model protocol does not satisfy the local security contract."""


class AssistantUnavailable(AssistantError):
    """Model server or configured local model is unavailable."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def local_ollama_request(payload: dict) -> dict:
    """Fixed URL, no proxies, no redirects and bounded response (stdlib only)."""
    request = urllib.request.Request(
        LOOPBACK_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=125) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as exc:
        raise AssistantUnavailable(
            "Lokales Ollama-Modell nicht erreichbar. Starte Ollama und installiere qwen3.5:9b."
        ) from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise AssistantError("Lokale Modellantwort überschreitet das Sicherheitslimit.")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssistantError("Ungültiges JSON vom lokalen Modellserver.") from exc
    if not isinstance(data, dict):
        raise AssistantError("Ungültige Antwort vom lokalen Modellserver.")
    return data


def _validate(prompt: str, history: list[dict], model: str, max_new: int,
              temperature: float, think: bool, quality: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise AssistantError("Bitte eine Nachricht eingeben.")
    if not isinstance(model, str) or model not in ALLOWED_MODELS:
        raise AssistantError("Unzulässige oder nicht zugelassene Modellbasis.")
    if type(quality) is not str or quality not in {"standard", "deep"}:
        raise AssistantError("Unbekanntes Assistant-Qualitätsprofil.")
    budget_limit = DEEP_MAX_NEW_TOKENS if quality == "deep" else 512
    if type(max_new) is not int or not 1 <= max_new <= budget_limit:
        raise AssistantError("Ungültiges Assistant-Antwortbudget.")
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or not 0 <= temperature <= 1.5:
        raise AssistantError("Ungültige Temperatur.")
    if type(think) is not bool:
        raise AssistantError("Ungültige Reasoning-Konfiguration.")
    if not isinstance(history, list) or len(history) > 12 or len(history) % 2:
        raise AssistantError("Ungültige Anzahl vorheriger Chatnachrichten.")
    size = len(prompt.encode("utf-8"))
    for idx, row in enumerate(history):
        if not isinstance(row, dict) or set(row) != {"role", "content"}:
            raise AssistantError("Ungültiger Chatverlauf.")
        if row["role"] != ("user" if idx % 2 == 0 else "assistant"):
            raise AssistantError("Ungültige Chat-Reihenfolge.")
        if not isinstance(row["content"], str) or not row["content"].strip():
            raise AssistantError("Leere oder ungültige Nachricht im Verlauf.")
        size += len(row["content"].encode("utf-8"))
    input_limit = DEEP_INPUT_BYTES if quality == "deep" else MAX_INPUT_BYTES
    if size > input_limit:
        raise AssistantError("Chatverlauf überschreitet das konfigurierte UTF-8-Bytelimit.")


def _message(data: dict, model: str) -> dict:
    if not isinstance(data, dict) or data.get("model") != model or data.get("done") is not True:
        raise AssistantError("Unerwartete oder unvollständige Modellantwort.")
    # done=True ends the HTTP generation, including when num_predict was exhausted.
    # The API may omit done_reason; retain that legacy contract, validate explicit values.
    if "done_reason" in data and (not isinstance(data["done_reason"], str)
                                  or data["done_reason"] not in {"stop", "length"}):
        raise AssistantError("Unbekannter Abschlussgrund aus Ollama.")
    message = data.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise AssistantError("Ungültiger Nachrichten-Typ aus Ollama.")
    return message


def _valid_tool_envelope(call: object) -> bool:
    """Accept only inert Ollama tool-call metadata, never extra tool actions.

    Real Ollama responses may attach an opaque call ID, call type or zero-based
    index alongside the function. Validate them with small hard bounds while
    keeping the executable function and its parameters strictly allowlisted.
    """
    if not isinstance(call, dict) or "function" not in call:
        return False
    if set(call) - {"function", "id", "type", "index"}:
        return False
    if not isinstance(call["function"], dict):
        return False
    if "type" in call and call["type"] != "function":
        return False
    if "index" in call and (type(call["index"]) is not int or call["index"] != 0):
        return False
    if "id" in call and (
        type(call["id"]) is not str
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", call["id"]) is None
    ):
        return False
    return True


def _calculate(call: dict) -> str:
    if not _valid_tool_envelope(call):
        raise AssistantError("Ungültiger Werkzeugaufruf.")
    fn = call["function"]
    if not isinstance(fn, dict) or fn.get("name") != "calculator":
        raise AssistantError("Das Modell hat ein nicht freigegebenes Werkzeug angefordert.")
    args = fn.get("arguments")
    if not isinstance(args, dict) or set(args) != {"expression"}:
        raise AssistantError("Ungültige Rechnerparameter.")
    expression = args["expression"]
    if not isinstance(expression, str) or not re.fullmatch(r"[0-9\s.+*/()%^\-]{1,120}", expression):
        raise AssistantError("Der Rechner akzeptiert ausschließlich numerische Ausdrücke.")
    try:
        result = arithmetic(expression)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise AssistantError("Unsicherer oder nicht auswertbarer Rechenausdruck.") from exc
    return json.dumps({"result": result["result"], "engine": "NERAL SOLVE", "model_generated": False})


def _code_check(call: dict) -> str:
    """Use existing bounded AST checker, never run untrusted Python source."""
    if not _valid_tool_envelope(call):
        raise AssistantError("Ungültiger Python-Prüfaufruf.")
    fn = call["function"]
    if not isinstance(fn, dict) or fn.get("name") != "code_check":
        raise AssistantError("Nicht freigegebenes Codewerkzeug.")
    args = fn.get("arguments")
    if not isinstance(args, dict) or set(args) != {"code"}:
        raise AssistantError("Ungültige Codeprüfparameter.")
    code = args["code"]
    if not isinstance(code, str) or not code.strip():
        raise AssistantError("Der Codeprüfer erwartet nichtleeren Python-Quelltext.")
    try:
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise AssistantError("Python-Code überschreitet das 4096-Byte-Limit.")
        result = check_code(code)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AssistantError("Python-Code kann nicht statisch geprüft werden.") from exc
    if result.get("executed") is not False:
        raise AssistantError("Nur rein statische Codeprüfung ist erlaubt.")
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized.encode("utf-8")) <= MAX_CODE_CHECK_RESULT_BYTES:
        return serialized

    # The AST checker may produce hundreds of repetitive findings for 4 KiB of
    # source. Preserve exact totals, prioritize high severity, and tell Qwen
    # explicitly that the transmitted findings are incomplete. Never silently
    # turn a partial list into a claim of comprehensive safety.
    findings = result["findings"]
    report = dict(result)
    report["findings"] = []
    report["truncated"] = True
    report["omitted_findings"] = len(findings)
    report["note"] = (
        "Gekürzte statische Analyse: priorisierte Beispiele statt aller Befunde; "
        "finding_count und severity_counts gelten weiterhin für ALLE Befunde. "
        "Kein ausgeführter Code, keine vollständige Sicherheitsprüfung."
    )
    severity = {"high": 0, "medium": 1, "low": 2}
    for finding in sorted(findings, key=lambda item: severity.get(item.get("severity"), 3)):
        report["findings"].append(finding)
        report["omitted_findings"] = len(findings) - len(report["findings"])
        candidate = json.dumps(report, ensure_ascii=False)
        if len(candidate.encode("utf-8")) > MAX_CODE_CHECK_RESULT_BYTES:
            report["findings"].pop()
            report["omitted_findings"] += 1
            break
    serialized = json.dumps(report, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > MAX_CODE_CHECK_RESULT_BYTES:
        raise AssistantError("Codeprüfung überschreitet das sichere Ergebnislimit.")
    return serialized


def assistant_turn(prompt: str, history: list[dict] | None = None, *,
                   transport=None, model: str = DEFAULT_MODEL, think: bool = False,
                   max_new: int = 256, temperature: float = 0.0,
                   enable_tools: bool = True, quality: str = "standard") -> dict:
    """One NERAL turn: at most two distinct allowlisted tools and one final answer."""
    history = [] if history is None else history
    _validate(prompt, history, model, max_new, temperature, think, quality)
    if type(enable_tools) is not bool:
        raise AssistantError('Ungültige Tool-Konfiguration.')
    transport = local_ollama_request if transport is None else transport
    actual_think = think or quality == "deep"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": row["role"], "content": row["content"].strip()} for row in history)
    messages.append({"role": "user", "content": prompt.strip()})
    tools_used: list[str] = []
    data = None
    for iteration in range(3):
        payload = {
            "model": model,
            "messages": messages,
            "tools": [CALCULATOR_SCHEMA, CODE_CHECK_SCHEMA] if enable_tools else [],
            "stream": False,
            "think": actual_think,
            "keep_alive": "5m",
            "options": {"num_ctx": 8192 if quality == "deep" else 4096,
                        "num_predict": max_new, "temperature": float(temperature)},
        }
        data = transport(payload)
        response = _message(data, model)
        calls = response.get("tool_calls", None)
        if calls is None:
            calls = []
        if not isinstance(calls, list) or len(calls) > 1:
            raise AssistantError("Unzulässige Anzahl Werkzeugaufrufe.")
        if calls:
            if data.get("done_reason") == "length":
                raise AssistantError("Unvollständiger Werkzeugaufruf am Antwortlimit.")
            if not enable_tools:
                raise AssistantError('Nicht freigegebener Werkzeugaufruf im Modell-Benchmark.')
            call = calls[0]
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise AssistantError("Ungültiger Werkzeugaufruf.")
            tool_name = call["function"].get("name")
            if len(tools_used) >= 2 or tool_name in tools_used:
                raise AssistantError("Werkzeugaufrufe überschreiten das Limit.")
            if tool_name == "calculator":
                tool_output = _calculate(call)
            elif tool_name == "code_check":
                tool_output = _code_check(call)
            else:
                raise AssistantError("Nicht freigegebenes Werkzeug.")
            tool_content = response.get("content")
            if tool_content is not None and not isinstance(tool_content, str):
                raise AssistantError("Ungültiger Textinhalt im Werkzeugaufruf.")
            messages.append({"role": "assistant", "content": tool_content or "",
                             "tool_calls": calls})
            messages.append({"role": "tool", "tool_name": tool_name, "content": tool_output})
            tools_used.append(tool_name)
            continue
        text = response.get("content")
        if not isinstance(text, str) or not text.strip() or len(text) > 24000:
            raise AssistantError("Das lokale Modell hat keine verwendbare Textantwort erzeugt.")
        # NERAL's native security redaction applies to both external-weight and own models.
        from security_guard import sanitize_output
        safe, redacted = sanitize_output(text.strip())
        if not safe.strip():
            raise AssistantError("Nach Sicherheitsprüfung blieb keine Antwort übrig.")
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        count = eval_count if type(eval_count) is int and 0 <= eval_count <= 1_000_000 else 0
        speed = round(count * 1e9 / eval_duration, 2) if type(eval_duration) is int and eval_duration > 0 and count else 0
        return {
            "mode": "neral-assistant",
            "source": "external-model",
            "model_family": "qwen3.5",
            "token_unit": "Ollama-Token",
            "persona_version": PERSONA_VERSION,
            "model_generated": True,
            "name": "NERAL Assistant",
            "base_model": model,
            "model_origin": "Qwen3.5 · Apache-2.0 · separate open-weight foundation",
            "text": safe,
            "finish_reason": data.get("done_reason"),
            "incomplete": data.get("done_reason") == "length",
            "output_secret_redacted": redacted,
            "tools_used": tools_used,
            "generated_tokens": count,
            "tokens_per_second": speed,
            "seconds": round(eval_duration / 1e9, 3) if type(eval_duration) is int and eval_duration >= 0 else 0,
            "active_checkpoint": "Qwen3.5 (Ollama local); not Arctus",
            "reasoning_enabled": actual_think,
            "quality_mode": quality,
        }
    raise AssistantError("Das Modell hat keine endgültige Antwort geliefert.")
