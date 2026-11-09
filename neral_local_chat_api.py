"""A genuine local HF NERAL adapter or local Ollama backend in conversational mode.

No network provider fallback is implemented here. The backend is the SAME
instance used by the local model-driven agent. Model provenance is explicit;
a Qwen-based experimental adapter is not an independently trained model.
This module is a synchronous, bounded first chat interface, not streaming.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from flask import Response, jsonify, request, stream_with_context

from neral_identity import build_identity_prompt
from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend

_SAFE_OLLAMA_ERROR_TEXT = {
    "token_limit": "Lokale Antwort am Ausgabelimit abgeschnitten. Bitte kürzer fragen oder die Aufgabe aufteilen.",
    "incomplete": "Lokale Modellantwort ohne bestätigten Abschluss. Bitte erneut versuchen.",
    "empty_answer": "Das lokale Modell hat keinen fertigen Antworttext geliefert.",
    "timeout": "Das lokale Modell hat die Wartezeit überschritten. Bitte erneut versuchen.",
    "invalid_response": "Das lokale Modell hat ein ungültiges Antwortformat geliefert.",
}


SYSTEM = build_identity_prompt("web", "local_foundation_not_neral_adapter")
BASE_MODEL = "Qwen2.5-Coder-7B-Instruct"
# Exactly the same neutral conditioning is sent to both weight variants.
# NERAL identity prompts, web search, SQL tools and exact-math tools stay OFF
# in this model-only A/B lab, so they cannot inflate the adapter result.
COMPARE_SYSTEM = (
    "Du bist ein hilfreicher lokaler Assistent. Antworte präzise, "
    "erfinde keine Fakten, Werkzeuge oder Tests. "
    "Befolge die vom Nutzer gewünschte Antwortform."
)
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
        "inference_profile": getattr(backend, "inference_profile", "legacy"),
        "foundation": BASE_MODEL if is_adapter else "local_model_see_name",
        "adapter_loaded": loaded,
        "adapter_sha256": verified_sha if loaded else None,
        "model_origin": ("local_neral_lora_experimental" if loaded
                         else "local_adapter_configured_not_loaded" if is_adapter
                         else "local_foundation_not_neral_adapter"),
        "quality_improvement_proven": False,
        "compare_available": bool(is_adapter and callable(
            getattr(backend, "compare_with_base", None))),
        "no_remote_qwen_service_configured": True,
        "response_mode": "local_non_streaming",
    }


def _runtime_identity_reply(prompt: str, backend) -> dict | None:
    """Answer narrowly recognized identity questions using runtime metadata.

    A generated reply must never substitute an older conversational answer
    for the backend's identity. This reports the configured model, not a
    successful weight-level inference or an improvement claim.
    """
    if not isinstance(prompt, str) or len(prompt) > 180:
        return None
    clean = " ".join(re.sub(r"[!?.,:;]", " ", prompt.casefold()).split())
    if clean not in {
        "wer bist du", "wie heißt du", "wie heisst du",
        "welches modell bist du", "welches modell verwendest du",
        "wie heißt du welches modell bist du",
        "wie heisst du welches modell bist du",
        "nein wie heißt du welches modell bist du",
        "nein wie heisst du welches modell bist du",
        "who are you", "what model are you",
    }:
        return None
    origin = _provenance(backend)
    name = str(origin["model"])
    if origin["adapter_loaded"]:
        detail = ("Ein experimenteller NERAL-LoRA-Adapter ist geladen; "
                  "eine Qualitätsüberlegenheit ist nicht nachgewiesen.")
    elif origin["model_origin"] == "local_adapter_configured_not_loaded":
        detail = ("Ein NERAL-LoRA-Adapter ist konfiguriert, "
                  "aber derzeit nicht geladen.")
    else:
        detail = "Kein verifizierter NERAL-LoRA-Adapter ist derzeit geladen."
    return {
        "text": f"Ich bin NERAL AI. Für diesen Chat ist das lokale Modell {name} konfiguriert. {detail}",
        "model_generated": False,
        "verified": True,
        "verification_scope": "runtime_metadata",
        "verification_status": "runtime_metadata",
        "tool_used": "runtime_identity",
    }


def _messages(payload: object, *, max_prompt_bytes: int = MAX_PROMPT_BYTES) -> list[dict]:
    if not isinstance(payload, dict) or not {"prompt"} <= set(payload) or not set(payload) <= {"prompt", "history"}:
        raise ValueError("Nur prompt und history sind zulässig.")
    prompt = payload.get("prompt")
    history = payload.get("history", [])
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > max_prompt_bytes:
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



def _verified_math_request(prompt: str) -> dict | None:
    """Exact local math only for explicitly requested standalone calculations.

    Fail closed to the ordinary model for prose, unsafe syntax, unsupported
    expressions, or tasks that ask for explanation rather than a bare answer.
    No eval/exec, shell, network or model invocation occurs here.
    """
    from neral_agent_math import MathInputError, eval_python_expr, solve_math_symbolic

    if not isinstance(prompt, str) or len(prompt) > 200 or "\n" in prompt:
        return None
    clean = prompt.strip()
    if clean.endswith("?"):
        clean = clean[:-1].rstrip()
    chars = r"[-+*/().0-9x=^−×÷ \t]{1,160}"
    equation = re.fullmatch(
        r"(?:bitte\s+)?(?:löse|solve)(?:\s+bitte)?(?:\s+die\s+gleichung)?"
        r"\s*:?\s*(?P<expr>" + chars + r")", clean, re.IGNORECASE)
    arithmetic = None
    if equation is None:
        arithmetic = re.fullmatch(
            r"(?:bitte\s+)?(?:berechne|calculate)(?:\s+bitte)?"
            r"\s*:?\s*(?P<expr>" + chars + r")", clean, re.IGNORECASE)
    try:
        if equation is not None:
            expr = equation.group("expr").strip().lower()
            if expr.count("=") != 1 or "x" not in expr:
                return None
            result = solve_math_symbolic(expr)
            if result["status"] == "PASSED":
                answer = "x = " + result["value"]
            elif result["status"] == "INFINITE_SOLUTIONS":
                answer = "Unendlich viele Lösungen."
            elif result["status"] == "NO_SOLUTION":
                answer = "Keine Lösung."
            else:
                return None
            return {"text": answer, "tool_used": "neral_math_exact",
                    "exact": True, "method": result["method"]}
        if arithmetic is not None:
            expr = arithmetic.group("expr").strip()
            if "=" in expr or "x" in expr.lower():
                return None
            result = eval_python_expr(expr)
            return {"text": result["value"], "tool_used": "neral_math_exact",
                    "exact": True, "method": result["method"]}
    except (MathInputError, ValueError, TypeError):
        return None
    return None


def register_neral_local_chat_api(app, backend, *, experimental_high_backend=None):
    """One local GPU chat lock for both foundations and all chat modes."""
    busy = threading.Lock()
    if experimental_high_backend is not None:
        if (not isinstance(backend, OllamaLocalBackend)
                or not isinstance(experimental_high_backend, OllamaLocalBackend)
                or backend.model != "qwen3:8b"
                or experimental_high_backend.model != "qwen3.5:9b"
                or experimental_high_backend.inference_profile != "measured_nonthinking_2048"):
            raise ValueError("Experimental HIGH requires exact approved local foundations.")

    high_backend = experimental_high_backend or backend
    active_model = [None]

    def prepare_model(selected):
        """Serialized switch, fail closed if the other model cannot unload.

        Also evict the other model before the FIRST generation because Ollama
        may have retained it from an earlier process. No call during direct
        verified-tool answers. Other routes share the chat busy lock.
        """
        if experimental_high_backend is None:
            return
        if active_model[0] != selected.model:
            other = high_backend if selected is backend else backend
            other.unload()
            active_model[0] = selected.model

    # Shared by the attachment chat route, which acquires the same busy lock.
    app._neral_prepare_model = prepare_model

    @app.get("/api/neral/chat/status")
    def local_chat_status():
        return jsonify({**_provenance(backend),
                        "experimental_high_enabled": experimental_high_backend is not None,
                        "experimental_high_model": (
                            high_backend.model if experimental_high_backend is not None else None)})

    @app.post("/api/neral/chat/compare")
    def local_chat_compare():
        """Local only: equal prompt and decoding, real PEFT adapter OFF vs ON."""
        if not request.is_json:
            return jsonify(error="Nur JSON-Anfragen erlaubt."), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"prompt"}:
            return jsonify(error="A/B-Test erlaubt ausschließlich einen Prompt ohne Verlauf."), 400
        try:
            data = _messages(payload)
        except (TypeError, ValueError, UnicodeError):
            return jsonify(error="Ungültiger oder zu langer Vergleichs-Prompt."), 400
        origin = _provenance(backend)
        compare = getattr(backend, "compare_with_base", None)
        if not origin["compare_available"] or not callable(compare):
            return jsonify(error="A/B-Vergleich nur mit lokalem NERAL-LoRA verfügbar."), 409
        if not busy.acquire(blocking=False):
            return jsonify(error="NERAL berechnet bereits eine Antwort."), 409
        try:
            data[0]["content"] = COMPARE_SYSTEM
            started = time.monotonic()
            try:
                result = compare(data)
            except Exception:
                # Never expose local GPU paths, exception traces, or private text.
                return jsonify(error="Lokaler Adapter-A/B-Vergleich fehlgeschlagen."), 503
            if (not isinstance(result, dict) or set(result) != {"base", "neral"}
                    or any(not isinstance(value, str) or not value.strip()
                           or len(value.encode("utf-8")) > 32768
                           for value in result.values())):
                return jsonify(error="Lokales Modell lieferte ungültige A/B-Antworten."), 503
            return jsonify({
                **result,
                "model_generated": True,
                "same_prompt": True,
                "same_decoding": True,
                "tools_used": False,
                "seconds": round(time.monotonic() - started, 3),
                **_provenance(backend),
            })
        finally:
            busy.release()

    @app.post("/api/neral/chat/generate-stream")
    def local_chat_generate_stream():
        """Live, bounded NDJSON from real Ollama FAST output; legacy API unchanged.

        The stream never includes internal model thoughts. It is not considered
        a completed answer until a `final` event passes all output checks.
        """
        if not request.is_json:
            return jsonify(error="Nur JSON-Anfragen erlaubt."), 415
        try:
            data = _messages(request.get_json(silent=True))
            data[0]["content"] = build_identity_prompt(
                "web", _provenance(backend)["model_origin"])
        except (TypeError, ValueError, UnicodeError):
            return jsonify(error="Ungültiger oder zu langer Chatinhalt."), 400
        if not busy.acquire(blocking=False):
            return jsonify(error="NERAL berechnet bereits eine Antwort."), 409

        def frame(event):
            return json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"

        def events():
            started = time.monotonic()
            try:
                identity = _runtime_identity_reply(data[-1]["content"], backend)
                if identity is None and len(data) == 2:
                    from neral_agent_quantitative import calculate_explicit
                    exact = calculate_explicit(data[-1]["content"])
                    if exact is None:
                        exact = _verified_math_request(data[-1]["content"])
                    if exact is None:
                        from neral_percentage_tool import calculate_percentage
                        exact = calculate_percentage(data[-1]["content"])
                else:
                    exact = None
                if identity is not None or exact is not None:
                    direct = identity if identity is not None else exact
                    result = {
                        **direct, "mode": "LOW",
                        "model_generated": False,
                        "seconds": round(time.monotonic() - started, 3),
                        **_provenance(backend),
                        "response_mode": "local_stream_protocol_direct",
                    }
                    yield frame({"type": "final", "result": result})
                    return

                prepare_model(backend)
                stream = getattr(backend, "respond_chat_stream", None)
                parts = []
                total_bytes = 0
                # Backends without streaming support still return a real
                # complete answer in a final event, never fake token chunks.
                if callable(stream):
                    for part in stream(data):
                        if not isinstance(part, str):
                            raise ValueError("Ungültiger Textabschnitt.")
                        total_bytes += len(part.encode("utf-8"))
                        if total_bytes > 32768:
                            raise ValueError("Antwort zu lang.")
                        if part:
                            parts.append(part)
                            yield frame({"type": "delta", "text": part})
                else:
                    reply = getattr(backend, "respond_chat", None)
                    answer = reply(data) if callable(reply) else backend.respond(data)
                    if not isinstance(answer, str):
                        raise ValueError("Ungültige Modellantwort.")
                    parts.append(answer)
                text = "".join(parts)
                if not text.strip() or len(text.encode("utf-8")) > 32768:
                    raise ValueError("Unvollständige Modellantwort.")
                result = {
                    "text": text, "mode": "LOW", "model_generated": True,
                    "seconds": round(time.monotonic() - started, 3),
                    **_provenance(backend),
                    "response_mode": "local_streaming" if callable(stream)
                                     else "local_non_streaming_fallback",
                }
                yield frame({"type": "final", "result": result})
            except GeneratorExit:
                raise
            except OllamaCompletionError as exc:
                yield frame({"type": "error",
                    "error": _SAFE_OLLAMA_ERROR_TEXT[exc.reason],
                    "error_code": exc.reason})
            except Exception:
                # Do not expose raw GPU prompts, hidden reasoning or filesystem data.
                yield frame({"type": "error", "error":
                    "Lokale Generierung fehlgeschlagen oder nicht vollständig abgeschlossen."})
            finally:
                busy.release()

        return Response(stream_with_context(events()),
                        mimetype="application/x-ndjson",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    @app.post("/api/neral/chat/generate")
    def local_chat_generate():
        if not request.is_json:
            return jsonify(error="Nur JSON-Anfragen erlaubt."), 415
        try:
            data = _messages(request.get_json(silent=True))
            # Tell the model the actual runtime origin; UI provenance stays explicit.
            data[0]["content"] = build_identity_prompt(
                "web", _provenance(backend)["model_origin"]
            )
        except (TypeError, ValueError, UnicodeError):
            return jsonify(error="Ungültiger oder zu langer Chatinhalt."), 400
        if not busy.acquire(blocking=False):
            return jsonify(error="NERAL berechnet bereits eine Antwort."), 409
        try:
            started = time.monotonic()
            identity = _runtime_identity_reply(data[-1]["content"], backend)
            if identity is not None:
                return jsonify({**identity, "mode": "LOW",
                                "seconds": round(time.monotonic() - started, 3),
                                **_provenance(backend)})
            # Empty history only: never override an ongoing conversation.
            if len(data) == 2:
                from neral_agent_quantitative import calculate_explicit
                exact = calculate_explicit(data[-1]["content"])
                if exact is None:
                    exact = _verified_math_request(data[-1]["content"])
                if exact is None:
                    from neral_percentage_tool import calculate_percentage
                    exact = calculate_percentage(data[-1]["content"])
            else:
                exact = None
            if exact is not None:
                return jsonify({**exact, "mode": "LOW", "model_generated": False,
                                "seconds": round(time.monotonic() - started, 3),
                                **_provenance(backend)})
            try:
                # Ollama agent requests enforce JSON; normal conversations do
                # not. HF adapter backends use their existing local respond().
                prepare_model(backend)
                reply = getattr(backend, "respond_chat", None)
                text = reply(data) if callable(reply) else backend.respond(data)
            except OllamaCompletionError as exc:
                return jsonify(error=_SAFE_OLLAMA_ERROR_TEXT[exc.reason],
                               error_code=exc.reason), 503
            except Exception:
                # Do not leak local paths, model credentials or private prompts.
                return jsonify(error="Lokale Modell-Inferenz nicht verfügbar oder fehlgeschlagen."), 503
            if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32768:
                return jsonify(error="Lokales Modell lieferte eine ungültige Antwort."), 503
            return jsonify({
                "text": text,
                "mode": "LOW",
                "model_generated": True,
                "seconds": round(time.monotonic() - started, 3),
                **_provenance(backend),
            })
        finally:
            busy.release()

    @app.post("/api/neral/chat/think")
    def local_chat_think():
        """Opt-in local THINK: one response and at most one checked correction.

        Shares the normal chat's busy lock. No remote APIs, generated code
        execution, automatic LoRA promotion or persistent prompt logs.
        """
        if not request.is_json:
            return jsonify(error="Nur JSON-Anfragen erlaubt."), 415
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError("Invalid JSON")
            experimental_tir = payload.get("experimental_tir", False)
            if type(experimental_tir) is not bool:
                raise ValueError("Invalid experimental flag")
            # Keep _messages strict; a TIR switch is not a new chat message.
            clean = {k: v for k, v in payload.items()
                     if k != "experimental_tir"}
            data = _messages(clean)
            data[0]["content"] = build_identity_prompt(
                "web", _provenance(backend)["model_origin"]
            )
        except (TypeError, ValueError, UnicodeError):
            return jsonify(error="Ungültiger oder zu langer THINK-Chatinhalt."), 400
        if experimental_tir and os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1":
            return jsonify(error="Experimenteller TIR-Modus nicht freigeschaltet."), 403
        if (experimental_tir and (
                not isinstance(high_backend, OllamaLocalBackend)
                or high_backend.model != "qwen3.5:9b"
                or high_backend.inference_profile != "measured_nonthinking_2048")):
            return jsonify(error="TIR erfordert das lokale Qwen3.5:9B Testprofil."), 409
        if not busy.acquire(blocking=False):
            return jsonify(error="NERAL berechnet bereits eine Antwort."), 409
        try:
            started = time.monotonic()
            identity = _runtime_identity_reply(data[-1]["content"], high_backend)
            if identity is not None:
                return jsonify({**identity, "mode": "HIGH",
                                "seconds": round(time.monotonic() - started, 3),
                                **_provenance(high_backend)})
            from neral_think import think_reply
            # Dedicated Ollama reasoning only for explicit THINK requests;
            # other HF backends keep their existing verified response path.
            local_respond = getattr(high_backend, "respond_think", None)
            if not callable(local_respond):
                local_respond = getattr(high_backend, "respond_chat", None)
            if not callable(local_respond):
                local_respond = high_backend.respond
            # Only a real local Ollama backend receives opt-in second review.
            deep_review = isinstance(high_backend, OllamaLocalBackend)
            if deep_review:
                from neral_high_verified_first import direct_high_reply
                direct = direct_high_reply(data)
                if direct is not None:
                    if experimental_tir:
                        direct = {**direct, "tool_calls_triggered": 1,
                                  "tool_calls_succeeded": 1,
                                  "experimental_tir": True}
                    return jsonify({**direct, **_provenance(high_backend)})
            prepare_model(high_backend)
            if experimental_tir:
                from neral_tir_experimental import run_experimental_tir
                result = run_experimental_tir(data, high_backend.respond_chat)
                return jsonify({**result, "experimental_tir": True,
                                **_provenance(high_backend)})
            result = think_reply(data, local_respond, deep_review=deep_review)
            return jsonify({**result, **_provenance(high_backend)})
        except OllamaCompletionError as exc:
            return jsonify(error=_SAFE_OLLAMA_ERROR_TEXT[exc.reason],
                           error_code=exc.reason), 503
        except Exception:
            # Never leak local paths, checkpoints, prompts or model outputs.
            return jsonify(error="Lokaler THINK-Modus konnte nicht ausgeführt werden."), 503
        finally:
            busy.release()

    return busy
