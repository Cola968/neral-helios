"""Experimental NERAL 1.1 local QLoRA strategy + tool proposal.

Off by default. One local Ollama /api/chat call yields strategy and payload
together. Rejected/incomplete responses are never tools or final answers.
Trusted execution/verification/fallback live in neral_adaptive_planner.py.
"""
from __future__ import annotations

import json
from neral_nl_local_ollama import _localhost_transport
from neral_adaptive_planner import ERROR_CODES, MAX_REPAIRS, ROUTES

MODEL = "neral-qwen35-qlora-v1:latest"
MAX_OUTPUT_BYTES = 12_000
MAX_CODE_BYTES = 4096
MAX_SCHEMA_BYTES = 8192

# Ollama structured output constrains formatting, NOT correctness.
_FORMAT = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string",
                     "enum": ["Z3_SCHEMA", "PYTHON_SANDBOX", "DIRECT"]},
        "code": {"type": "string", "maxLength": 4096},
        "schema": {"type": "object"},
    },
    "required": ["strategy"],
    "additionalProperties": False,
}
_INSTRUCTION = (
    "NERAL 1.1 EXPERIMENTAL ADAPTIVE PLANNER. "
    "Return ONE JSON object only. Choose EXACTLY ONE strategy: "
    "Z3_SCHEMA with key schema (NERAL constraints v2), "
    "PYTHON_SANDBOX with key code (complete short Python standard-library "
    "script printing exactly one nonnegative integer), or DIRECT (no code/schema). "
    "Do NOT include verified, safe_to_promote, a final answer, confidence, "
    "explanations, Markdown, native tool tags or extra keys. "
    "Prefer Z3_SCHEMA only when the full input is an explicitly supported "
    "fully source-certified finite constraint format. "
    "Prefer PYTHON_SANDBOX for finite combinatorics, constrained enumerations "
    "and exact arithmetic; keep code minimal, deterministic, and within limits. "
    "The application will independently check every problem clause: "
    "executing your code does NOT itself verify your interpretation. "
    "Return DIRECT for natural conversation or unsupported tasks."
)


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate proposal member")
        result[key] = value
    return result


def _signal(signal: object) -> dict:
    if type(signal) is not dict or set(signal) != {
            "preferred_strategy", "allowed_strategies", "repair_round", "feedback_code"}:
        raise ValueError("Invalid controller signal")
    preferred = signal["preferred_strategy"]
    allowed = signal["allowed_strategies"]
    round_ = signal["repair_round"]
    feedback = signal["feedback_code"]
    if (type(preferred) is not str or preferred not in ROUTES
            or type(allowed) not in (list, tuple)
            or not allowed or len(allowed) > 3
            or any(type(x) is not str or x not in ROUTES for x in allowed)
            or len(set(allowed)) != len(allowed)
            or "DIRECT" not in allowed or preferred not in allowed
            or type(round_) is not int or not 0 <= round_ <= MAX_REPAIRS
            or (feedback is not None and
                (type(feedback) is not str or feedback not in ERROR_CODES))
            or (round_ == 0 and feedback is not None)
            or (round_ > 0 and feedback is None)):
        raise ValueError("Untrusted or inconsistent controller signal")
    return dict(signal)


def _parse_proposal(text: object, allowed: list[str] | tuple[str, ...]) -> dict:
    if (type(text) is not str or not 2 <= len(text.encode("utf8")) <= MAX_OUTPUT_BYTES):
        raise ValueError("Invalid proposal size")
    packet = json.loads(text, object_pairs_hook=_strict_pairs)
    if type(packet) is not dict or packet.get("strategy") not in allowed:
        raise ValueError("Unsupported proposal route")
    if packet["strategy"] == "DIRECT":
        if set(packet) != {"strategy"}:
            raise ValueError("Unexpected direct proposal payload")
    elif packet["strategy"] == "PYTHON_SANDBOX":
        if (set(packet) != {"strategy", "code"}
                or type(packet["code"]) is not str
                or not 1 <= len(packet["code"].encode("utf8")) <= MAX_CODE_BYTES):
            raise ValueError("Invalid Python proposal")
    elif packet["strategy"] == "Z3_SCHEMA":
        if (set(packet) != {"strategy", "schema"} or
                type(packet["schema"]) is not dict or
                len(json.dumps(packet["schema"], separators=(",", ":")).encode("utf8"))
                    > MAX_SCHEMA_BYTES):
            raise ValueError("Invalid Z3 schema proposal")
    else:
        raise ValueError("Unknown proposal route")
    return packet


def make_local_adaptive_proposer(model_tag: str, *, transport=None):
    """Require the pinned local QLoRA tag; provide injectable CPU-only transport.

    The call is not a separate planner model pass: its single response already
    contains the chosen tool and payload. All repair calls keep this same tag.
    """
    if type(model_tag) is not str or model_tag != MODEL:
        raise ValueError("Adaptive planner requires explicitly pinned local QLoRA")
    if transport is not None and not callable(transport):
        raise ValueError("Invalid transport")
    send = _localhost_transport if transport is None else transport

    def propose(messages: list[dict], signal: dict) -> dict:
        if (type(messages) is not list or len(messages) != 2 or
                any(type(m) is not dict or set(m) != {"role", "content"}
                    or type(m["content"]) is not str or not m["content"].strip()
                    for m in messages) or
                messages[0]["role"] != "system" or messages[1]["role"] != "user"):
            raise ValueError("Only standalone one-turn local planning allowed")
        params = _signal(signal)
        allowed = list(params["allowed_strategies"])
        feedback = params["feedback_code"] or "none"
        additional = (
            "\nAllowed strategies: " + ", ".join(allowed)
            + ". Preferred strategy: " + params["preferred_strategy"]
            + ". Repair round: " + str(params["repair_round"])
            + ". Previous rejection (fixed code only): " + feedback
            + ". If the previous route failed, revise the proposed plan. "
            + "Never guess the result or claim independent verification."
        )
        packet = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content":
                 messages[0]["content"] + "\n" + _INSTRUCTION + additional},
                {"role": "user", "content": messages[1]["content"]},
            ],
            "stream": False, "think": False,
            "format": _FORMAT, "keep_alive": "5m",
            "options": {
                "temperature": 0, "repeat_penalty": 1.05,
                "num_predict": 1024, "num_ctx": 4096,
            },
        }
        reply = send(packet)
        if (type(reply) is not dict or reply.get("done") is not True
                or reply.get("done_reason") != "stop"
                or type(reply.get("message")) is not dict):
            raise ValueError("Incomplete local model proposal")
        return _parse_proposal(reply["message"].get("content"), allowed)

    return propose
