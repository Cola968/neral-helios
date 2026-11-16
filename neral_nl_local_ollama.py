"""Production-safe opt-in local Ollama structured v2 sampler.

No imports of developer cases, old benchmark data or sealed corpora. Requests
cannot leave 127.0.0.1; host redirects and environment proxies are denied.
CPU test transports can replace the actual transport, no implicit downloads.
"""
from __future__ import annotations

import json
import re
import urllib.request

from neral_agent_autonomy import local_only_opener
from neral_nl_schema_format import get_v2_ollama_format

_LOCAL_ENDPOINT = "http://127.0.0.1:11434/api/chat"
_MODEL_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}:[A-Za-z0-9_.-]{1,50}\Z")


def _localhost_transport(payload: dict) -> dict:
    req = urllib.request.Request(
        _LOCAL_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with local_only_opener().open(req, timeout=90) as response:
        body = response.read(65537)
    if len(body) > 65536:
        raise ValueError("Oversized local inference result")
    reply = json.loads(body)
    if type(reply) is not dict:
        raise ValueError("Invalid local inference JSON envelope")
    return reply


def make_local_extractor(model_tag: str, *, transport=None):
    """Return one strict structured sampling function for an installed local model.

    This is a *candidate* extractor only. Trusted source verification and
    baseline fallback occur in neral_nl_verified_route.py, never here.
    """
    if type(model_tag) is not str or not _MODEL_TAG.fullmatch(model_tag):
        raise ValueError("Unexpected installed-local-model tag")
    if transport is not None and not callable(transport):
        raise ValueError("Invalid local Ollama transport")
    send = transport if transport is not None else _localhost_transport

    def extract(messages: list[dict]) -> str:
        if (type(messages) is not list or len(messages) != 2 or
                any(type(row) is not dict or set(row) != {"role", "content"}
                    or row["role"] not in ("system", "user")
                    or type(row["content"]) is not str or
                    not row["content"].strip()
                    for row in messages) or
                messages[0]["role"] != "system" or
                messages[1]["role"] != "user"):
            raise ValueError("Invalid bounded local extraction request")
        packet = {
            "model": model_tag,
            "messages": [dict(row) for row in messages],
            "stream": False, "think": False,
            "format": get_v2_ollama_format(messages[1]["content"]),
            "options": {
                "temperature": 0, "num_predict": 1024,
                "repeat_penalty": 1.05,
            },
        }
        reply = send(packet)
        if (type(reply) is not dict or reply.get("done") is not True
                or reply.get("done_reason") != "stop"
                or type(reply.get("message")) is not dict
                or type(reply["message"].get("content")) is not str
                or not reply["message"]["content"].strip()
                or len(reply["message"]["content"].encode("utf-8")) > 32768):
            raise ValueError("Local model reply not fully completed")
        return reply["message"]["content"]

    return extract
