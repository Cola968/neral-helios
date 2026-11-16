"""NERAL 1.1 experimental *local-only* bounded raw Qwen fallback.

Optional, explicitly gated for the adaptive THINK experiment. PR537 still
chooses the fallback using untouched original user messages and cannot
promote unverified candidates. This transport may shorten or refuse truncated
Qwen answers; it is NOT guaranteed quality-equivalent to 2048-token Qwen.
"""
from __future__ import annotations

from neral_nl_local_ollama import _localhost_transport

MODEL="qwen3.5:9b"
FALLBACK_MAX_TOKENS=1536
MAX_REPLY_BYTES=32768
MAX_ORIGINAL_BYTES=16384

def make_bounded_fallback(model_tag: str, *, transport=None,
                          formatted_contract: bool=False):
    if type(model_tag) is not str or model_tag!=MODEL:
        raise ValueError("Only pinned local HIGH Qwen is eligible for bounded fallback")
    if type(formatted_contract) is not bool:
        raise ValueError("Invalid benchmark format gate")
    if transport is not None and not callable(transport):
        raise ValueError("Invalid local transport")
    send=_localhost_transport if transport is None else transport

    def respond(original: list[dict]) -> str:
        if (type(original) is not list or not 2<=len(original)<=14
                or type(original[0]) is not dict
                or type(original[-1]) is not dict
                or original[0].get("role")!="system"
                or original[-1].get("role")!="user"
                or any(type(m) is not dict or set(m)!={"role","content"}
                       or m["role"] not in ("system","assistant","user")
                       or type(m["content"]) is not str or not m["content"].strip()
                       for m in original)
                or sum(len(m["content"].encode("utf8")) for m in original)
                   > MAX_ORIGINAL_BYTES):
            raise ValueError("Invalid original source messages")
        for i,m in enumerate(original[1:],start=1):
            expected="user" if i==len(original)-1 else (
                "user" if i%2==1 else "assistant")
            if m["role"]!=expected:
                raise ValueError("Untrusted chat role ordering")
        messages=[dict(row) for row in original]
        if formatted_contract:
            from neral_answer_contract import BENCHMARK_FORMAT_V1
            if BENCHMARK_FORMAT_V1 not in messages[0]["content"]:
                messages[0]["content"] += BENCHMARK_FORMAT_V1
        payload={
            "model":MODEL,
            "messages":messages,
            "stream":False,"think":False,
            "keep_alive":"5m",
            "options":{
                "temperature":0,"num_predict":FALLBACK_MAX_TOKENS,
                "num_ctx":4096,"repeat_penalty":1.0,"seed":0,
            },
        }
        result=send(payload)
        # The unmodified local Qwen control has a 2048-token budget. To
        # prevent a formerly correct control answer from being lost solely
        # to the experimental 1536 limit, retry exactly once with the same
        # original turns, seed, decoding and a 2048-token budget.
        # Never recycle or expose truncated text from the first attempt.
        if (type(result) is dict and result.get("done") is True
                and result.get("done_reason")=="length"):
            retry={
                **payload,
                "options":{**payload["options"],"num_predict":2048},
            }
            result=send(retry)
        if (type(result) is not dict or result.get("done") is not True
                or result.get("done_reason")!="stop"
                or type(result.get("message")) is not dict):
            raise ValueError("Qwen budget reached or response not fully completed")
        content=result["message"].get("content")
        if (type(content) is not str or not content.strip()
                or len(content.encode("utf8"))>MAX_REPLY_BYTES
                or "<think>" in content.lower() or "</think>" in content.lower()):
            raise ValueError("Invalid or private Qwen final response")
        if formatted_contract:
            from neral_answer_contract import parse_terminal_final_integer
            if parse_terminal_final_integer(content) is None:
                raise ValueError("No complete FINAL_ANSWER from Qwen")
        return content
    return respond
