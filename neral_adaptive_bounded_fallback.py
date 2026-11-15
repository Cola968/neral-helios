"""NERAL 1.1 experimental *local-only* bounded raw Qwen fallback.

Optional, explicitly gated for the adaptive THINK experiment. PR537 still
chooses the fallback using untouched original user messages and cannot
promote unverified candidates. This transport may shorten or refuse truncated
Qwen answers; it is NOT guaranteed quality-equivalent to 2048-token Qwen.
"""
from __future__ import annotations

from neral_nl_local_ollama import _localhost_transport

MODEL="qwen3.5:9b"
FALLBACK_MAX_TOKENS=640
MAX_REPLY_BYTES=32768
MAX_ORIGINAL_BYTES=16384

def make_bounded_fallback(model_tag: str, *, transport=None):
    if type(model_tag) is not str or model_tag!=MODEL:
        raise ValueError("Only pinned local HIGH Qwen is eligible for bounded fallback")
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
        payload={
            "model":MODEL,
            "messages":[dict(row) for row in original],
            "stream":False,"think":False,
            "keep_alive":"5m",
            "options":{
                "temperature":0,"num_predict":FALLBACK_MAX_TOKENS,
                "num_ctx":4096,"repeat_penalty":1.05,
            },
        }
        result=send(payload)
        if (type(result) is not dict or result.get("done") is not True
                or result.get("done_reason")!="stop"
                or type(result.get("message")) is not dict):
            raise ValueError("Qwen budget reached or response not fully completed")
        content=result["message"].get("content")
        if (type(content) is not str or not content.strip()
                or len(content.encode("utf8"))>MAX_REPLY_BYTES
                or "<think>" in content.lower() or "</think>" in content.lower()):
            raise ValueError("Invalid or private Qwen final response")
        return content
    return respond
