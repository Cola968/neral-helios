"""Bounded, read-only native+Flask smoke against an already installed local Ollama model.

Runs only when explicitly triggered via the RTX 4070 branch CI. No training,
downloads, remote models, user chat/file reads, or model changes.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3.5:9b"
LOCAL_OLLAMA = "http://127.0.0.1:11434/api/tags"
FIRST_PROMPT = "Antworte ausschließlich mit BER."
SECOND_PROMPT = "Wie viel ist 17 mal 23? Nenne das Ergebnis."

def preflight():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(LOCAL_OLLAMA, timeout=8) as response:
        data = json.load(response)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise RuntimeError("Ollama tags endpoint did not return a valid model list")
    if MODEL not in [x.get("name") for x in data["models"] if isinstance(x, dict)]:
        raise RuntimeError("Qwen3.5-9B is not locally installed; no automatic pull allowed")
    print("PREFLIGHT_OK local Ollama and Qwen3.5-9B installed", flush=True)

def native_test():
    from neral_assistant import assistant_turn
    before = time.perf_counter()
    result = assistant_turn(FIRST_PROMPT, enable_tools=False, max_new=48,
                            temperature=0.0, think=False)
    elapsed = round(time.perf_counter() - before, 2)
    assert result["base_model"] == MODEL and result["model_generated"] is True
    assert result["mode"] == "neral-assistant" and result["tools_used"] == []
    answer = result["text"].strip()
    print("NATIVE_OUTPUT " + json.dumps({
        "question": FIRST_PROMPT, "answer": answer[:180],
        "instruction_correct": answer == "BER", "wall_seconds": elapsed,
        "generated_tokens": result["generated_tokens"],
        "tokens_per_second": result["tokens_per_second"],
        "tool_calls": result["tools_used"],
    }, ensure_ascii=False), flush=True)
    if answer != "BER":
        raise AssertionError("Native model failed a simple exact-output instruction")

def flask_real_api_test():
    # Never touch any actual project chat DB or existing NERAL deployment.
    with tempfile.TemporaryDirectory(prefix="neral-live-gpu-smoke-") as tempdir:
        os.environ.update({
            "NERAL_ASSISTANT_ENABLED": "1", "NERAL_ASSISTANT_MODEL": MODEL,
            "NERAL_ASSISTANT_THINK": "0",
            "HELIOS_PREPARATION_MODE": "0", "HELIOS_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "HELIOS_STATE_ROOT": tempdir,
            "HELIOS_ASSETS_ROOT": tempdir,
        })
        from app import app
        client = app.test_client()
        landing = client.get("/", base_url="http://localhost")
        assert landing.status_code == 200
        info = client.get("/api/info", base_url="http://localhost")
        assert info.status_code == 200
        cfg = info.get_json()
        assert cfg["assistant"]["enabled"] is True
        assert cfg["assistant"]["base_model"] == MODEL
        assert cfg["assistant"]["local_only"] is True
        print("NERAL_HTTP_SESSION_AND_MODEL_PROVENANCE_OK", flush=True)
        before = time.perf_counter()
        response = client.post("/api/generate", json={
            "prompt": SECOND_PROMPT, "mode": "assistant",
            "history": [], "max_new_tokens": 128,
            "temperature": 0.0, "top_k": 40, "stream": False,
        }, base_url="http://localhost")
        elapsed = round(time.perf_counter() - before, 2)
        payload = response.get_json(silent=True)
        if response.status_code != 200:
            print("NERAL_HTTP_FAILURE " + json.dumps({
                "status": response.status_code,
                "error": str(payload.get("error", ""))[:200] if isinstance(payload, dict) else "invalid JSON",
            }, ensure_ascii=False), flush=True)
            raise AssertionError("NERAL Assistant Flask endpoint could not obtain a real local model answer")
        assert payload["base_model"] == MODEL
        assert payload["mode"] == "neral-assistant"
        assert payload["model_generated"] is True
        assert isinstance(payload.get("text"), str) and payload["text"].strip()
        answer = payload["text"].strip()
        print("NERAL_HTTP_LIVE_OUTPUT " + json.dumps({
            "question": SECOND_PROMPT, "answer": answer[:280],
            "expected": "391", "answer_contains_391": bool(re.search(r"(?<!\d)391(?!\d)", answer)),
            "wall_seconds": elapsed,
            "generated_tokens": payload.get("generated_tokens"),
            "tokens_per_second": payload.get("tokens_per_second"),
            "tools_used": payload.get("tools_used"),
            "model_origin": payload.get("model_origin"),
        }, ensure_ascii=False), flush=True)
        print("NERAL_HTTP_E2E_PASS real Ollama-backed Flask generation; browser itself tested separately in CI",flush=True)

def main():
    preflight()
    native_test()
    flask_real_api_test()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("NERAL_LIVE_SMOKE_FAILED " + type(exc).__name__ + ": " + str(exc)[:260], file=sys.stderr, flush=True)
        raise
