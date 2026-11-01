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
        "max_new_tokens": 128,
        "quality": "standard",
    },
)


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
            started = time.monotonic()
            response = client.post(
                "/api/generate",
                json={"mode": "assistant", "prompt": case["prompt"],
                      "history": [], "stream": False, "quality": case["quality"],
                      "max_new_tokens": case["max_new_tokens"], "temperature": 0.0},
                base_url="http://localhost",
            )
            payload = response.get_json(silent=True)
            summaries.append(assess(response.status_code, payload, case,
                                    time.monotonic() - started))
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
