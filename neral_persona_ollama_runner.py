"""NERAL persona smoke runner: explicitly invoked, fixed localhost Ollama API.

Never runs during CI, never trains or downloads models, never accesses cloud
services, and never sends prompts to untrusted hosts. The exact-match smoke
suite is a narrow diagnostic, NOT NIS2 nor a personality quality benchmark.
Injected test transports produce fake-marked outputs that cannot be scored.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path

from neral_persona_data import load_seed
from neral_persona_eval import load_holdout, validate_holdout

MODELS = frozenset({"qwen3.5:4b", "qwen3.5:9b"})
BASE_URL = "http://127.0.0.1:11434"
MAX_REPLY_BYTES = 256 * 1024
MAX_PROMPT_LEN = 2000


class LocalRunError(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _local_api(endpoint: str, payload: dict | None = None) -> dict:
    """Only allow these two literal Ollama endpoints on fixed loopback."""
    if endpoint not in ("/api/tags", "/api/chat"):
        raise LocalRunError("Disallowed local inference endpoint")
    if endpoint == "/api/chat" and not isinstance(payload, dict):
        raise LocalRunError("Model payload required")
    if endpoint == "/api/tags" and payload is not None:
        raise LocalRunError("Tags endpoint never accepts a payload")
    data = None if payload is None else json.dumps(
        payload, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + endpoint, data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type":"application/json", "Accept":"application/json"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    try:
        with opener.open(req, timeout=120 if endpoint == "/api/chat" else 4) as response:
            raw = response.read(MAX_REPLY_BYTES+1)
    except (OSError, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise LocalRunError("Local Ollama is not reachable; no fallback inference") from exc
    if len(raw)>MAX_REPLY_BYTES:
        raise LocalRunError("Ollama response exceeds maximum bytes")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise LocalRunError("Local Ollama returned invalid JSON") from exc
    if not isinstance(result,dict):
        raise LocalRunError("Local Ollama payload is not an object")
    return result


def make_payload(prompt: str, model: str) -> dict:
    if not isinstance(model,str) or model not in MODELS:
        raise LocalRunError("Only the two explicit, already-local Qwen model ids are allowed")
    if (not isinstance(prompt,str) or not prompt.strip()
        or len(prompt)>MAX_PROMPT_LEN or len(prompt.encode("utf-8"))>4096):
        raise LocalRunError("Invalid frozen prompt")
    return {
        "model":model,
        "messages":[{"role":"user","content":prompt.strip()}],
        "stream":False, "think":False, "tools":[], "keep_alive":"5m",
        "options":{"temperature":0.0,"num_ctx":4096,"num_predict":128},
    }


def _check_digest(inventory: dict, model: str) -> str:
    rows=inventory.get("models") if isinstance(inventory,dict) else None
    if not isinstance(rows,list):
        raise LocalRunError("Local Ollama model inventory unavailable")
    for row in rows:
        if isinstance(row,dict) and row.get("name")==model:
            digest=row.get("digest")
            if isinstance(digest,str) and re.fullmatch(r"[0-9a-fA-F]{64}",digest):
                return digest.lower()
    raise LocalRunError("Requested Qwen model not installed or model digest unavailable")


def collect_ollama(cases: list[dict], *, model: str,
                   model_call=None, tags_call=None) -> dict:
    """Collect complete answers; test injection is explicitly NOT real inference."""
    if not isinstance(cases,list) or not cases or len(cases)>100:
        raise LocalRunError("Invalid bounded case collection")
    try:
        validate_holdout(cases,load_seed())
    except (ValueError,KeyError,TypeError) as exc:
        raise LocalRunError("Frozen holdout validation failed") from exc
    # Intentional: no user-controllable arbitrary model names or providers.
    make_payload(cases[0]["prompt"],model)
    actual = model_call is None and tags_call is None
    model_call = model_call or (lambda payload:_local_api("/api/chat",payload))
    tags_call = tags_call or (lambda:_local_api("/api/tags"))
    try:
        model_digest=_check_digest(tags_call(),model)
    except (ValueError,KeyError,TypeError,AttributeError) as exc:
        raise LocalRunError("No installed model identity could be verified") from exc
    predictions={}
    for case in cases:
        data=model_call(make_payload(case["prompt"],model))
        if not isinstance(data,dict) or data.get("model")!=model or data.get("done") is not True:
            raise LocalRunError("Model identity mismatch or incomplete local response")
        msg=data.get("message")
        if (not isinstance(msg,dict) or msg.get("role")!="assistant"
            or msg.get("tool_calls") or not isinstance(msg.get("content"),str)
            or not msg["content"].strip() or len(msg["content"])>10000):
            raise LocalRunError("Malformed, empty or unauthorized local model response")
        predictions[case["id"]]={
            "text":msg["content"].strip(),
            "model_generated":actual,
            "base_model":model,
            "tools_used":[],
        }
    suite_blob=json.dumps(cases,ensure_ascii=False,sort_keys=True,
                          separators=(",",":"),allow_nan=False).encode("utf-8")
    return {
        "provenance":{
            "runner":"NERAL-P1-native-ollama-smoke",
            "mode":"local_ollama_http" if actual else "simulated_test_transport",
            "model_id":model,
            "model_digest":model_digest,
            "suite_sha256":hashlib.sha256(suite_blob).hexdigest(),
            "generation":{"temperature":0.0,"num_ctx":4096,"num_predict":128,
                          "think":False,"tools":False},
            "warning":(
                "This file alone cannot cryptographically prove genuine hardware inference. "
                "Keep device logs and digest. Benchmark is narrow and is NOT NIS2."
            ),
        },
        "predictions":predictions,
    }


def save_run(report: dict, output_path: Path | str) -> None:
    """Never silently overwrite or upload private run files."""
    path=Path(output_path)
    if path.suffix!=".json" or not path.parent.is_dir():
        raise LocalRunError("Output requires an existing directory and .json suffix")
    if not isinstance(report,dict) or set(report)!={"provenance","predictions"}:
        raise LocalRunError("Invalid native run format")
    data=(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
    if len(data)>512_000:
        raise LocalRunError("Oversized baseline run")
    with path.open("xb") as f:
        f.write(data)


def main() -> None:
    import argparse
    from neral_persona_eval import grade_predictions
    parser=argparse.ArgumentParser(
        description="Run real native NERAL persona smoke on local Ollama (never in CI)"
    )
    parser.add_argument("--model",choices=sorted(MODELS),default="qwen3.5:9b")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    cases=load_holdout()
    result=collect_ollama(cases,model=args.model)
    save_run(result,args.output)
    summary=grade_predictions(cases,result["predictions"])
    print(json.dumps({
        "output":str(args.output), "provenance":result["provenance"],
        "smoke_score":summary,
    },ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
