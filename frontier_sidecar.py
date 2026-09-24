"""Local-only adapter between Helios Workspace and a Frontier 350 inference sidecar."""
from __future__ import annotations

import json
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen

from model_catalog import FRONTIER350_RUNTIME

MAX_RESPONSE_BYTES=256*1024

class FrontierUnavailable(ValueError):
    pass

def _base(row):
    if not isinstance(row,dict) or row.get("runtime_family")!=FRONTIER350_RUNTIME:
        raise FrontierUnavailable("Ungültiger Frontier-Runtime-Eintrag.")
    port=row.get("runtime_port")
    if type(port) is not int or not 1024<=port<=65535:
        raise FrontierUnavailable("Kein lokaler Frontier-Runtime-Port registriert.")
    return f"http://127.0.0.1:{port}"

def _json_request(url,payload=None,timeout=1.0):
    data=None if payload is None else json.dumps(payload,separators=(",",":")).encode("utf-8")
    req=Request(url,data=data,method="GET" if data is None else "POST",
                headers={} if data is None else {"Content-Type":"application/json"})
    try:
        with urlopen(req,timeout=timeout) as response:
            if response.status!=200:raise FrontierUnavailable("Frontier-Runtime antwortet nicht bereit.")
            raw=response.read(MAX_RESPONSE_BYTES+1)
    except (HTTPError,URLError,TimeoutError,OSError) as exc:
        raise FrontierUnavailable("Frontier-Runtime ist lokal nicht erreichbar.") from exc
    if len(raw)>MAX_RESPONSE_BYTES:raise FrontierUnavailable("Frontier-Runtime-Antwort ist zu groß.")
    try:value=json.loads(raw.decode("utf-8"))
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise FrontierUnavailable("Frontier-Runtime lieferte ungültiges JSON.") from exc
    if not isinstance(value,dict):raise FrontierUnavailable("Frontier-Runtime lieferte kein Objekt.")
    return value

def probe(row,timeout=0.6):
    health=_json_request(_base(row)+"/health",timeout=timeout)
    if health.get("runtime_family")!=FRONTIER350_RUNTIME:
        raise FrontierUnavailable("Frontier-Runtime-Familie stimmt nicht überein.")
    if health.get("checkpoint_sha256")!=row.get("model_sha256"):
        raise FrontierUnavailable("Frontier-Checkpoint stimmt nicht mit der App-Registrierung überein.")
    if health.get("tokenizer_sha256")!=row.get("tokenizer_sha256"):
        raise FrontierUnavailable("Frontier-Tokenizer stimmt nicht mit der App-Registrierung überein.")
    if health.get("status") not in {"ready","experimental-ready"}:
        raise FrontierUnavailable("Frontier-Runtime ist nicht bereit.")
    device=health.get("device")
    if not isinstance(device,str) or device not in {"cpu","cuda"}:
        raise FrontierUnavailable("Frontier-Runtime meldet ein ungültiges Gerät.")
    return health

def chat(row,messages,max_new_tokens,timeout=125.0):
    health=probe(row)
    limit=health.get("max_new_tokens",64)
    if type(limit) is not int or not 1<=limit<=512:limit=64
    budget=min(max_new_tokens,limit)
    result=_json_request(_base(row)+"/chat",
                         {"messages":messages,"max_new_tokens":budget},timeout=timeout)
    answer=result.get("answer")
    generated=result.get("generated_tokens")
    if not isinstance(answer,str) or type(generated) is not int or generated<0:
        raise FrontierUnavailable("Frontier-Runtime lieferte eine ungültige Chatantwort.")
    return result,health
