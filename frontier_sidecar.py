"""Local-only adapter between Helios Workspace and a Frontier 350 inference sidecar."""
from __future__ import annotations

import json
from urllib.error import HTTPError,URLError
from urllib.request import HTTPRedirectHandler,ProxyHandler,Request,build_opener

from model_catalog import FRONTIER350_RUNTIME

MAX_RESPONSE_BYTES=256*1024

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None

# The sidecar is a loopback-only trust boundary. Never inherit system proxy
# settings and never follow redirects that could escape 127.0.0.1.
_OPENER=build_opener(ProxyHandler({}),_NoRedirect())

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
        with _OPENER.open(req,timeout=timeout) as response:
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

def _budget(health,max_new_tokens):
    limit=health.get("max_new_tokens",64)
    if type(limit) is not int or not 1<=limit<=512:limit=64
    if type(max_new_tokens) is not int or max_new_tokens<1:raise FrontierUnavailable("Ungültiges Antwortbudget.")
    return min(max_new_tokens,limit)

def chat(row,messages,max_new_tokens,request_id=None,timeout=125.0):
    health=probe(row);budget=_budget(health,max_new_tokens)
    payload={"messages":messages,"max_new_tokens":budget}
    if request_id is not None:payload["request_id"]=request_id
    result=_json_request(_base(row)+"/chat",payload,timeout=timeout)
    answer=result.get("answer");generated=result.get("generated_tokens")
    if not isinstance(answer,str) or type(generated) is not int or generated<0:
        raise FrontierUnavailable("Frontier-Runtime lieferte eine ungültige Chatantwort.")
    return result,health

def stream_chat(row,messages,max_new_tokens,request_id,on_event,timeout=125.0):
    health=probe(row);budget=_budget(health,max_new_tokens)
    payload={"messages":messages,"max_new_tokens":budget,"request_id":request_id}
    raw=json.dumps(payload,separators=(",",":")).encode("utf-8")
    req=Request(_base(row)+"/stream",data=raw,method="POST",headers={"Content-Type":"application/json"})
    try:
        with _OPENER.open(req,timeout=timeout) as response:
            if response.status!=200:raise FrontierUnavailable("Frontier-Streaming konnte nicht gestartet werden.")
            final=None
            for line in response:
                if not line.strip():continue
                try:event=json.loads(line.decode("utf-8"))
                except (UnicodeError,json.JSONDecodeError) as exc:raise FrontierUnavailable("Ungültiger Frontier-Stream.") from exc
                if not isinstance(event,dict):raise FrontierUnavailable("Ungültiges Frontier-Stream-Ereignis.")
                kind=event.get("type")
                if kind=="progress":on_event(event)
                elif kind=="done":
                    final=event.get("result");break
                elif kind=="error":raise FrontierUnavailable(str(event.get("error") or "Frontier-Streaming fehlgeschlagen."))
            if not isinstance(final,dict):raise FrontierUnavailable("Frontier-Stream endete ohne Ergebnis.")
    except (HTTPError,URLError,TimeoutError,OSError) as exc:
        raise FrontierUnavailable("Frontier-Streaming ist lokal fehlgeschlagen.") from exc
    answer=final.get("answer");generated=final.get("generated_tokens")
    if not isinstance(answer,str) or type(generated) is not int or generated<0:
        raise FrontierUnavailable("Frontier-Runtime lieferte ein ungültiges Stream-Ergebnis.")
    return final,health

def cancel(row,request_id,timeout=1.0):
    if not isinstance(request_id,str):return False
    result=_json_request(_base(row)+"/cancel",{"request_id":request_id},timeout=timeout)
    return result.get("cancelled") is True
