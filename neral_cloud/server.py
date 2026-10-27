"""NERAL Cloud Beta: minimal, auditable serverless-inference bridge.

No local GPU, model weights, shell tools, or model training.
This service can be hosted on a CPU web host while Qwen3.5 runs at a separate
Hugging Face Inference Provider. The user must enable/provider-fund inference.
"""
from __future__ import annotations

from collections import defaultdict, deque
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
import json
import os
import socket
import time
import urllib.error
import urllib.request

HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "Qwen/Qwen3.5-9B"
MAX_UPSTREAM_BYTES = 128 * 1024
MAX_REQUEST_BYTES = 16 * 1024
from neral_cloud.personality import PERSONA_VERSION, SYSTEM_PROMPT

STATIC = Path(__file__).with_name("static")


class NeralCloudError(ValueError):
    pass


class UpstreamUnavailable(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RateGate:
    def __init__(self, max_calls=8, window_s=60):
        self.max_calls=max_calls
        self.window_s=window_s
        self.lock=Lock()
        self.hits=defaultdict(deque)

    def allow(self, key, now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            q=self.hits[key]
            while q and now-q[0]>=self.window_s:
                q.popleft()
            if len(q)>=self.max_calls:
                return False
            q.append(now)
            return True


RATE=RateGate()

class ProviderBudget:
    """Best-effort *per-process* cap. NOT a durable HF billing/spend limit.

    Count every dispatched request (including failures) because failed
    upstream calls can still consume billable tokens.
    """
    def __init__(self, max_attempts=60, max_inflight=2):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 100000:
            raise ValueError("max_attempts must be a positive integer")
        if type(max_inflight) is not int or not 1 <= max_inflight <= 100:
            raise ValueError("max_inflight must be a positive integer")
        self.max_attempts=max_attempts
        self.max_inflight=max_inflight
        self.total_attempts=0
        self.active=0
        self.lock=Lock()

    def enter(self):
        with self.lock:
            if self.total_attempts>=self.max_attempts:
                return "exhausted"
            if self.active>=self.max_inflight:
                return "busy"
            self.total_attempts+=1
            self.active+=1
            return "ok"

    def leave(self):
        with self.lock:
            if self.active<1:
                raise RuntimeError("ProviderBudget lease imbalance")
            self.active-=1


PROVIDER_BUDGET=ProviderBudget()



def _enabled(env):
    return bool(
        env.get("NERAL_CLOUD_INFERENCE_ENABLED")=="1"
        and isinstance(env.get("NERAL_HF_TOKEN"),str)
        and env["NERAL_HF_TOKEN"].startswith("hf_")
        and len(env["NERAL_HF_TOKEN"])>=12
        and isinstance(env.get("NERAL_BETA_ACCESS_KEY"),str)
        and len(env["NERAL_BETA_ACCESS_KEY"])>=16
    )


def public_configuration(env=None):
    env=os.environ if env is None else env
    return {
        "product":"NERAL Assistant",
        "persona_version":PERSONA_VERSION,
        "model":MODEL_ID,
        "model_provenance":"Qwen3.5-9B (Qwen/Alibaba, Apache-2.0); NERAL is the application",
        "available":_enabled(env),
        "requires_invite":True,
        "cloud_inference":True,
        "tool_access":False,
        "demo":False,
        "beta":True,
    }


def clean_messages(prompt,history):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
        raise NeralCloudError("Invalid message length")
    if not isinstance(history,list) or len(history)>12 or len(history)%2:
        raise NeralCloudError("Invalid chat history")
    if len(prompt.encode("utf-8"))>8000:
        raise NeralCloudError("Message exceeds byte limit")
    msgs=[{"role":"system","content":SYSTEM_PROMPT}]
    total=len(prompt.encode("utf-8"))
    for i,row in enumerate(history):
        if not isinstance(row,dict) or set(row)!={"role","content"}:
            raise NeralCloudError("Invalid history item")
        role="user" if i%2==0 else "assistant"
        if row["role"]!=role or not isinstance(row["content"],str) or not row["content"].strip():
            raise NeralCloudError("Invalid chat roles or content")
        if len(row["content"])>4000:
            raise NeralCloudError("Previous message exceeds per-turn limit")
        total+=len(row["content"].encode("utf-8"))
        if total>12000:
            raise NeralCloudError("Chat context exceeds byte limit")
        msgs.append({"role":role,"content":row["content"].strip()})
    msgs.append({"role":"user","content":prompt.strip()})
    return msgs


def upstream_chat(url,body,token,timeout=35):
    """Fixed HTTPS endpoint, proxy bypass, no redirects, bounded response."""
    if url!=HF_ROUTER:
        raise NeralCloudError("Non-allowlisted inference host")
    payload=json.dumps(body,ensure_ascii=False).encode("utf-8")
    request=urllib.request.Request(
        HF_ROUTER,data=payload,method="POST",
        headers={"Content-Type":"application/json",
                 "Accept":"application/json","Authorization":"Bearer "+token}
    )
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    try:
        with opener.open(request,timeout=timeout) as response:
            raw=response.read(MAX_UPSTREAM_BYTES+1)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, socket.timeout, TimeoutError) as exc:
        raise UpstreamUnavailable("Upstream inference unavailable") from exc
    if len(raw)>MAX_UPSTREAM_BYTES:
        raise UpstreamUnavailable("Upstream response too large")
    try:
        result=json.loads(raw.decode("utf-8"))
    except (ValueError,UnicodeDecodeError) as exc:
        raise UpstreamUnavailable("Invalid inference JSON") from exc
    if not isinstance(result,dict):
        raise UpstreamUnavailable("Invalid inference result")
    return result


def chat_turn(data, *, access_key, client_ip, env=None, transport=upstream_chat,
              rate_gate=None, provider_budget=None):
    env=os.environ if env is None else env
    if not _enabled(env):
        return 503,{"error":"NERAL Cloud Beta is not configured for online inference.",
                    "code":"configuration_required"}
    if not isinstance(access_key,str) or not compare_digest(access_key,env["NERAL_BETA_ACCESS_KEY"]):
        return 401,{"error":"Beta access required.","code":"unauthorized"}
    if not isinstance(data,dict) or set(data)-{"message","history"}:
        return 400,{"error":"Invalid request.","code":"invalid_request"}
    try:
        messages=clean_messages(data.get("message"),data.get("history",[]))
    except NeralCloudError:
        return 400,{"error":"Invalid message or conversation history.","code":"invalid_request"}
    gate=RATE if rate_gate is None else rate_gate
    if not gate.allow(client_ip):
        return 429,{"error":"Too many requests. Please try again later.","code":"rate_limited"}
    budget=PROVIDER_BUDGET if provider_budget is None else provider_budget
    permitted=budget.enter()
    if permitted!="ok":
        return 429,{
            "error":"NERAL Cloud Beta request allowance is temporarily exhausted." if permitted=="exhausted"
                else "NERAL Cloud Beta is at its concurrent request limit.",
            "code":"provider_budget_exhausted" if permitted=="exhausted" else "provider_busy",
        }
    body={"model":MODEL_ID,"messages":messages,"stream":False,
          "temperature":0.3,"max_tokens":256}
    try:
        result=transport(HF_ROUTER,body,env["NERAL_HF_TOKEN"],35)
        if not isinstance(result,dict):
            raise UpstreamUnavailable("Invalid provider response")
        choices=result.get("choices")
        if not isinstance(choices,list) or not choices:
            raise UpstreamUnavailable("Empty provider choices")
        msg=choices[0].get("message") if isinstance(choices[0],dict) else None
        if not isinstance(msg,dict) or msg.get("role")!="assistant":
            raise UpstreamUnavailable("Invalid provider message")
        answer=msg.get("content")
        if not isinstance(answer,str) or not answer.strip() or len(answer)>12000:
            raise UpstreamUnavailable("Empty or malformed answer")
    except Exception:
        # Never surface upstream API body, HF token, or user prompt to client/log.
        return 503,{"error":"Model service is temporarily unavailable.","code":"upstream_unavailable"}
    finally:
        budget.leave()
    return 200,{
        "answer":answer.strip(),"mode":"online-inference",
        "base_model":MODEL_ID,"product":"NERAL Assistant",
        "persona_version":PERSONA_VERSION,
        "model_origin":"Qwen3.5-9B, Apache-2.0; not trained from scratch by NERAL",
        "tools_used":[],"cloud":True,
        "warning":"AI-generated text can be inaccurate.",
    }


class NeralHandler(BaseHTTPRequestHandler):
    server_version="NERALBeta/0.1"
    sys_version=""

    def _send(self,status,content,ctype="application/json; charset=utf-8"):
        body=content if isinstance(content,bytes) else json.dumps(content,ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("X-Frame-Options","DENY")
        self.send_header("Referrer-Policy","no-referrer")
        self.send_header("Permissions-Policy","camera=(), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self):
        origin=self.headers.get("Origin")
        if not origin:
            return True
        p=urlsplit(origin)
        return p.scheme in ("http","https") and p.netloc==self.headers.get("Host","") and not p.username

    def do_GET(self):
        if self.path=="/health":
            return self._send(200,{"status":"ok","service":"neral-cloud-beta",
                                   "inference_ready":_enabled(os.environ)})
        if self.path=="/api/config":
            return self._send(200,public_configuration())
        resources={
            "/":("index.html","text/html; charset=utf-8"),
            "/app.js":("app.js","application/javascript; charset=utf-8"),
            "/style.css":("style.css","text/css; charset=utf-8"),
        }
        if self.path not in resources:
            return self._send(404,{"error":"Not found"})
        name,ctype=resources[self.path]
        try:
            data=(STATIC/name).read_bytes()
        except OSError:
            return self._send(503,{"error":"Static app unavailable"})
        return self._send(200,data,ctype)

    def do_POST(self):
        if self.path!="/api/chat":
            return self._send(404,{"error":"Not found"})
        if not self._same_origin():
            return self._send(403,{"error":"Origin not allowed"})
        if self.headers.get("Content-Type","").split(";")[0].strip().lower()!="application/json":
            return self._send(415,{"error":"JSON required"})
        try:
            size=int(self.headers.get("Content-Length","-1"))
            if size<1 or size>MAX_REQUEST_BYTES:
                return self._send(413,{"error":"Payload too large or absent"})
            body=json.loads(self.rfile.read(size).decode("utf-8"))
        except (ValueError,UnicodeDecodeError):
            return self._send(400,{"error":"Invalid JSON"})
        status,response=chat_turn(
            body,access_key=self.headers.get("X-Neral-Access",""),
            client_ip=self.client_address[0]
        )
        return self._send(status,response)

    def log_message(self, fmt, *args):
        # Never log chat input, access code, inference provider secrets.
        return


def main():
    port=int(os.environ.get("PORT","8765"))
    if not 1<=port<=65535:
        raise ValueError("Invalid port")
    httpd=ThreadingHTTPServer(("0.0.0.0",port),NeralHandler)
    print("NERAL Cloud Beta HTTP service started; inference enabled: "+str(_enabled(os.environ)),
          flush=True)
    httpd.serve_forever()


if __name__=="__main__":
    main()
