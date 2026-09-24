from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,threading

import pytest

from frontier_sidecar import FrontierUnavailable,cancel,chat,probe,stream_chat
from model_catalog import FRONTIER350_RUNTIME

MODEL_SHA="a"*64
TOKENIZER_SHA="b"*64

class Handler(BaseHTTPRequestHandler):
    health_model_sha=MODEL_SHA
    seen=None
    cancelled=None
    def log_message(self,*args):pass
    def _send(self,status,payload,kind="application/json"):
        raw=payload if isinstance(payload,bytes) else json.dumps(payload).encode()
        self.send_response(status);self.send_header("Content-Type",kind)
        self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if self.path!="/health":return self._send(404,{})
        return self._send(200,{"status":"experimental-ready","runtime_family":FRONTIER350_RUNTIME,
                 "checkpoint_sha256":self.health_model_sha,
                 "tokenizer_sha256":TOKENIZER_SHA,"device":"cpu","streaming":True,
                 "cancel_supported":True,"kv_cache":True,"max_new_tokens":256,
                 "processed_tokens":2073161216})
    def do_POST(self):
        size=int(self.headers.get("Content-Length","0"));body=json.loads(self.rfile.read(size))
        if self.path=="/cancel":
            type(self).cancelled=body.get("request_id")
            return self._send(200,{"cancelled":True})
        if self.path=="/chat":
            type(self).seen=body
            return self._send(200,{"answer":"Hallo von Arctus","generated_tokens":7,
                                   "finish_reason":"eos","seconds":0.2,"tokens_per_second":35})
        if self.path=="/stream":
            type(self).seen=body
            lines=[
                {"type":"start","request_id":body.get("request_id")},
                {"type":"progress","text":"Hallo","generated_tokens":2,"seconds":0.1,"tokens_per_second":20},
                {"type":"done","result":{"answer":"Hallo Stream","generated_tokens":4,
                 "finish_reason":"eos","seconds":0.2,"tokens_per_second":20}},
            ]
            raw=("".join(json.dumps(x)+"\n" for x in lines)).encode()
            return self._send(200,raw,"application/x-ndjson")
        return self._send(404,{})

@pytest.fixture
def server():
    Handler.health_model_sha=MODEL_SHA;Handler.seen=None;Handler.cancelled=None
    httpd=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    try:yield httpd
    finally:httpd.shutdown();httpd.server_close();thread.join()

def row(port):
    return {"runtime_family":FRONTIER350_RUNTIME,"runtime_port":port,
            "model_sha256":MODEL_SHA,"tokenizer_sha256":TOKENIZER_SHA}

def test_probe_requires_exact_checkpoint_and_tokenizer(server):
    health=probe(row(server.server_port))
    assert health["device"]=="cpu" and health["processed_tokens"]==2073161216
    assert health["kv_cache"] is True and health["streaming"] is True
    Handler.health_model_sha="c"*64
    with pytest.raises(FrontierUnavailable,match="Checkpoint"):probe(row(server.server_port))

def test_chat_uses_local_sidecar_and_clamps_budget(server):
    result,health=chat(row(server.server_port),
        [{"role":"user","content":"Hallo"}],512,request_id="a"*16)
    assert result["answer"]=="Hallo von Arctus" and result["generated_tokens"]==7
    assert health["max_new_tokens"]==256
    assert Handler.seen=={"messages":[{"role":"user","content":"Hallo"}],
                          "max_new_tokens":256,"request_id":"a"*16}

def test_stream_and_cancel_contract(server):
    updates=[]
    result,health=stream_chat(row(server.server_port),
        [{"role":"user","content":"Hallo"}],512,"b"*16,updates.append)
    assert result["answer"]=="Hallo Stream" and result["generated_tokens"]==4
    assert updates and updates[0]["text"]=="Hallo"
    assert health["cancel_supported"] is True
    assert cancel(row(server.server_port),"b"*16) is True
    assert Handler.cancelled=="b"*16

def test_non_frontier_or_unreachable_endpoint_rejected():
    with pytest.raises(FrontierUnavailable):probe({"runtime_family":"legacy-byte-v1","runtime_port":9683})
    with pytest.raises(FrontierUnavailable):probe(row(65534),timeout=0.05)
