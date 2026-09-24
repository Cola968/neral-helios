from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,threading

import pytest

from frontier_sidecar import FrontierUnavailable,chat,probe
from model_catalog import FRONTIER350_RUNTIME

MODEL_SHA="a"*64
TOKENIZER_SHA="b"*64

class Handler(BaseHTTPRequestHandler):
    health_model_sha=MODEL_SHA
    seen=None
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path!="/health":self.send_response(404);self.end_headers();return
        payload={"status":"experimental-ready","runtime_family":FRONTIER350_RUNTIME,
                 "checkpoint_sha256":self.health_model_sha,
                 "tokenizer_sha256":TOKENIZER_SHA,"device":"cpu",
                 "max_new_tokens":64,"processed_tokens":2073161216}
        raw=json.dumps(payload).encode()
        self.send_response(200);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        if self.path!="/chat":self.send_response(404);self.end_headers();return
        size=int(self.headers.get("Content-Length","0"));body=json.loads(self.rfile.read(size))
        type(self).seen=body
        raw=json.dumps({"answer":"Hallo von Arctus","generated_tokens":7,
                        "finish_reason":"eos"}).encode()
        self.send_response(200);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)

@pytest.fixture
def server():
    Handler.health_model_sha=MODEL_SHA;Handler.seen=None
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
    Handler.health_model_sha="c"*64
    with pytest.raises(FrontierUnavailable,match="Checkpoint"):probe(row(server.server_port))

def test_chat_uses_local_sidecar_and_clamps_budget(server):
    result,health=chat(row(server.server_port),
        [{"role":"user","content":"Hallo"}],256)
    assert result["answer"]=="Hallo von Arctus" and result["generated_tokens"]==7
    assert health["max_new_tokens"]==64
    assert Handler.seen=={"messages":[{"role":"user","content":"Hallo"}],"max_new_tokens":64}

def test_non_frontier_or_unreachable_endpoint_rejected():
    with pytest.raises(FrontierUnavailable):probe({"runtime_family":"legacy-byte-v1","runtime_port":9683})
    with pytest.raises(FrontierUnavailable):probe(row(65534),timeout=0.05)
