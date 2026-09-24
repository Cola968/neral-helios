"""Local Helios sprint chatbot for a SHA-verified sprint72h checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from nextgen_v1.helios_chat_core import generate_turn

TOKENIZER_SHA="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
MAX_BODY=20_480


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser()
 p.add_argument("--execute",action="store_true")
 p.add_argument("--checkpoint",type=Path,required=True)
 p.add_argument("--checkpoint-sha256",required=True)
 p.add_argument("--tokenizer",type=Path,required=True)
 p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
 p.add_argument("--port",type=int,default=9683)
 a=p.parse_args()
 if not a.execute:p.error("--execute required")
 if sha(a.checkpoint)!=a.checkpoint_sha256.lower():raise ValueError("checkpoint SHA mismatch")
 if sha(a.tokenizer)!=TOKENIZER_SHA:raise ValueError("tokenizer SHA mismatch")
 if not 1024<=a.port<=65535:p.error("port must be 1024..65535")

 import torch
 from tokenizers import Tokenizer
 from model350 import FRONTIER_350,Neral350M
 state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
 if state.get("checkpoint_schema")!="helios_chat_sprint72h_v1":
  raise ValueError("not a sprint72h chat checkpoint")
 if state.get("config")!=FRONTIER_350.to_dict() or state.get("tokenizer_sha256")!=TOKENIZER_SHA:
  raise ValueError("checkpoint model/tokenizer contract mismatch")

 if a.device=="auto":
  device="cuda" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "cpu"
 else:device=a.device
 if device=="cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
  raise RuntimeError("CUDA/BF16 unavailable")
 dtype=torch.bfloat16 if device=="cuda" else torch.float32
 model=Neral350M().to(device=device,dtype=dtype)
 model.load_state_dict(state["model"],strict=True);model.eval()
 tok=Tokenizer.from_file(str(a.tokenizer));bos=tok.token_to_id("<|bos|>");eos=tok.token_to_id("<|eos|>")
 if bos is None or eos is None:raise ValueError("special tokens missing")
 ui=(HERE/"nextgen_v1"/"helios_chat_ui.html").read_bytes()
 mutex=threading.Lock()

 class Handler(BaseHTTPRequestHandler):
  server_version="HeliosSprint/1"
  def _send(self,status,body,ctype):
   self.send_response(status);self.send_header("Content-Type",ctype)
   self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store")
   self.end_headers();self.wfile.write(body)
  def _json(self,status,obj):
   self._send(status,json.dumps(obj,ensure_ascii=False).encode(),"application/json; charset=utf-8")
  def _local(self):
   return self.client_address[0] in {"127.0.0.1","::1"}
  def do_GET(self):
   if not self._local():return self._json(403,{"error":"localhost only"})
   if self.path=="/":return self._send(200,ui,"text/html; charset=utf-8")
   if self.path=="/status":return self._json(200,{"model":"helios-sprint72h","device":device,
      "checkpoint_sha256":a.checkpoint_sha256.lower(),"quality_verified":False})
   return self._json(404,{"error":"not found"})
  def do_POST(self):
   if not self._local():return self._json(403,{"error":"localhost only"})
   if self.path!="/chat":return self._json(404,{"error":"not found"})
   try:size=int(self.headers.get("Content-Length","0"))
   except ValueError:return self._json(400,{"error":"invalid content length"})
   if not 1<=size<=MAX_BODY:return self._json(413,{"error":"request too large"})
   try:
    obj=json.loads(self.rfile.read(size).decode("utf8"))
    messages=obj["messages"];budget=obj.get("max_new_tokens",64)
    def next_token(ids,emitted):
     x=torch.tensor([ids],device=device,dtype=torch.long)
     logits,_=model(x);scores=logits[0,-1].float().clone()
     for token in set(emitted[-32:]):
      if token!=eos:scores[token]-=0.35
     return int(scores.argmax().item())
    with mutex,torch.inference_mode():
     out=generate_turn(tok,bos,eos,messages,budget,next_token)
    return self._json(200,{**out,"model":"helios-sprint72h","device":device,
                           "quality_verified":False,"history_stored_on_server":False})
   except (ValueError,KeyError,UnicodeError,json.JSONDecodeError) as exc:
    return self._json(400,{"error":str(exc)})
   except Exception as exc:
    return self._json(500,{"error":type(exc).__name__})
  def log_message(self,fmt,*args):
   return

 server=ThreadingHTTPServer(("127.0.0.1",a.port),Handler)
 print(json.dumps({"event":"helios_sprint72h_chat_ready","url":f"http://127.0.0.1:{a.port}/",
                   "device":device,"checkpoint_sha256":a.checkpoint_sha256.lower()}),flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()


if __name__=="__main__":main()
