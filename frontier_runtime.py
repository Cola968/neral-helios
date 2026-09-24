"""Inference-only local runtime for registered own Neral/Arctus Frontier 350M bundles.

Features: exact SHA-bound bundle loading, CUDA auto-selection, BF16 CUDA inference,
incremental KV-cache decoding, NDJSON streaming, cancellation and runtime metrics.
No training, downloads, checkpoint writes or quality promotion.
"""
from __future__ import annotations

import argparse,hashlib,json,re,threading,time
from contextlib import nullcontext
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from frontier350_contract import (
    FRONTIER350_CONFIG,FRONTIER350_PARAMETERS,FRONTIER350_RUNTIME,
    FRONTIER350_TOKENIZER_SHA256,expected_state_shapes,
)

MAX_BODY_BYTES=64*1024
MAX_NEW_TOKENS=256
MAX_MESSAGES=13
MAX_MESSAGE_CHARS=4096
ROLE_BOUNDARY=re.compile(r"(?im)(?:^|\n)\s*(?:user|assistant|system|nutzer|benutzer)\s*:")

def sha256_file(path:Path)->str:
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def read_sha(path:Path)->str:
    parts=path.read_text(encoding="ascii").split()
    value=parts[0].lower() if parts else ""
    if not re.fullmatch(r"[a-f0-9]{64}",value):raise ValueError("invalid SHA-256 file")
    return value

def verify_bundle(bundle:Path):
    bundle=bundle.resolve(strict=True)
    model=bundle/"model.pt";model_sha=bundle/"model.sha256"
    tokenizer=bundle/"tokenizer.json";tokenizer_sha=bundle/"tokenizer.sha256"
    for p in (model,model_sha,tokenizer,tokenizer_sha):
        if p.is_symlink() or not p.is_file() or p.resolve().parent!=bundle:
            raise ValueError("incomplete or unsafe Frontier bundle")
    expected_model=read_sha(model_sha);expected_tok=read_sha(tokenizer_sha)
    if sha256_file(model)!=expected_model:raise ValueError("Frontier checkpoint SHA mismatch")
    if sha256_file(tokenizer)!=expected_tok or expected_tok!=FRONTIER350_TOKENIZER_SHA256:
        raise ValueError("Frontier tokenizer SHA mismatch")
    return model,tokenizer,expected_model,expected_tok

@dataclass(frozen=True)
class Config:
    vocab_size:int=FRONTIER350_CONFIG["vocab_size"]
    d_model:int=FRONTIER350_CONFIG["d_model"]
    n_layers:int=FRONTIER350_CONFIG["n_layers"]
    n_heads:int=FRONTIER350_CONFIG["n_heads"]
    n_kv_heads:int=FRONTIER350_CONFIG["n_kv_heads"]
    ffn_hidden:int=FRONTIER350_CONFIG["ffn_hidden"]
    max_seq_len:int=FRONTIER350_CONFIG["max_seq_len"]
    rope_theta:float=FRONTIER350_CONFIG["rope_theta"]
    rms_eps:float=FRONTIER350_CONFIG["rms_eps"]
    qk_rms_eps:float=FRONTIER350_CONFIG["qk_rms_eps"]
    @property
    def head_dim(self):return self.d_model//self.n_heads
    @property
    def kv_dim(self):return self.n_kv_heads*self.head_dim
    @property
    def groups(self):return self.n_heads//self.n_kv_heads

def build_model(torch):
    nn=torch.nn;F=torch.nn.functional;cfg=Config()
    class RMSNorm(nn.Module):
        def __init__(self,dim,eps):super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
        def forward(self,x):
            xf=x.float();y=xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+self.eps)
            return y.to(x.dtype)*self.weight.to(x.dtype)
    class Attention(nn.Module):
        def __init__(self):
            super().__init__();d=cfg.d_model
            self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,cfg.kv_dim,bias=False)
            self.v=nn.Linear(d,cfg.kv_dim,bias=False);self.o=nn.Linear(d,d,bias=False)
            self.qn=RMSNorm(cfg.head_dim,cfg.qk_rms_eps);self.kn=RMSNorm(cfg.head_dim,cfg.qk_rms_eps)
        def _rope(self,q,k,offset):
            t=q.size(-2);pos=torch.arange(offset,offset+t,device=q.device,dtype=torch.float32)
            inv=1.0/(cfg.rope_theta**(torch.arange(0,cfg.head_dim,2,device=q.device,dtype=torch.float32)/cfg.head_dim))
            phase=torch.outer(pos,inv).repeat_interleave(2,-1)[None,None]
            c=phase.cos().to(q.dtype);s=phase.sin().to(q.dtype)
            def rotate(x):
                a,b=x[...,::2],x[...,1::2]
                return torch.stack((-b,a),dim=-1).flatten(-2)
            return q*c+rotate(q)*s,k*c+rotate(k)*s
        def forward(self,x,past=None):
            b,t,_=x.shape
            if past is None:past_len=0
            elif len(past)==3:past_len=int(past[2])
            else:past_len=past[0].size(-2)
            q=self.q(x).view(b,t,cfg.n_heads,cfg.head_dim).transpose(1,2)
            k_new=self.k(x).view(b,t,cfg.n_kv_heads,cfg.head_dim).transpose(1,2)
            v_new=self.v(x).view(b,t,cfg.n_kv_heads,cfg.head_dim).transpose(1,2)
            q,k_new=self.qn(q),self.kn(k_new);q,k_new=self._rope(q,k_new,past_len)
            if past is None:
                k,v=k_new,v_new;next_cache=(k,v)
            elif len(past)==3:
                if t!=1:raise ValueError("cached decode accepts exactly one new token")
                k_buf,v_buf,used=past;end=int(used)+1
                if end>k_buf.size(-2):raise ValueError("KV cache capacity exceeded")
                k_buf[...,int(used):end,:].copy_(k_new);v_buf[...,int(used):end,:].copy_(v_new)
                k,v=k_buf[...,:end,:],v_buf[...,:end,:];next_cache=(k_buf,v_buf,end)
            else:
                if t!=1:raise ValueError("cached decode accepts exactly one new token")
                k=torch.cat((past[0],k_new),dim=-2);v=torch.cat((past[1],v_new),dim=-2);next_cache=(k,v)
            try:y=F.scaled_dot_product_attention(q,k,v,is_causal=past is None,dropout_p=0.0,enable_gqa=True)
            except (TypeError,RuntimeError):
                kk=k.repeat_interleave(cfg.groups,1);vv=v.repeat_interleave(cfg.groups,1)
                y=F.scaled_dot_product_attention(q,kk,vv,is_causal=past is None,dropout_p=0.0)
            out=self.o(y.transpose(1,2).contiguous().view(b,t,cfg.d_model))
            return out,next_cache
    class FFN(nn.Module):
        def __init__(self):
            super().__init__();self.gate=nn.Linear(cfg.d_model,cfg.ffn_hidden,bias=False)
            self.up=nn.Linear(cfg.d_model,cfg.ffn_hidden,bias=False);self.down=nn.Linear(cfg.ffn_hidden,cfg.d_model,bias=False)
        def forward(self,x):return self.down(F.silu(self.gate(x))*self.up(x))
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.n1=RMSNorm(cfg.d_model,cfg.rms_eps);self.attn=Attention();self.n2=RMSNorm(cfg.d_model,cfg.rms_eps);self.ffn=FFN()
        def forward(self,x,past=None):
            a,kv=self.attn(self.n1(x),past);x=x+a;return x+self.ffn(self.n2(x)),kv
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
            self.blocks=nn.ModuleList([Block() for _ in range(cfg.n_layers)])
            self.norm=RMSNorm(cfg.d_model,cfg.rms_eps);self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
            self.lm_head.weight=self.tok_emb.weight
        def forward(self,idx,cache=None):
            if idx.ndim!=2 or idx.size(1)<1:raise ValueError("invalid input")
            if cache is not None and len(cache)!=cfg.n_layers:raise ValueError("invalid KV cache")
            if cache is None:past_len=0
            elif len(cache[0])==3:past_len=int(cache[0][2])
            else:past_len=cache[0][0].size(-2)
            if past_len+idx.size(1)>cfg.max_seq_len:raise ValueError("context exceeds model maximum")
            x=self.tok_emb(idx);new=[]
            for i,block in enumerate(self.blocks):
                x,kv=block(x,None if cache is None else cache[i]);new.append(kv)
            return self.lm_head(self.norm(x)),new
    return Model(),cfg

def load_bundle(bundle:Path,device_request:str):
    model_path,tok_path,model_sha,tok_sha=verify_bundle(bundle)
    import torch
    try:from tokenizers import Tokenizer
    except ImportError as exc:raise RuntimeError("Python package 'tokenizers' is required") from exc
    if device_request=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA requested but unavailable")
    use_cuda=(device_request=="cuda") or (device_request=="auto" and torch.cuda.is_available())
    device=torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and not torch.cuda.is_bf16_supported():raise ValueError("CUDA BF16 is required for Frontier 350")
    payload=torch.load(model_path,map_location="cpu",weights_only=True)
    if payload.get("config")!=FRONTIER350_CONFIG:raise ValueError("Frontier config mismatch")
    if payload.get("tokenizer_sha256")!=FRONTIER350_TOKENIZER_SHA256:raise ValueError("checkpoint tokenizer mismatch")
    state=payload.get("model");shapes=expected_state_shapes()
    if not isinstance(state,dict) or set(state)!=set(shapes):raise ValueError("Frontier state keys mismatch")
    model,cfg=build_model(torch);model.load_state_dict(state,strict=True)
    if sum(p.numel() for p in model.parameters())!=FRONTIER350_PARAMETERS:raise ValueError("Frontier parameter mismatch")
    dtype=torch.bfloat16 if use_cuda else torch.float32
    model=model.to(device=device,dtype=dtype).eval()
    tok=Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size()!=cfg.vocab_size:raise ValueError("tokenizer vocabulary mismatch")
    bos,eos=tok.token_to_id("<|bos|>"),tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos==eos:raise ValueError("Frontier BOS/EOS missing")
    processed=payload.get("processed_tokens")
    if type(processed) is not int:processed=None
    del state,payload
    if use_cuda:torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    return dict(model=model,tokenizer=tok,bos=int(bos),eos=int(eos),torch=torch,device=device,
                cfg=cfg,checkpoint_sha=model_sha,tokenizer_sha=tok_sha,processed_tokens=processed)

def validate_messages(messages,budget):
    if type(budget) is not int or not 1<=budget<=MAX_NEW_TOKENS:raise ValueError("max_new_tokens out of range")
    if not isinstance(messages,list) or not 1<=len(messages)<=MAX_MESSAGES:raise ValueError("messages must contain 1 to 13 turns")
    clean=[]
    for i,item in enumerate(messages):
        role="user" if i%2==0 else "assistant"
        if not isinstance(item,dict) or item.get("role")!=role or set(item)!={"role","content"}:raise ValueError("messages must alternate user/assistant")
        text=item.get("content")
        if not isinstance(text,str) or not text.strip() or len(text)>MAX_MESSAGE_CHARS:raise ValueError("invalid message")
        clean.append({"role":role,"content":text.strip()})
    if len(clean)%2!=1:raise ValueError("last message must be user")
    return clean

def encode_context(rt,messages,budget,context_limit):
    clean=validate_messages(messages,budget);dropped=0
    while True:
        dialogue="".join(("User: " if m["role"]=="user" else "Assistant: ")+m["content"]+"\n" for m in clean)+"Assistant:"
        ids=[rt["bos"]]+list(rt["tokenizer"].encode(dialogue).ids)
        if len(ids)+budget<=context_limit:return ids,dropped
        if len(clean)==1:raise ValueError("latest message exceeds available context")
        clean=clean[2:];dropped+=1

def reserve_cache(cache,capacity,torch):
    reserved=[]
    for k,v in cache:
        used=k.size(-2)
        if used>capacity:raise ValueError("KV cache prefill exceeds capacity")
        shape=(*k.shape[:-2],capacity,k.shape[-1])
        kb=torch.empty(shape,device=k.device,dtype=k.dtype);vb=torch.empty(shape,device=v.device,dtype=v.dtype)
        kb[...,:used,:].copy_(k);vb[...,:used,:].copy_(v)
        reserved.append((kb,vb,used))
    return reserved

def _looping(text):
    words=re.findall(r"\w+",text.casefold())
    if len(words)<15:return False
    grams=[tuple(words[i:i+3]) for i in range(len(words)-2)]
    return grams.count(grams[-1])>=3 and (len(grams)-len(set(grams)))/max(1,len(grams))>=0.28

def _autocast(torch,device):
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else nullcontext()

def generate(rt,messages,budget,context_limit,cancel,on_progress=None):
    torch=rt["torch"];model=rt["model"];tok=rt["tokenizer"];eos=rt["eos"];device=rt["device"]
    ids,dropped=encode_context(rt,messages,budget,context_limit)
    start=time.perf_counter();emitted=[];finish="max_tokens";cache=None
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        x=torch.tensor([ids],device=device,dtype=torch.long)
        with _autocast(torch,device):
            logits,cache=model(x,None)
        cache=reserve_cache(cache,context_limit,torch)
        for step in range(1,budget+1):
            if cancel.is_set():finish="cancelled";break
            token=int(logits[0,-1].float().argmax().item())
            if token==eos:finish="eos";break
            emitted.append(token)
            text=tok.decode(emitted,skip_special_tokens=True)
            boundary=ROLE_BOUNDARY.search(text)
            if boundary:text=text[:boundary.start()].rstrip();finish="role_boundary"
            elif _looping(text):finish="repetition_guard"
            elapsed=time.perf_counter()-start
            if on_progress:on_progress({"text":text,"generated_tokens":len(emitted),"seconds":round(elapsed,4),
                                        "tokens_per_second":round(len(emitted)/max(elapsed,1e-9),2)})
            if finish in {"role_boundary","repetition_guard"}:break
            if step<budget:
                x=torch.tensor([[token]],device=device,dtype=torch.long)
                with _autocast(torch,device):
                    logits,cache=model(x,cache)
    elapsed=time.perf_counter()-start
    text=tok.decode(emitted,skip_special_tokens=True)
    boundary=ROLE_BOUNDARY.search(text)
    if boundary:text=text[:boundary.start()].rstrip()
    peak=0
    allocated=0
    if device.type=="cuda":
        peak=torch.cuda.max_memory_allocated(device);allocated=torch.cuda.memory_allocated(device)
    return {"answer":text,"generated_tokens":len(emitted),"finish_reason":finish,
            "dropped_history_pairs":dropped,"input_tokens":len(ids),"seconds":round(elapsed,4),
            "tokens_per_second":round(len(emitted)/max(elapsed,1e-9),2),
            "peak_vram_mib":round(peak/1048576,1),"vram_allocated_mib":round(allocated/1048576,1)}

def runtime_health(rt,context_limit,busy=False):
    device=rt["device"];torch=rt["torch"];gpu=None;allocated=reserved=0
    if device.type=="cuda":
        gpu=torch.cuda.get_device_name(device);allocated=torch.cuda.memory_allocated(device);reserved=torch.cuda.memory_reserved(device)
    return {"status":"experimental-ready","runtime_family":FRONTIER350_RUNTIME,
            "checkpoint_sha256":rt["checkpoint_sha"],"tokenizer_sha256":rt["tokenizer_sha"],
            "processed_tokens":rt["processed_tokens"],"device":device.type,"gpu":gpu,
            "parameters":FRONTIER350_PARAMETERS,"context_tokens":context_limit,
            "max_new_tokens":MAX_NEW_TOKENS,"decoder":"greedy-preallocated-kv-v1",
            "streaming":True,"cancel_supported":True,"kv_cache":True,"busy":busy,
            "vram_allocated_mib":round(allocated/1048576,1),"vram_reserved_mib":round(reserved/1048576,1)}

def make_handler(rt,context_limit):
    mutex=threading.Lock();jobs={};jobs_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        server_version="HeliosFrontier/1"
        def log_message(self,*_):pass
        def _allowed(self):return self.client_address[0] in {"127.0.0.1","::1"} and self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
        def _json(self,status,obj):
            raw=json.dumps(obj,ensure_ascii=False,separators=(",",":")).encode("utf-8")
            self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(raw)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(raw)
        def _body(self):
            try:size=int(self.headers.get("Content-Length","0"))
            except ValueError:raise ValueError("invalid content length")
            if not 1<=size<=MAX_BODY_BYTES:raise ValueError("request body out of range")
            try:data=json.loads(self.rfile.read(size).decode("utf-8"))
            except (UnicodeError,json.JSONDecodeError) as exc:raise ValueError("valid UTF-8 JSON required") from exc
            if not isinstance(data,dict):raise ValueError("JSON object required")
            return data
        def do_GET(self):
            if not self._allowed():return self._json(403,{"error":"localhost only"})
            if self.path=="/health":return self._json(200,runtime_health(rt,context_limit,mutex.locked()))
            return self._json(404,{"error":"not found"})
        def do_POST(self):
            if not self._allowed():return self._json(403,{"error":"localhost only"})
            if self.headers.get("Content-Type","").split(";",1)[0].strip().lower()!="application/json":return self._json(415,{"error":"application/json required"})
            try:data=self._body()
            except ValueError as exc:return self._json(400,{"error":str(exc)})
            if self.path=="/cancel":
                rid=data.get("request_id")
                with jobs_lock:event=jobs.get(rid) if isinstance(rid,str) else None
                if event:event.set()
                return self._json(200,{"cancelled":event is not None})
            if self.path not in {"/chat","/stream"}:return self._json(404,{"error":"not found"})
            messages=data.get("messages");budget=data.get("max_new_tokens");rid=data.get("request_id")
            if not isinstance(rid,str) or not re.fullmatch(r"[A-Za-z0-9-]{16,64}",rid):rid=hashlib.sha256(f"{time.monotonic_ns()}-{threading.get_ident()}".encode()).hexdigest()[:32]
            event=threading.Event()
            if not mutex.acquire(blocking=False):return self._json(409,{"error":"runtime busy"})
            with jobs_lock:jobs[rid]=event
            def finish():
                with jobs_lock:jobs.pop(rid,None)
                mutex.release()
            try:
                if self.path=="/chat":
                    result=generate(rt,messages,budget,context_limit,event)
                    result.update({"request_id":rid,"runtime_family":FRONTIER350_RUNTIME})
                    return self._json(200,result)
                self.send_response(200);self.send_header("Content-Type","application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control","no-store");self.send_header("X-Accel-Buffering","no");self.end_headers()
                def emit(obj):
                    raw=(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
                    self.wfile.write(raw);self.wfile.flush()
                emit({"type":"start","request_id":rid})
                try:
                    result=generate(rt,messages,budget,context_limit,event,lambda p:emit({"type":"progress",**p}))
                    emit({"type":"done","result":{**result,"request_id":rid,"runtime_family":FRONTIER350_RUNTIME}})
                except (BrokenPipeError,ConnectionResetError):
                    event.set()
                return
            except ValueError as exc:
                if self.path=="/stream":
                    try:
                        self.wfile.write((json.dumps({"type":"error","error":str(exc)})+"\n").encode());self.wfile.flush()
                    except OSError:pass
                    return
                return self._json(400,{"error":str(exc)})
            except Exception:
                if self.path=="/stream":
                    try:self.wfile.write(b'{"type":"error","error":"local Frontier inference failed"}\n');self.wfile.flush()
                    except OSError:pass
                    return
                return self._json(500,{"error":"local Frontier inference failed"})
            finally:finish()
    return Handler

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--port",type=int,default=9683)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--context-tokens",type=int,default=512)
    a=p.parse_args(argv)
    if not 1024<=a.port<=65535:raise SystemExit("port must be 1024..65535")
    if not 64<=a.context_tokens<=FRONTIER350_CONFIG["max_seq_len"]:raise SystemExit("invalid context-tokens")
    rt=load_bundle(a.bundle,a.device)
    health=runtime_health(rt,a.context_tokens)
    print(json.dumps({"event":"frontier_runtime_ready",**health},ensure_ascii=False),flush=True)
    server=ThreadingHTTPServer(("127.0.0.1",a.port),make_handler(rt,a.context_tokens))
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":main()
