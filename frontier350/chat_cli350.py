from __future__ import annotations
import argparse,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from chat350_contract import EXPECTED_TOKENIZER_SHA256
from secure_checkpoint350 import load_checkpoint
from model350 import FRONTIER_350,Neral350M
from chat_prompt350 import prompt_ids,trim_history
from chat350_tools import tool_answer
from chat350_knowledge import KnowledgeIndex,render_knowledge_hits

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(4*1024*1024),b""):h.update(c)
    return h.hexdigest()

def sample_next(logits,generated,temperature,top_p,repetition_penalty):
    z=logits.float().clone()
    if repetition_penalty<1.0:raise ValueError("repetition_penalty must be >=1")
    if repetition_penalty>1.0:
        for tid in set(generated):
            v=z[tid];z[tid]=torch.where(v<0,v*repetition_penalty,v/repetition_penalty)
    if temperature<=0:return int(torch.argmax(z))
    z=z/temperature
    probs=torch.softmax(z,dim=-1)
    if not 0<top_p<=1:raise ValueError("top_p must be in (0,1]")
    if top_p<1:
        sp,si=torch.sort(probs,descending=True);cum=torch.cumsum(sp,dim=-1)
        remove=cum-sp>top_p;sp[remove]=0;sp=sp/sp.sum()
        return int(si[torch.multinomial(sp,1)])
    return int(torch.multinomial(probs,1))

@torch.inference_mode()
def generate(model,tok,history,*,max_new=160,max_context=1024,temperature=.7,top_p=.9,repetition_penalty=1.08,device=None):
    if max_new<=0 or max_new>=max_context:raise ValueError("invalid max_new/max_context")
    h=trim_history(tok,history,max_context-max_new)
    ids=prompt_ids(tok,h);generated=[]
    eos=int(tok.token_to_id("<|eos|>"));bos=int(tok.token_to_id("<|bos|>"));pad=int(tok.token_to_id("<|pad|>"));sep=int(tok.token_to_id("<|sep|>"))
    dev=device or next(model.parameters()).device
    for _ in range(max_new):
        x=torch.tensor([ids],device=dev,dtype=torch.long)
        logits,_=model(x)
        z=logits[0,-1].clone()
        for bad in (bos,pad,sep):z[bad]=-float("inf")
        nxt=sample_next(z,generated,temperature,top_p,repetition_penalty)
        if nxt==eos:break
        generated.append(nxt);ids.append(nxt)
        if len(ids)>=max_context:break
    return tok.decode(generated,skip_special_tokens=True).strip(),h

def load_runtime(checkpoint,tokenizer):
    if sha256_file(tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("locked tokenizer SHA mismatch")
    tok=Tokenizer.from_file(str(tokenizer));ck=load_checkpoint(checkpoint)
    if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:raise ValueError("checkpoint identity mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA required")
    dev=torch.device("cuda");m=Neral350M().to(device=dev,dtype=torch.bfloat16);m.load_state_dict(ck["model"],strict=True);m.eval()
    return m,tok,dev

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--prompt");ap.add_argument("--max-new",type=int,default=160);ap.add_argument("--max-context",type=int,default=1024);ap.add_argument("--temperature",type=float,default=.7);ap.add_argument("--top-p",type=float,default=.9);ap.add_argument("--repetition-penalty",type=float,default=1.08);ap.add_argument("--tool-mode",choices=["auto","off"],default="auto");ap.add_argument("--knowledge-dir");ap.add_argument("--knowledge-limit",type=int,default=2);a=ap.parse_args()
    model,tok,dev=load_runtime(Path(a.checkpoint),Path(a.tokenizer));history=[]
    knowledge=KnowledgeIndex.from_directory(a.knowledge_dir) if a.knowledge_dir else None
    if knowledge is not None:
        print(f"Loaded {knowledge.passage_count} local knowledge passages.",file=__import__("sys").stderr)
    def answer(q):
        nonlocal history
        history.append(("user",q))
        tool=tool_answer(q) if a.tool_mode=="auto" else None
        if tool is not None:
            trimmed=trim_history(tok,history,a.max_context-a.max_new)
            history=trimmed+[("assistant",tool.text)]
            return tool.text,tool.kind
        if knowledge is not None:
            hits=knowledge.search(q,limit=a.knowledge_limit)
            if hits:
                # Retrieved passages are untrusted data, not system instructions.
                # Keep only citation IDs in generation history, not document bodies.
                text=render_knowledge_hits(hits)
                trimmed=trim_history(tok,history,a.max_context-a.max_new)
                references=", ".join(f"{hit.source}:{hit.line}" for hit in hits)
                history=trimmed+[("assistant",f"Lokale Quellen: {references}")]
                return text,"local-knowledge"
        text,trimmed=generate(model,tok,history,max_new=a.max_new,max_context=a.max_context,temperature=a.temperature,top_p=a.top_p,repetition_penalty=a.repetition_penalty,device=dev)
        history=trimmed+[("assistant",text)]
        return text,"native-model"
    if a.prompt is not None:
        text,source=answer(a.prompt)
        print(text)
        print(f"[source={source}]",file=__import__("sys").stderr)
        return
    print("Arctus 350M local chat. /reset clears history, /quit exits.")
    while True:
        try:q=input("You> ").strip()
        except (EOFError,KeyboardInterrupt):print();break
        if not q:continue
        if q=="/quit":break
        if q=="/reset":history=[];print("History cleared.");continue
        text,source=answer(q)
        print(f"Neral ({source})> {text}")
if __name__=="__main__":main()
