from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chat350_contract import EXPECTED_TOKENIZER_SHA256
from chat350_generation_metrics import summarize_generations
from model350 import FRONTIER_350, Neral350M

def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(4*1024*1024),b""): h.update(c)
    return h.hexdigest()

def load_cases(path: Path):
    obj=json.loads(path.read_text(encoding="utf8"))
    if not isinstance(obj,list) or not obj: raise ValueError("cases must be a non-empty JSON list")
    ids=set()
    for row in obj:
        if row.get("id") in ids: raise ValueError("duplicate case id")
        ids.add(row.get("id"))
        if not row.get("prefix") or int(row.get("max_new_tokens",0))<=0: raise ValueError("invalid generation case")
    return obj

@torch.inference_mode()
def generate(model,tok,prefix,max_new,device):
    bos=int(tok.token_to_id("<|bos|>")); eos=int(tok.token_to_id("<|eos|>"))
    ids=[bos]+list(tok.encode(prefix).ids); generated=[]
    for _ in range(max_new):
        if len(ids)>=FRONTIER_350.max_seq_len: break
        x=torch.tensor([ids],device=device,dtype=torch.long)
        logits,_=model(x); nxt=int(logits[0,-1].argmax())
        generated.append(nxt); ids.append(nxt)
        if nxt==eos: break
    content=generated[:-1] if generated and generated[-1]==eos else generated
    return generated,tok.decode(content,skip_special_tokens=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--cases",required=True); ap.add_argument("--out",required=True)
    args=ap.parse_args()
    token_path=Path(args.tokenizer)
    if sha256_file(token_path)!=EXPECTED_TOKENIZER_SHA256: raise ValueError("tokenizer digest mismatch")
    tok=Tokenizer.from_file(str(token_path)); cases=load_cases(Path(args.cases))
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if ck.get("config")!=FRONTIER_350.to_dict(): raise ValueError("checkpoint architecture mismatch")
    if ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256: raise ValueError("checkpoint/tokenizer mismatch")
    state=ck.get("model")
    if not isinstance(state,dict): raise ValueError("checkpoint has no model state")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")
    device=torch.device("cuda"); model=Neral350M().to(device=device,dtype=torch.bfloat16); model.load_state_dict(state,strict=True); model.eval()
    rows=[]
    for case in cases:
        ids,text=generate(model,tok,case["prefix"],int(case["max_new_tokens"]),device)
        row={"id":case["id"],"category":case.get("category","unknown"),"generated_ids":ids,"text":text}
        if "expected_exact" in case: row["expected_exact"]=case["expected_exact"]
        rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    summary=summarize_generations(rows,int(tok.token_to_id("<|eos|>")))
    case_sha=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
    report={"checkpoint_sha256":sha256_file(Path(args.checkpoint)),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
            "case_sha256":case_sha,"summary":summary,"rows":rows}
    Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({"summary":summary,"case_sha256":case_sha},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
