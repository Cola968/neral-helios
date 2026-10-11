from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from chat350_contract import EXPECTED_TOKENIZER_SHA256
from chat350_preference_format import encode_final_response
from model350 import FRONTIER_350,Neral350M

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(4*1024*1024),b""):h.update(c)
    return h.hexdigest()

def response_logp(model,e,device):
    x=torch.tensor([e.input_ids],dtype=torch.long,device=device);y=torch.tensor(e.targets,dtype=torch.long,device=device)
    logits,_=model(x);mask=y!=-100
    lp=F.log_softmax(logits[0].float(),dim=-1)
    idx=torch.arange(y.numel(),device=device)[mask]
    return float(lp[idx,y[mask]].sum()),int(mask.sum())

def load_rows(path):
    out=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf8").splitlines(),1):
        if not line.strip():continue
        r=json.loads(line)
        for k in ("id","context","chosen","rejected","pair_sha256"):
            if k not in r:raise ValueError(f"{path}:{n}: missing {k}")
        out.append(r)
    if not out:raise ValueError("empty preference split")
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--preferences",required=True);ap.add_argument("--out",required=True);ap.add_argument("--max-seq",type=int,default=768);a=ap.parse_args()
    if sha256_file(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("tokenizer SHA mismatch")
    tok=Tokenizer.from_file(a.tokenizer);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:raise ValueError("checkpoint identity mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA required")
    dev=torch.device("cuda");m=Neral350M().to(device=dev,dtype=torch.bfloat16);m.load_state_dict(ck["model"],strict=True);m.eval()
    rows=load_rows(a.preferences);results=[]
    with torch.inference_mode():
        for i,r in enumerate(rows,1):
            c=encode_final_response(tok,r["context"],r["chosen"],max_seq_len=a.max_seq)
            j=encode_final_response(tok,r["context"],r["rejected"],max_seq_len=a.max_seq)
            cl,cn=response_logp(m,c,dev);rl,rn=response_logp(m,j,dev)
            results.append({"id":r["id"],"pair_sha256":r["pair_sha256"],"chosen_logp":cl,"rejected_logp":rl,"chosen_tokens":cn,"rejected_tokens":rn})
            if i%100==0:print(json.dumps({"done":i,"total":len(rows)}),flush=True)
    payload={"schema":1,"checkpoint_sha256":sha256_file(a.checkpoint),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"preference_sha256":sha256_file(a.preferences),"rows":results}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":")).encode("utf8");payload["payload_sha256"]=hashlib.sha256(raw).hexdigest()
    Path(a.out).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"rows":len(results),"payload_sha256":payload["payload_sha256"]}),flush=True)
if __name__=="__main__":main()
