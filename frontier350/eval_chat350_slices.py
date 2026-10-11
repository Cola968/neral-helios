from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
import torch
from chat350_contract import EXPECTED_TOKENIZER_SHA256
from chat350_data import validate_dataset_root
from chat350_sft_format import encode_messages
from model350 import FRONTIER_350,Neral350M
from train_chat350 import load_tokenizer,eval_sft

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(4*1024*1024),b""):h.update(c)
    return h.hexdigest()

def group_rows(rows):
    groups={"overall":rows}
    def add(name,row):groups.setdefault(name,[]).append(row)
    for r in rows:
        add("language:"+r["language"],r)
        add("domain:"+r["domain"],r)
        add("source:"+r["source"],r)
        add("turns:"+("multi" if len(r["messages"])>=4 else "single"),r)
    return dict(sorted(groups.items()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--data",required=True);ap.add_argument("--split",choices=["val","test"],default="val");ap.add_argument("--out",required=True);ap.add_argument("--max-seq",type=int,default=512);ap.add_argument("--batch-size",type=int,default=4);a=ap.parse_args()
    manifest,splits=validate_dataset_root(Path(a.data));rows=splits[a.split]
    tok=load_tokenizer(Path(a.tokenizer));ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if ck.get("config")!=FRONTIER_350.to_dict() or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256:raise ValueError("checkpoint identity mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA required")
    dev=torch.device("cuda");m=Neral350M().to(device=dev,dtype=torch.bfloat16);m.load_state_dict(ck["model"],strict=True);m.eval()
    pad=int(tok.token_to_id("<|pad|>"));eos=int(tok.token_to_id("<|eos|>"));report={}
    for name,rs in group_rows(rows).items():
        encoded=[];skipped=0
        for r in rs:
            try:encoded.append(encode_messages(tok,r["messages"],max_seq_len=a.max_seq))
            except ValueError:skipped+=1
        if not encoded:continue
        ev=eval_sft(m,encoded,a.batch_size,pad,eos,dev)
        report[name]={"rows":len(encoded),"skipped_too_long":skipped,**ev}
    out={"schema":1,"checkpoint_sha256":sha256_file(a.checkpoint),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"dataset_manifest_sha256":manifest["manifest_sha256"],"split":a.split,"max_seq":a.max_seq,"slices":report}
    Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({"split":a.split,"slices":len(report),"overall":report.get("overall")},ensure_ascii=False))
if __name__=="__main__":main()
