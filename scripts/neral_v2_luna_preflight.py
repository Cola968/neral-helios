"""CPU-only full native Qwen3.5-4B chat-label preflight on Docker-proven split."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.neral_v2_full_train import load_splits,to_messages
from scripts.neral_v2_luna_encoding import assistant_only_4b
from scripts.neral_v2_luna_checkpoint import complete_model

def main():
    ap=argparse.ArgumentParser()
    for name in ("train","val","manifest","docker-proof","checkpoint"):
        ap.add_argument("--"+name,required=True,type=Path)
    args=ap.parse_args()
    if not complete_model(args.checkpoint):
        raise RuntimeError("Luna original 4B full Safetensors missing")
    train,val,_=load_splits(args.train,args.val,args.manifest,args.docker_proof,"luna")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.checkpoint),use_fast=True,local_files_only=True)
    a=assistant_only_4b(to_messages(train),tokenizer,max_seq_length=2048)
    b=assistant_only_4b(to_messages(val),tokenizer,max_seq_length=2048)
    if len(a)!=480 or len(b)!=75:raise RuntimeError("Luna split dropped cases")
    for item in a+b:
        labels=item["labels"]
        if len(labels)!=len(item["input_ids"]) or not any(x>=0 for x in labels):
            raise RuntimeError("Luna assistant-only label mask incomplete")
        supervised=next(i for i,x in enumerate(labels) if x!=-100)
        if supervised<1 or any(x!=-100 for x in labels[:supervised]):
            raise RuntimeError("Luna prompt supervision leak")
    print("NERAL_LUNA_4B_NATIVE_TEMPLATE_PREFLIGHT "+json.dumps({
      "status":"pass","all_555_assistant_masks_verified":True,
      "train":len(a),"validation":len(b),
      "max_tokens":max(len(x["input_ids"]) for x in a+b),
      "min_answer_tokens":min(sum(v>=0 for v in x["labels"]) for x in a+b),
      "base_is_local_4b_safetensors":True},sort_keys=True),flush=True)

if __name__=="__main__":main()
