"""One-time frozen 40-case direct adapter evaluation on the same HF Qwen3.5-9B.

The test split is first accessed only AFTER proof of complete 1100/140 Docker
gated training and the rank16 v2-full adapter bytes are frozen.
No retraining, test-based checkpoint selection, or raw answer log publication.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from scripts.neral_qlora_v2_compare40 import run_model,TEST_SHA,_digest

def validate_frozen(full:Path,v1:Path,output:Path):
    if output.exists() or output.is_symlink():raise FileExistsError("Refuse to overwrite evaluation")
    receipt=json.loads((full/"manifest.json").read_text(encoding="utf8"))
    if (receipt.get("status")!="frozen_adapter_ready_for_heldout_comparison"
        or receipt.get("family")!="helios"
        or receipt.get("rank")!=16
        or receipt.get("train_rows")!=1100
        or receipt.get("val_rows")!=140
        or receipt.get("test_rows_opened")!=0
        or receipt.get("docker_verified") is not True
        or receipt.get("epochs")!=2
        or receipt.get("selected_epoch") not in (1,2)
        or receipt.get("max_steps")!=0):
        raise RuntimeError("Full 9B rank16 training not completed or test leaked")
    adapter=full/"adapter"
    if not (adapter/"adapter_config.json").is_file() or not (v1/"adapter_config.json").is_file():
        raise FileNotFoundError("Frozen 9B PEFT adapter(s) missing")
    if hashlib.sha256((adapter/"adapter_model.safetensors").read_bytes()).hexdigest()!=receipt.get("frozen_adapter_sha256"):
        raise RuntimeError("Adapter model bytes drifted after training")
    return receipt

def load40(test:Path,full:Path,v1:Path,output:Path):
    manifest=validate_frozen(full,v1,output)
    if _digest(test)!=TEST_SHA:raise RuntimeError("Original protected 40-case dataset SHA drift")
    rows=[json.loads(line) for line in test.read_text(encoding="utf8").splitlines() if line.strip()]
    if len(rows)!=40 or len({row["id"] for row in rows})!=40:
        raise RuntimeError("Original test casepack incomplete or duplicated")
    return rows,manifest

def pairwise(a,b):
    abits=a["correct_bitset"];bbits=b["correct_bitset"]
    if len(abits)!=40 or len(bbits)!=40:raise RuntimeError("Incomplete bitsets")
    return {"gained":sum(x=="0" and y=="1" for x,y in zip(abits,bbits)),
            "regressed":sum(x=="1" and y=="0" for x,y in zip(abits,bbits)),
            "net":b["correct"]-a["correct"]}

def main():
    p=argparse.ArgumentParser()
    for option in ("full","v1","base","test","out"):
        p.add_argument("--"+option,required=True,type=Path)
    args=p.parse_args()
    rows,training_receipt=load40(args.test,args.full,args.v1,args.out)
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError("Real RTX4070 needed")
    torch.manual_seed(2026111602)
    t=AutoTokenizer.from_pretrained(str(args.base),use_fast=True,local_files_only=True)
    raw={
      "base_qwen35_9b":run_model("Unchanged-Qwen3.5-9B",None,args.base,rows,t),
      "qlora_v1":run_model("QLoRA-v1",args.v1,args.base,rows,t),
      "qlora_v2_full":run_model("QLoRA-v2-full",args.full/"adapter",args.base,rows,t),
    }
    out={"status":"frozen_three_way_40_complete","case_count":40,
        "test_sha256":TEST_SHA,"same_base":True,
        "same_prompt":True,"same_decode":True,"no_training":True,
        "full_9b_training_sha256":training_receipt["frozen_adapter_sha256"],
        "models":raw,"full_vs_v1":pairwise(raw["qlora_v1"],raw["qlora_v2_full"]),
        "full_vs_qwen_base":pairwise(raw["base_qwen35_9b"],raw["qlora_v2_full"])}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NERAL_V2_FULL_V1_QWEN_40_COMPARISON "+json.dumps({
      key:{"correct":v["correct"],"format_errors":v["format_errors"],
           "median_latency_s":v["median_latency_s"],"peak_reserved_mib":v["peak_reserved_mib"]}
      for key,v in raw.items()},sort_keys=True),flush=True)
    print("NERAL_V2_FULL_40_REGRESSIONS "+json.dumps(out["full_vs_v1"],sort_keys=True),flush=True)

if __name__=="__main__":main()
