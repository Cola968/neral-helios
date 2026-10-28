"""Read-only, low-VRAM native 350M validation NLL partition (NO training).

Evaluate exactly one pre-existing checkpoint against a deterministic sample
from the unchanged validation split. Returns aggregated numbers, never emits
underlying texts, labels, raw data, or model weights. For shared Roblox GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--max-rows",type=int,default=24)
    ap.add_argument("--max-seq",type=int,default=512)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError("refuse overwrite evidence")
    if not 1<=args.max_rows<=48 or not 64<=args.max_seq<=512:
        raise ValueError("fixed safe evaluation budget violated")
    result={"status":"skipped","read_only":True,"training_started":False,
            "stage":"native_arctus350_token_partition_nll_v1",
            "max_rows":args.max_rows,"max_seq":args.max_seq,
            "windows_free_mib_preflight":args.windows_free_mib,
            "split":"val","partition_note":
                "opening token contains possible word content; body includes newline; "
                "EOS alone is unambiguous special-format token"}
    def write(status,reason):
        result["status"]=status
        result["reason"]=reason
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if status in ("completed","skipped") else 2
    if os.environ.get("NERAL_GPU_PROBE_ALLOW")!="1":
        return write("skipped","not_authorized")
    if args.windows_free_mib<8192:
        return write("skipped","insufficient_windows_vram")
    try:
        import torch
        import torch.nn.functional as F
        from chat350_contract import EXPECTED_TOKENIZER_SHA256, checkpoint_processed_tokens,validate_checkpoint_metadata,EXPECTED_PARAMS
        from chat350_data import validate_dataset_root
        from chat350_nll_partition import stable_subset,classify_supervised_tokens,aggregate_nll
        from chat350_sft_format import encode_messages
        from train_chat350 import load_tokenizer
        from model350 import Neral350M
        from secure_checkpoint350 import load_checkpoint
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return write("skipped","CUDA_BF16_unavailable")
        free_before,total=torch.cuda.mem_get_info(0)
        result["cuda_free_mib_before"]=free_before//2**20
        if free_before//2**20<8192:
            return write("skipped","insufficient_cuda_vram")
        tokenizer_path=Path(args.tokenizer)
        if file_sha(tokenizer_path)!=EXPECTED_TOKENIZER_SHA256:
            return write("failed","wrong_tokenizer_hash")
        tok=load_tokenizer(tokenizer_path)
        manifest,splits=validate_dataset_root(Path(args.data))
        rows=stable_subset(splits["val"],args.max_rows)
        result["sample_ids_sha256"]=hashlib.sha256(
            json.dumps([r["id"] for r in rows],separators=(",",":")).encode("utf8")).hexdigest()
        result["dataset_manifest_sha256"]=manifest["manifest_sha256"]
        result["validation_sha256"]=manifest["splits"]["val"]["sha256"]
        checkpoint_path=Path(args.checkpoint)
        ck=load_checkpoint(checkpoint_path)
        validate_checkpoint_metadata(ck)
        result["source_processed_tokens"]=checkpoint_processed_tokens(ck)
        result["checkpoint_sha256"]=file_sha(checkpoint_path)
        result["checkpoint_size_bytes"]=checkpoint_path.stat().st_size
        result["tokenizer_sha256"]=EXPECTED_TOKENIZER_SHA256
        model=Neral350M()
        model.load_state_dict(ck["model"],strict=True)
        del ck
        if model.parameter_report()["total"]!=EXPECTED_PARAMS:
            return write("failed","unexpected_parameter_count")
        if torch.cuda.mem_get_info(0)[0]//2**20<8192:
            return write("skipped","vram_changed_before_gpu_transfer")
        model=model.to(device="cuda",dtype=torch.bfloat16)
        model.eval()
        torch.cuda.reset_peak_memory_stats(0)
        partitions={}
        skipped=0
        with torch.inference_mode():
            for row in rows:
                if torch.cuda.mem_get_info(0)[0]//2**20<6144:
                    return write("aborted","roblox_vram_pressure")
                try:
                    encoded=encode_messages(tok,row["messages"],max_seq_len=args.max_seq)
                except ValueError:
                    skipped+=1
                    continue
                tags=classify_supervised_tokens(tok,row["messages"],encoded,max_seq_len=args.max_seq)
                x=torch.tensor([encoded.input_ids],device="cuda",dtype=torch.long)
                y=torch.tensor(encoded.targets,device="cuda",dtype=torch.long)
                logits,_=model(x)
                mask=y!=-100
                per=F.cross_entropy(logits[0,mask].float(),y[mask],reduction="none")
                # Small CPU reduction: only a single row of loss values is transferred.
                losses=[float(v) for v in per.detach().cpu().tolist()]
                real_tags=[t for t in tags if t!="masked"]
                if len(losses)!=len(real_tags):
                    raise ValueError("teacher-forced token loss classification mismatch")
                item={k:(sum(l for l,t in zip(losses,real_tags) if t==k),sum(t==k for t in real_tags))
                      for k in ("opening","body","eos")}
                groups=("overall","language:"+row["language"],
                        "turns:"+("multi" if len(row["messages"])>=4 else "single"))
                for group in groups:partitions.setdefault(group,[]).append(item)
                del logits,per,x,y
                if torch.cuda.max_memory_reserved(0)>3*2**30:
                    return write("aborted","pytorch_reserved_vram_gt_3GiB")
        if not partitions.get("overall"):
            return write("failed","zero_eligible_rows")
        result["groups"]={group:{"rows":len(items),**aggregate_nll(items)}
                          for group,items in sorted(partitions.items())}
        result["encoded_rows"]=len(partitions["overall"])
        result["skipped_overlong_rows"]=skipped
        result["peak_pytorch_allocated_mib"]=torch.cuda.max_memory_allocated(0)//2**20
        result["peak_pytorch_reserved_mib"]=torch.cuda.max_memory_reserved(0)//2**20
        result["cuda_free_mib_after"]=torch.cuda.mem_get_info(0)[0]//2**20
        if skipped:
            # Incomplete selection cannot silently pass a quality gate.
            return write("partial","some_rows_exceeded_seq_limit")
        return write("completed","measured_real_validation_token_losses")
    except (OSError,RuntimeError,ValueError,KeyError,ImportError,TypeError,IndexError) as ex:
        result["detail"]=str(ex)[:320]
        return write("failed",type(ex).__name__)


if __name__=="__main__":
    sys.exit(main())
