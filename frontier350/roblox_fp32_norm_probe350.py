"""RTX4070 shared-GPU opt-in FP32-master/BF16-AMP gradient smoke test.

Loads immutable native 2B foundation; updates ONLY the tiny RMSNorm tensors
inside THIS process, then discards them. Never saves/checkpoints full model,
never executes full-parameter AdamW. Allocator capped to 38% to protect Roblox.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


PROBE_NAME="arctus350_fp32_master_norm_only_smoke_v1"

def allow_norm_probe(authorized:bool,windows_free_mib:int,
                     cuda_free_mib:int,allocator_fraction:float)->bool:
    if (type(authorized) is not bool or type(windows_free_mib) is not int
        or type(cuda_free_mib) is not int or
        type(allocator_fraction) not in (int,float) or
        not math.isfinite(allocator_fraction) or
        windows_free_mib<0 or cuda_free_mib<0 or
        not 0<allocator_fraction<=0.40):
        raise ValueError("invalid GPU precision probe safety policy")
    return (authorized and windows_free_mib>=8192 and cuda_free_mib>=8192)


def summarize_update(before,after):
    if (not isinstance(before,(tuple,list)) or not isinstance(after,(tuple,list))
        or not before or len(before)!=len(after)):
        raise ValueError("inconsistent parameter observations")
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in (*before,*after)):
        raise ValueError("nonfinite parameter values")
    changed=sum(x!=y for x,y in zip(before,after))
    return {"elements":len(before),"changed_fraction":changed/len(before),
            "mean_abs_change":sum(abs(a-b) for a,b in zip(before,after))/len(before)}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--windows-free-mib",required=True,type=int)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError("never overwrite old evidence")
    report={"probe":PROBE_NAME,"status":"skipped","training_corpus_used":False,
            "checkpoint_written":False,"full_parameter_optimizer":False,
            "in_memory_norm_adamw_steps":0,"windows_free_mib_preflight":args.windows_free_mib}
    def write(status,reason):
        report["status"]=status
        report["reason"]=reason
        out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
        print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if status in ("completed","skipped") else 2
    allowed=os.getenv("NERAL_GPU_PROBE_ALLOW")=="1"
    if args.windows_free_mib<8192 or not allowed:
        return write("skipped","windows_vram_or_authorization")
    try:
        import torch
        from secure_checkpoint350 import load_checkpoint
        from chat350_contract import validate_checkpoint_metadata,checkpoint_processed_tokens,EXPECTED_TOKENIZER_SHA256,EXPECTED_PARAMS
        from train_chat350 import load_tokenizer,assistant_loss
        from chat350_sft_format import encode_prompt_answer
        from model350 import Neral350M
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return write("skipped","no_bf16_cuda")
        free,total=torch.cuda.mem_get_info(0)
        if not allow_norm_probe(allowed,args.windows_free_mib,free//2**20,.38):
            return write("skipped","cuda_headroom")
        if "RTX 4070" not in torch.cuda.get_device_name(0):
            return write("skipped","unexpected_gpu")
        torch.cuda.set_per_process_memory_fraction(.38,device=0)
        ck=Path(args.checkpoint);tokp=Path(args.tokenizer)
        if digest(tokp)!=EXPECTED_TOKENIZER_SHA256:
            return write("failed","tokenizer_hash")
        tok=load_tokenizer(tokp)
        payload=load_checkpoint(ck)
        validate_checkpoint_metadata(payload,min_processed_tokens=2_000_000_000)
        report["processed_tokens"]=checkpoint_processed_tokens(payload)
        report["checkpoint_sha256_before"]=digest(ck)
        model=Neral350M()
        model.load_state_dict(payload["model"],strict=True)
        del payload
        if model.parameter_report()["total"]!=EXPECTED_PARAMS:raise ValueError("model parameters changed")
        if torch.cuda.mem_get_info(0)[0]//2**20<8192:
            return write("skipped","headroom_changed_before_cuda_transfer")
        model=model.to("cuda",dtype=torch.float32)
        model.set_gradient_checkpointing(True)
        model.train()
        norms=[p for name,p in model.named_parameters()
               if p.ndim==1 and name.endswith(".weight")]
        count=sum(p.numel() for p in norms)
        if count!=59520: raise ValueError("unexpected RMSNorm parameter count")
        original=torch.cat([p.detach().float().cpu().flatten() for p in norms]).tolist()
        example=encode_prompt_answer(tok,"Was ist 2 + 2?","4",max_seq_len=64)
        x=torch.tensor([example.input_ids],device="cuda",dtype=torch.long)
        y=torch.tensor([example.targets],device="cuda",dtype=torch.long)
        opt=torch.optim.AdamW(norms,lr=1.5e-5,betas=(.9,.95),eps=1e-8,
                              weight_decay=.03,fused=True)
        torch.cuda.reset_peak_memory_stats(0)
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,int(tok.token_to_id("<|eos|>")),1.5)
        if not torch.isfinite(loss):
            return write("failed","nonfinite_initial_loss")
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(norms,1.0,error_if_nonfinite=True))
        if not math.isfinite(grad_norm) or grad_norm<=0:
            return write("failed","invalid_norm_gradient")
        if torch.cuda.mem_get_info(0)[0]//2**20<4096:
            return write("aborted","roblox_vram_pressure_before_update")
        opt.step()
        torch.cuda.synchronize()
        updated=torch.cat([p.detach().float().cpu().flatten() for p in norms]).tolist()
        stats=summarize_update(original,updated)
        report.update(
            in_memory_norm_adamw_steps=1,
            gradient_l2_before_clip=grad_norm,
            loss_before_update=float(loss.detach()),
            rmsnorm_update=stats,
            peak_pytorch_allocated_mib=torch.cuda.max_memory_allocated(0)//2**20,
            peak_pytorch_reserved_mib=torch.cuda.max_memory_reserved(0)//2**20,
            cuda_free_mib_after=torch.cuda.mem_get_info(0)[0]//2**20,
            checkpoint_sha256_after=digest(ck),
        )
        if report["checkpoint_sha256_after"]!=report["checkpoint_sha256_before"]:
            return write("failed","checkpoint_file_changed")
        if stats["changed_fraction"]==0:
            return write("failed","fp32_norm_weights_not_updated")
        if report["peak_pytorch_reserved_mib"]>4700:
            return write("aborted","gpu_memory_cap_exceeded")
        return write("completed","real_2b_fp32_master_norm_gradient_verified")
    except (RuntimeError,ValueError,IndexError,ImportError,TypeError,OSError) as ex:
        report["detail"]=str(ex)[:240]
        return write("failed",type(ex).__name__)


if __name__=="__main__":sys.exit(main())
