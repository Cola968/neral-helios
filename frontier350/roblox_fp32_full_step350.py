"""Single full-parameter 350M FP32-master AdamW smoke, NEVER saves weights.

Shared-gaming-GPU policy: >=9GiB Windows and CUDA free, 50% PyTorch CUDA
allocator cap, sequence <=64, ONE optimizer step from immutable native 2B.
Synthetic developer QA only; not a model training campaign, quality benchmark
or checkpoint promotion. If CUDA OOMs, record failure instead of overcommitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def admit_full_step(authorized,windows_free,cuda_free,allocator_fraction):
    if (type(authorized) is not bool or type(windows_free) is not int or
        type(cuda_free) is not int or windows_free<0 or cuda_free<0 or
        type(allocator_fraction) not in (float,int) or
        not math.isfinite(allocator_fraction) or
        not 0<allocator_fraction<=.50):
        raise ValueError("unsafe FP32 full-step admission")
    return bool(authorized and windows_free>=9216 and cuda_free>=9216)


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(4*1024*1024),b""):
            sha.update(x)
    return sha.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    args=ap.parse_args()
    output=Path(args.out)
    if output.exists():raise FileExistsError("do not overwrite smoke-test report")
    report={
        "probe":"arctus350_full_fp32_adamw_one_step_v1",
        "status":"skipped","optimizer_steps":0,"saved_weights":False,
        "source_checkpoint_modified":False,"training_dataset_used":False,
        "synthetic_input_only":True,"max_sequence_tokens":64,
        "allocator_cap_fraction":.50,"windows_free_mib_preflight":args.windows_free_mib,
    }
    def finish(status,reason):
        report["status"]=status;report["reason"]=reason
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
        print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if status in ("completed","skipped") else 2
    if not admit_full_step(os.getenv("NERAL_GPU_PROBE_ALLOW")=="1",args.windows_free_mib,9216,.50):
        return finish("skipped","authorization_or_windows_vram")
    try:
        import torch
        from secure_checkpoint350 import load_checkpoint
        from chat350_contract import validate_checkpoint_metadata,checkpoint_processed_tokens,EXPECTED_TOKENIZER_SHA256,EXPECTED_PARAMS
        from train_chat350 import load_tokenizer,assistant_loss
        from chat350_sft_format import encode_prompt_answer
        from model350 import Neral350M
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return finish("skipped","cuda_bf16_unavailable")
        free,tot=torch.cuda.mem_get_info(0)
        if not admit_full_step(True,args.windows_free_mib,free//2**20,.50):
            return finish("skipped","cuda_vram_below_9gib")
        if "RTX 4070" not in torch.cuda.get_device_name(0):
            return finish("skipped","unexpected_gpu")
        torch.cuda.set_per_process_memory_fraction(.50,device=0)
        tokenizer_path=Path(args.tokenizer)
        if digest(tokenizer_path)!=EXPECTED_TOKENIZER_SHA256:
            return finish("failed","tokenizer_mismatch")
        tok=load_tokenizer(tokenizer_path)
        path=Path(args.checkpoint)
        ck=load_checkpoint(path)
        validate_checkpoint_metadata(ck,min_processed_tokens=2_000_000_000)
        report["source_processed_tokens"]=checkpoint_processed_tokens(ck)
        report["checkpoint_sha256_before"]=digest(path)
        model=Neral350M()
        model.load_state_dict(ck["model"],strict=True)
        del ck
        if model.parameter_report()["total"]!=EXPECTED_PARAMS:
            raise ValueError("unexpected model parameter count")
        if torch.cuda.mem_get_info(0)[0]//2**20<9216:
            return finish("skipped","vram_changed_before_transfer")
        model=model.to("cuda",dtype=torch.float32)
        model.set_gradient_checkpointing(True)
        model.train()
        # A few exact reference weights, not a full copy of 350M parameters.
        named=dict(model.named_parameters())
        keys=("norm.weight","blocks.0.attn.q.weight","blocks.0.ffn.gate.weight")
        before={name:named[name].detach().flatten()[:4096].clone()
                for name in keys}
        example=encode_prompt_answer(tok,"Was ist 2 + 2?","4",max_seq_len=64)
        x=torch.tensor([example.input_ids],device="cuda",dtype=torch.long)
        y=torch.tensor([example.targets],device="cuda",dtype=torch.long)
        torch.cuda.reset_peak_memory_stats(0)
        opt=torch.optim.AdamW(model.parameters(),lr=1.5e-5,betas=(.9,.95),
                              eps=1e-8,weight_decay=.03,fused=True)
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,int(tok.token_to_id("<|eos|>")),1.5)
        if not torch.isfinite(loss):
            return finish("failed","nonfinite_loss")
        loss.backward()
        for name in keys:
            g=named[name].grad
            if g is None or not torch.isfinite(g).all():
                return finish("failed","nonfinite_gradient_"+name)
        if torch.cuda.mem_get_info(0)[0]//2**20<4096:
            return finish("aborted","roblox_free_vram_guard_before_optimizer")
        opt.step()
        torch.cuda.synchronize()
        report["optimizer_steps"]=1
        report["loss_before_update"]=float(loss.detach())
        report["selected_parameter_changes"]={
            name:float((named[name].detach().flatten()[:4096]!=original).float().mean())
            for name,original in before.items()
        }
        report["optimizer_state_bytes"]=int(sum(t.numel()*t.element_size()
            for state in opt.state.values()
            for t in state.values() if isinstance(t,torch.Tensor)))
        report["peak_pytorch_allocated_mib"]=torch.cuda.max_memory_allocated(0)//2**20
        report["peak_pytorch_reserved_mib"]=torch.cuda.max_memory_reserved(0)//2**20
        report["cuda_free_mib_after"]=torch.cuda.mem_get_info(0)[0]//2**20
        report["checkpoint_sha256_after"]=digest(path)
        if report["checkpoint_sha256_after"]!=report["checkpoint_sha256_before"]:
            return finish("failed","source_checkpoint_mutated")
        if not all(v>0 for v in report["selected_parameter_changes"].values()):
            return finish("failed","unchanged_sampled_linear_or_norm_weights")
        if report["peak_pytorch_reserved_mib"]>6144:
            return finish("aborted","pytorch_memory_cap_exceeded")
        return finish("completed","genuine_2b_full_fp32_adamw_step_measured")
    except (RuntimeError,ValueError,KeyError,OSError,ImportError,TypeError) as ex:
        report["error_type"]=type(ex).__name__
        report["detail"]=str(ex)[:240]
        return finish("failed","memory_or_optimizer_error")


if __name__=="__main__":
    sys.exit(main())
