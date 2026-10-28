"""Roblox-safe opt-in 2B Arctus350 FP32-master SFT micro-pilot on local NERAL data.

HARD limits: 6 steps, 1 row/batch, <=64 tokens, <=50% PyTorch VRAM,
>=9GiB Windows+CUDA free, new output directory, no final-audit benchmark,
no replay (hence NOT RELEASE/PROMOTABLE), no external pretrained weights.
Any snapshot is labeled experimental and kept solely on the user's PC.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path


def admit_pilot(authorized:bool, windows_free:int, cuda_free:int, fraction:float,
                steps:int, max_seq:int)->bool:
    if (type(authorized) is not bool or type(windows_free) is not int or type(cuda_free) is not int
        or windows_free<0 or cuda_free<0 or type(fraction) not in (float,int)
        or not math.isfinite(fraction) or not 0<fraction<=.50
        or type(steps) is not int or not 1<=steps<=8
        or type(max_seq) is not int or not 16<=max_seq<=64):
        raise ValueError("unsafe short SFT pilot budget")
    return authorized and windows_free>=9216 and cuda_free>=9216


def select_short_examples(tok,rows,*,count:int,max_seq:int):
    from chat350_sft_format import encode_messages
    if type(count) is not int or not 1<=count<=32 or type(max_seq) is not int or not 16<=max_seq<=64:
        raise ValueError("invalid short training sample policy")
    ids=[row["id"] for row in rows]
    if len(ids)!=len(set(ids)):raise ValueError("duplicated row IDs")
    selected=[]
    for row in sorted(rows,key=lambda r:(hashlib.sha256(r["id"].encode("utf8")).hexdigest(),r["id"])):
        try:
            encoded=encode_messages(tok,row["messages"],max_seq_len=max_seq)
        except ValueError:
            continue
        selected.append((row["id"],encoded))
        if len(selected)==count:break
    if len(selected)!=count:
        raise ValueError("insufficient strictly short eligible samples")
    return selected


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--data",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--report",required=True)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    ap.add_argument("--steps",type=int,default=6)
    args=ap.parse_args()
    report_path=Path(args.report)
    out=Path(args.out_dir)
    if report_path.exists() or out.exists():raise FileExistsError("fresh report and trial directory required")
    rep={
        "schema":1,"stage":"native_arctus350_2b_fp32_micro_sft_not_promotable",
        "status":"skipped","steps_completed":0,"requested_steps":args.steps,
        "batch_size":1,"max_seq":64,"replay_used":False,
        "frozen_final_benchmark_used":False,"auto_promotion":False,
        "trial_directory_is_new":True,"source_checkpoint_mutated":False,
        "windows_free_mib_preflight":args.windows_free_mib,
    }
    def finish(status,reason):
        rep["status"]=status;rep["reason"]=reason
        report_path.write_text(json.dumps(rep,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
        print(json.dumps(rep,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if status in ("completed","skipped") else 2
    allowed=os.environ.get("NERAL_GPU_PROBE_ALLOW")=="1"
    if not admit_pilot(allowed,args.windows_free_mib,9216,.5,args.steps,64):
        return finish("skipped","not_authorized_or_windows_vram")
    try:
        import torch
        from chat350_data import validate_dataset_root,validate_prompt_holdouts
        from chat350_contract import EXPECTED_PARAMS,EXPECTED_TOKENIZER_SHA256,validate_checkpoint_metadata,checkpoint_processed_tokens
        from secure_checkpoint350 import load_checkpoint
        from train_chat350 import load_tokenizer,eval_sft,assistant_loss,collate,sha256_file
        from train_precision350 import checkpoint_model_state
        from model350 import Neral350M,FRONTIER_350
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return finish("skipped","cuda_bf16_unavailable")
        free,total=torch.cuda.mem_get_info(0)
        if not admit_pilot(allowed,args.windows_free_mib,free//2**20,.5,args.steps,64):
            return finish("skipped","insufficient_cuda_vram")
        if "RTX 4070" not in torch.cuda.get_device_name(0):
            return finish("skipped","unexpected_gpu")
        torch.cuda.set_per_process_memory_fraction(.5,device=0)
        manifest,splits=validate_dataset_root(Path(args.data))
        validate_prompt_holdouts(splits)
        tok=load_tokenizer(Path(args.tokenizer))
        selected_train=select_short_examples(tok,splits["train"],count=16,max_seq=64)
        selected_val=select_short_examples(tok,splits["val"],count=8,max_seq=64)
        rep["train_sample_ids_sha256"]=hashlib.sha256(
            json.dumps([x[0] for x in selected_train]).encode()).hexdigest()
        rep["val_sample_ids_sha256"]=hashlib.sha256(
            json.dumps([x[0] for x in selected_val]).encode()).hexdigest()
        rep["dataset_manifest_sha256"]=manifest["manifest_sha256"]
        ckpath=Path(args.checkpoint)
        source=load_checkpoint(ckpath)
        validate_checkpoint_metadata(source,min_processed_tokens=2_000_000_000)
        rep["source_processed_tokens"]=checkpoint_processed_tokens(source)
        rep["checkpoint_sha256_before"]=digest(ckpath)
        m=Neral350M()
        m.load_state_dict(source["model"],strict=True)
        del source
        if m.parameter_report()["total"]!=EXPECTED_PARAMS:
            raise ValueError("unexpected model parameter count")
        if torch.cuda.mem_get_info(0)[0]//2**20<9216:
            return finish("skipped","insufficient_free_vram_before_model_transfer")
        m=m.to("cuda",dtype=torch.float32)
        m.set_gradient_checkpointing(True)
        pad=int(tok.token_to_id("<|pad|>"))
        eos=int(tok.token_to_id("<|eos|>"))
        before=eval_sft(m,[x[1] for x in selected_val],1,pad,eos,"cuda")
        rep["baseline_dev_answer_nll"]=before["answer_nll"]
        opt=torch.optim.AdamW(m.parameters(),lr=1.5e-5,betas=(.9,.95),
                              weight_decay=.03,eps=1e-8,fused=True)
        torch.cuda.reset_peak_memory_stats(0)
        train_losses=[]
        for step in range(args.steps):
            if torch.cuda.mem_get_info(0)[0]//2**20<4096:
                return finish("aborted","roblox_memory_pressure_before_step")
            x,y=collate([selected_train[step%len(selected_train)][1]],pad,"cuda")
            opt.zero_grad(set_to_none=True)
            loss=assistant_loss(m,x,y,eos,1.5)
            if not torch.isfinite(loss):return finish("failed","nonfinite_sft_loss")
            loss.backward()
            g=m.norm.weight.grad
            if g is None or not torch.isfinite(g).all():
                return finish("failed","nonfinite_norm_gradients")
            opt.step()
            torch.cuda.synchronize()
            rep["steps_completed"]=step+1
            train_losses.append(float(loss.detach()))
            if torch.cuda.max_memory_reserved(0)>6144*2**20:
                return finish("aborted","allocator_budget_exceeded")
            del x,y,loss
        opt.zero_grad(set_to_none=True)
        after=eval_sft(m,[x[1] for x in selected_val],1,pad,eos,"cuda")
        rep["final_dev_answer_nll"]=after["answer_nll"]
        rep["dev_answer_nll_change"]=after["answer_nll"]-before["answer_nll"]
        rep["train_step_losses"]=train_losses
        rep["peak_pytorch_allocated_mib"]=torch.cuda.max_memory_allocated(0)//2**20
        rep["peak_pytorch_reserved_mib"]=torch.cuda.max_memory_reserved(0)//2**20
        rep["cuda_free_mib_end"]=torch.cuda.mem_get_info(0)[0]//2**20
        rep["checkpoint_sha256_after"]=digest(ckpath)
        if rep["checkpoint_sha256_before"]!=rep["checkpoint_sha256_after"]:
            return finish("failed","foundation_source_digest_changed")
        rep["source_checkpoint_mutated"]=False
        rep["candidate_saved"]=False
        # This is an exploratory candidate, not a validated or resumable
        # promotion; store only if the untouched VAL sample improves.
        if after["answer_nll"]<before["answer_nll"]-.002:
            if shutil.disk_usage(out.parent).free>=2*2**30:
                out.mkdir(parents=True,exist_ok=False)
                candidate=out/"experiment_only_not_release.pt"
                ckpt={
                    "model":checkpoint_model_state(m,"fp32-master",resume=False),
                    "config":FRONTIER_350.to_dict(),
                    "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
                    "source_checkpoint_sha256":rep["checkpoint_sha256_before"],
                    "source_processed_tokens":rep["source_processed_tokens"],
                    "data_manifest_sha256":manifest["manifest_sha256"],
                    "chat_step":args.steps,
                    "validation":after,
                    "experimental_non_promotable":True,
                    "no_replay_or_capability_gate":True,
                }
                torch.save(ckpt,candidate)
                rep["candidate_sha256"]=digest(candidate)
                rep["candidate_size_bytes"]=candidate.stat().st_size
                rep["candidate_saved"]=True
                rep["candidate_local_only"]=True
            else:
                rep["candidate_not_saved_reason"]="insufficient_disk_free"
        return finish("completed","tiny_real_local_data_fp32_sft_experiment_no_promotion")
    except (ValueError,KeyError,OSError,RuntimeError,ImportError,TypeError) as ex:
        rep["error_type"]=type(ex).__name__
        rep["detail"]=str(ex)[:220]
        return finish("failed","pilot_runtime_or_data_error")


if __name__=="__main__":
    sys.exit(main())
