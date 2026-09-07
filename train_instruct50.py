"""Neral-50M Stage D: assistant-only instruction tuning with base-language replay.

Stage D is deliberately model-only: it changes the exact 50,012,976 Neral weights and does
not call SOLVE, VERIFY, Helios tools or any external model. Prompt tokens are context only;
assistant answer bytes, newline and EOS are the supervised targets. A small document-bounded
base-LM replay loss protects pretraining capability while SFT teaches the chat format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from pretrain50 import loss as lm_loss
from pretrain50_b2 import load_streams as load_b2_streams, load_documents as load_b2_documents
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch
from tokenizer import ByteTokenizer
from train_raw_math50 import dev_math_nll

EXPECTED_PARAMS=50_012_976
STAGE_VERSION=1
SEED=968_540
TOKENIZER=ByteTokenizer()


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def data_hashes(root: Path):
    names=("train.jsonl","val.jsonl","test.jsonl","holdout.jsonl","manifest.json")
    out={}
    for name in names:
        path=root/name
        if not path.is_file():raise FileNotFoundError(path)
        out[name]=sha256_file(path)
    return out


def encode_sft(row,seq):
    prefix=TOKENIZER.encode("User: "+row["prompt"].strip()+"\nAssistant:",bos=True)
    answer=TOKENIZER.encode(" "+row["answer"].strip()+"\n",eos=True)
    ids=prefix+answer
    if len(ids)-1>seq:raise ValueError(f"SFT row {row.get('id')} exceeds seq={seq}: {len(ids)-1}")
    targets=[-100]*len(prefix)+answer
    x=torch.full((seq,),TOKENIZER.pad_id,dtype=torch.long)
    y=torch.full((seq,),-100,dtype=torch.long)
    a=torch.tensor(ids[:-1],dtype=torch.long);b=torch.tensor(targets[1:],dtype=torch.long)
    x[:len(a)]=a;y[:len(b)]=b
    return x,y


def trim_sft_padding(x,y):
    """Remove only trailing columns that are padding/ignored for every row in the batch.

    For a causal LM, removing common trailing padding cannot alter logits at any supervised
    position because those positions never attend to future tokens. This therefore preserves
    the exact SFT objective while reducing attention work for short/mixed-length batches.
    """
    if x.ndim!=2 or y.ndim!=2 or x.shape!=y.shape:raise ValueError("SFT tensors must be same-shape rank-2")
    active=(x!=TOKENIZER.pad_id)|(y!=-100)
    if not bool(active.any()):raise ValueError("SFT batch contains no active tokens")
    width=int(torch.nonzero(active,as_tuple=False)[:,1].max().item())+1
    return x[:,:width].contiguous(),y[:,:width].contiguous()


def make_sft_batch(rows,seq,device):
    encoded=[encode_sft(row,seq) for row in rows]
    x=torch.stack([a for a,_ in encoded])
    y=torch.stack([b for _,b in encoded])
    x,y=trim_sft_padding(x,y)
    return x.to(device),y.to(device)


def assistant_loss(model,x,y,eos_weight):
    with torch.autocast("cuda",dtype=torch.bfloat16):logits,_=model(x)
    flat_logits=logits.float().reshape(-1,logits.size(-1));flat_y=y.reshape(-1)
    per=F.cross_entropy(flat_logits,flat_y,ignore_index=-100,reduction="none")
    valid=flat_y!=-100
    weights=torch.ones_like(per)
    weights[(flat_y==TOKENIZER.eos_id)&valid]=eos_weight
    value=(per[valid]*weights[valid]).sum()/weights[valid].sum().clamp_min(1.0)
    return value


@torch.inference_mode()
def evaluate_sft(model,rows,device,seq,batch_size=16):
    was=model.training;model.eval();groups={}
    total_nll=0.0;total_tokens=0;eos_correct=0;eos_total=0
    slices=sorted({(r["language"],r["domain"]) for r in rows})
    for language,domain in slices:
        chosen=[r for r in rows if r["language"]==language and r["domain"]==domain]
        nll=0.0;tokens=0;ceos=0;teos=0
        for start in range(0,len(chosen),batch_size):
            x,y=make_sft_batch(chosen[start:start+batch_size],seq,device)
            with torch.autocast("cuda",dtype=torch.bfloat16):logits,_=model(x)
            flat=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100,reduction="none").reshape_as(y)
            mask=y!=-100;nll+=float(flat[mask].sum());tokens+=int(mask.sum())
            eos_mask=y==TOKENIZER.eos_id
            if eos_mask.any():
                pred=logits.argmax(-1);ceos+=int((pred[eos_mask]==TOKENIZER.eos_id).sum());teos+=int(eos_mask.sum())
        groups[f"{language}/{domain}"]={"answer_nll":nll/max(1,tokens),"answer_tokens":tokens,"examples":len(chosen),"terminal_eos_top1_pct":100*ceos/max(1,teos)}
        total_nll+=nll;total_tokens+=tokens;eos_correct+=ceos;eos_total+=teos
    macro=sum(v["answer_nll"] for v in groups.values())/max(1,len(groups))
    model.train(was)
    return {"macro_answer_nll":macro,"micro_answer_nll":total_nll/max(1,total_tokens),"answer_tokens":total_tokens,"terminal_eos_top1_pct":100*eos_correct/max(1,eos_total),"slices":groups}


def lr_at(step,total,peak,floor,warmup_fraction):
    warm=max(1,round(total*warmup_fraction))
    if step<=warm:return peak*step/warm
    p=(step-warm)/max(1,total-warm);p=min(1.0,max(0.0,p))
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*p))


def next_rows(train,order,cursor,count,rng):
    picked=[]
    while len(picked)<count:
        if cursor>=len(order):rng.shuffle(order);cursor=0
        take=min(count-len(picked),len(order)-cursor)
        picked.extend(train[order[i]] for i in range(cursor,cursor+take));cursor+=take
    return picked,cursor


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--init-from",required=True)
    ap.add_argument("--data",default="sft50_data")
    ap.add_argument("--replay-data",default="pretrain50_b2_data")
    ap.add_argument("--out",default="runs/neral50m_stage_d_instruct_v1")
    ap.add_argument("--steps",type=int,default=2000);ap.add_argument("--seq",type=int,default=512)
    ap.add_argument("--batch-size",type=int,default=16);ap.add_argument("--replay-batch",type=int,default=8)
    ap.add_argument("--sft-weight",type=float,default=.85);ap.add_argument("--eos-weight",type=float,default=1.5)
    ap.add_argument("--peak-lr",type=float,default=3e-5);ap.add_argument("--min-lr",type=float,default=3e-6)
    ap.add_argument("--warmup-fraction",type=float,default=.05);ap.add_argument("--eval-every",type=int,default=100)
    ap.add_argument("--patience",type=int,default=6);ap.add_argument("--min-delta",type=float,default=.003)
    args=ap.parse_args()
    if min(args.steps,args.seq,args.batch_size,args.replay_batch,args.eval_every,args.patience)<=0:raise ValueError("positive budgets required")
    if args.seq>FIFTY.max_seq_len:raise ValueError("sequence exceeds model context")
    if not .65<=args.sft_weight<=.95:raise ValueError("sft-weight must be 0.65..0.95")
    if not 1.0<=args.eos_weight<=3.0:raise ValueError("eos-weight must be 1..3")
    if not 0<args.min_lr<=args.peak_lr:raise ValueError("invalid learning rates")
    if not 0<=args.warmup_fraction<=.2:raise ValueError("warmup-fraction must be 0..0.2")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("Stage D requires BF16 CUDA")

    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_float32_matmul_precision("high")
    rng=random.Random(SEED);device=torch.device("cuda");root=Path(args.data);replay_root=Path(args.replay_data)
    hashes=data_hashes(root);train=read_jsonl(root/"train.jsonl");val=read_jsonl(root/"val.jsonl")
    if not train or not val:raise ValueError("SFT train/val must be non-empty")
    for row in train+val:encode_sft(row,args.seq)

    replay_streams=load_b2_streams(replay_root,"train");replay_val=load_b2_streams(replay_root,"val");replay_docs=load_b2_documents(replay_root)
    replay_blocks=document_blocks(replay_docs,"de",args.seq,"train")+document_blocks(replay_docs,"en",args.seq,"train")
    rng.shuffle(replay_blocks)
    replay_val_schedule,replay_val_meta=build_validation_schedule(replay_docs,args.seq,96,SEED+17)

    model,payload=load_model(args.init_from,device)
    if model.parameter_report()["trainable"]!=EXPECTED_PARAMS:raise ValueError("Stage D accepts exact Neral-50M only")
    model.train();source_tokens=int(payload.get("tokens",0));source_sha=sha(args.init_from)
    baseline_sft=evaluate_sft(model,val,device,args.seq)
    baseline_lm=validate_document_bounded(model,replay_val,replay_val_schedule,args.seq,device)
    baseline_math=dev_math_nll(model,device,min(args.seq,256))

    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Use a fresh Stage-D output directory")
    out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),weight_decay=.03,fused=True)
    order=list(range(len(train)));rng.shuffle(order);cursor=0;replay_cursor=0
    best=baseline_sft["macro_answer_nll"];best_eos=baseline_sft["terminal_eos_top1_pct"];bad=0;step=0
    run={
        "stage":"D assistant-only instruction tuning","stage_version":STAGE_VERSION,"parameters":EXPECTED_PARAMS,
        "source_checkpoint":str(Path(args.init_from).resolve()),"source_sha256":source_sha,"source_cumulative_tokens":source_tokens,
        "data_hashes":hashes,"replay_root":str(replay_root.resolve()),"replay_validation":replay_val_meta,
        "loss":{"assistant_only":True,"assistant_eos_in_loss":True,"eos_weight":args.eos_weight,"sft_weight":args.sft_weight,"base_lm_replay_weight":1-args.sft_weight},
        "sampling":"deterministic shuffled train sweep; no domain-balanced hidden oversampling","solve_used":False,"tools_used":False,
        "padding_policy":"right-trim batch columns that are padding/ignored for every SFT row; target tokens and causal logits are unchanged",
        "selection":"lowest SFT validation macro answer NLL subject to <=5% B2-LM validation regression and <=10% RawMath dev NLL regression",
        "benchmark_policy":"Frozen free/v3/v4/RawMath/NIS benchmark required after Stage D; no promotion from SFT loss alone.",
        "baseline_sft":baseline_sft,"baseline_lm":baseline_lm,"baseline_math":baseline_math,"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"args":vars(args),
    }
    (out/"run.json").write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding="utf8")

    def checkpoint():
        return {"model":model.state_dict(),"optimizer":opt.state_dict(),"config":FIFTY.to_dict(),"tokens":source_tokens,
                "source_cumulative_tokens":source_tokens,"stage_d_version":STAGE_VERSION,"stage_d_step":step,
                "best_sft_macro_answer_nll":best,"best_sft_terminal_eos_top1_pct":best_eos,
                "source_sha256":source_sha,"sft_data_hashes":hashes}

    atomic_save(checkpoint(),out/"best_instruct.pt");atomic_save(checkpoint(),out/"latest.pt")
    start=time.perf_counter();update_seconds=0.0;input_tokens=0;answer_tokens=0;replay_tokens=0;stop="max_steps";torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1,args.steps+1):
            rows,cursor=next_rows(train,order,cursor,args.batch_size,rng);x,y=make_sft_batch(rows,args.seq,device)
            replay_items=[]
            for _ in range(args.replay_batch):
                if replay_cursor>=len(replay_blocks):rng.shuffle(replay_blocks);replay_cursor=0
                replay_items.append(replay_blocks[replay_cursor]);replay_cursor+=1
            lx,ly=make_batch(replay_streams,replay_items,args.seq,device)
            lr=lr_at(step,args.steps,args.peak_lr,args.min_lr,args.warmup_fraction)
            for g in opt.param_groups:g["lr"]=lr
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            sft=assistant_loss(model,x,y,args.eos_weight);base=lm_loss(model,lx,ly);total=args.sft_weight*sft+(1-args.sft_weight)*base
            if not torch.isfinite(total):raise FloatingPointError("non-finite Stage-D loss")
            total.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();dt=time.perf_counter()-tick
            update_seconds+=dt;input_tokens+=int((x!=TOKENIZER.pad_id).sum());answer_tokens+=int((y!=-100).sum());replay_tokens+=lx.numel()
            row={"step":step,"loss":float(total),"sft_loss":float(sft.detach()),"base_replay_loss":float(base.detach()),"lr":lr,"grad_norm":float(gn),"seconds":dt,
                 "sft_compute_width":int(x.shape[1]),"sft_max_seq":args.seq,
                 "input_tokens":input_tokens,"answer_tokens":answer_tokens,"replay_tokens":replay_tokens,"optimizer_tokens_per_second":(x.numel()+lx.numel())/max(dt,1e-9)}
            if step%args.eval_every==0 or step==args.steps:
                sft_val=evaluate_sft(model,val,device,args.seq);lm_val=validate_document_bounded(model,replay_val,replay_val_schedule,args.seq,device);math_val=dev_math_nll(model,device,min(args.seq,256))
                lm_ok=lm_val["macro"]<=baseline_lm["macro"]*1.05;math_ok=math_val["nll"]<=baseline_math["nll"]*1.10
                improved=lm_ok and math_ok and sft_val["macro_answer_nll"]<best-args.min_delta
                row.update(sft_validation=sft_val,base_validation=lm_val,math_dev=math_val,lm_guard=lm_ok,math_guard=math_ok,selected=improved)
                if improved:
                    best=sft_val["macro_answer_nll"];best_eos=sft_val["terminal_eos_top1_pct"];bad=0;atomic_save(checkpoint(),out/"best_instruct.pt")
                else:bad+=1
                row["bad_evals"]=bad;atomic_save(checkpoint(),out/"latest.pt");print(json.dumps(row),flush=True)
                if bad>=args.patience:stop="early_stopping";break
            with (out/"metrics.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
    finally:
        atomic_save(checkpoint(),out/"latest.pt")
        summary={"stop":stop,"complete":stop in {"max_steps","early_stopping"},"completed_step":step,"source_cumulative_tokens":source_tokens,
                 "sft_input_tokens":input_tokens,"sft_answer_tokens":answer_tokens,"base_replay_tokens":replay_tokens,
                 "best_sft_macro_answer_nll":best,"best_sft_terminal_eos_top1_pct":best_eos,
                 "baseline_sft_macro_answer_nll":baseline_sft["macro_answer_nll"],"baseline_sft_terminal_eos_top1_pct":baseline_sft["terminal_eos_top1_pct"],
                 "optimizer_seconds":update_seconds,"elapsed_seconds":time.perf_counter()-start,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
                 "best_sha256":sha(out/"best_instruct.pt"),"latest_sha256":sha(out/"latest.pt"),"solve_used":False,"tools_used":False}
        (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8");print(json.dumps(summary),flush=True)


if __name__=="__main__":main()