"""Neral-50M Stage C: contamination-safe RawMath specialization with language replay.

This stage changes the 50,012,976 Neral weights directly. SOLVE is never called.
Math targets are answer-only and include EOS. General-language replay is mixed into
all optimizer updates to reduce catastrophic forgetting.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from pretrain50 import validate
from pretrain50_stage_b import load_documents, load_streams, document_blocks
from raw_math50_data import TrainingProblemSampler, make_dev, validate_frozen_suite
from tokenizer import ByteTokenizer

EXPECTED_PARAMS=50_012_976
SEED=968_520
TOKENIZER=ByteTokenizer()


def encode_math(row, seq: int):
    prefix=TOKENIZER.encode("Question: "+row["prompt"].strip()+"\nAnswer:",bos=True)
    answer=TOKENIZER.encode(" "+row["answer"].strip()+"\n",eos=True)
    ids=prefix+answer
    if len(ids)-1>seq:
        raise ValueError(f"RawMath example exceeds seq={seq}: {row['canonical']}")
    targets=[-100]*len(prefix)+answer
    x=torch.full((seq,),TOKENIZER.pad_id,dtype=torch.long)
    y=torch.full((seq,),-100,dtype=torch.long)
    a=torch.tensor(ids[:-1],dtype=torch.long);b=torch.tensor(targets[1:],dtype=torch.long)
    x[:len(a)]=a;y[:len(b)]=b
    return x,y


def make_language_batch(streams, blocks, count, seq, rng):
    pieces=[]
    for _ in range(count):
        lang,start=rng.choice(blocks)
        z=np.asarray(streams[lang][start:start+seq+1],dtype=np.int64)
        if len(z)!=seq+1: raise ValueError("Short language replay block")
        pieces.append(z)
    arr=torch.from_numpy(np.stack(pieces))
    return arr[:,:-1],arr[:,1:]


def make_math_batch(sampler,count,seq):
    xs=[];ys=[]
    for _ in range(count):
        x,y=encode_math(sampler.next(),seq);xs.append(x);ys.append(y)
    return torch.stack(xs),torch.stack(ys)


def mixed_batch(streams,blocks,sampler,batch,math_fraction,seq,rng,device):
    n_math=max(1,min(batch-1,round(batch*math_fraction)));n_lang=batch-n_math
    mx,my=make_math_batch(sampler,n_math,seq);lx,ly=make_language_batch(streams,blocks,n_lang,seq,rng)
    x=torch.cat([mx,lx],0);y=torch.cat([my,ly],0)
    order=torch.randperm(batch,generator=torch.Generator().manual_seed(SEED+sampler.emitted))
    math_targets=int((my!=-100).sum())
    return x[order].to(device),y[order].to(device),n_math,n_lang,math_targets


def answer_only_loss(model,x,y):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits,_=model(x)
    return F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100)


@torch.inference_mode()
def dev_math_nll(model,device,seq=256,batch_size=24):
    model.eval();rows=make_dev();total=0.0;tokens=0;by={}
    for cat in sorted({r["category"] for r in rows}):
        selected=[r for r in rows if r["category"]==cat];closs=0.0;ctok=0
        for i in range(0,len(selected),batch_size):
            enc=[encode_math(r,seq) for r in selected[i:i+batch_size]]
            x=torch.stack([a for a,_ in enc]).to(device);y=torch.stack([b for _,b in enc]).to(device)
            with torch.autocast("cuda",dtype=torch.bfloat16): logits,_=model(x)
            v=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100,reduction="sum")
            n=int((y!=-100).sum());closs+=float(v);ctok+=n
        by[cat]={"nll":closs/max(1,ctok),"answer_tokens":ctok,"tasks":len(selected)};total+=closs;tokens+=ctok
    model.train();return {"nll":total/max(1,tokens),"answer_tokens":tokens,"tasks":len(rows),"by_category":by}


def cosine(progress,peak,floor):
    p=min(1.0,max(0.0,progress));return floor+(peak-floor)*0.5*(1+math.cos(math.pi*p))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--init-from",required=True)
    ap.add_argument("--out",default="runs/neral50m_stage_c_rawmath_v1")
    ap.add_argument("--tokens",type=int,default=15_000_000,help="Input-token budget including math and language replay")
    ap.add_argument("--seq",type=int,default=256);ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--math-fraction",type=float,default=.35)
    ap.add_argument("--peak-lr",type=float,default=5e-5);ap.add_argument("--min-lr",type=float,default=5e-6)
    ap.add_argument("--eval-every",type=int,default=100);args=ap.parse_args()
    if args.tokens<args.seq*args.batch:raise ValueError("Token budget too small")
    if not .15<=args.math_fraction<=.60:raise ValueError("math-fraction must be 0.15..0.60")
    if not 0<args.min_lr<=args.peak_lr:raise ValueError("Invalid LR range")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("Stage C requires BF16 CUDA")
    validate_frozen_suite()
    torch.set_num_threads(4);torch.set_float32_matmul_precision("high");torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    device=torch.device("cuda");model,payload=load_model(args.init_from,device)
    if model.parameter_report()["trainable"]!=EXPECTED_PARAMS:raise ValueError("RawMath Stage C accepts exact Neral-50M only")
    model.train();out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Use a fresh Stage-C output directory")
    out.mkdir(parents=True,exist_ok=True)
    streams=load_streams("train");val=load_streams("val");docs=load_documents()
    de=document_blocks(docs,"de",args.seq);en=document_blocks(docs,"en",args.seq)
    rr=random.Random(SEED);replay=[]
    # Deterministic 60/40 replay pool, independent of math sampling.
    n=min(len(de),max(1,int(len(en)*1.5)));replay=rr.sample(de,n)+rr.sample(en,min(len(en),max(1,int(n*2/3))));rr.shuffle(replay)
    sampler=TrainingProblemSampler();source_tokens=int(payload.get("tokens",payload.get("source_cumulative_tokens",0)+payload.get("stage_b_tokens",0)))
    opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),weight_decay=.08,fused=True)
    baseline_language=validate(model,val);baseline_math=dev_math_nll(model,device,args.seq)
    best_math=baseline_math["nll"];best_language=baseline_language["macro"];bad_regression=0;done=0;step=0
    run={"stage":"rawmath_v1","parameters":EXPECTED_PARAMS,"source":str(Path(args.init_from).resolve()),"source_sha256":sha(args.init_from),"source_cumulative_tokens":source_tokens,"suite":validate_frozen_suite(),"solve_used":False,"tool_use":False,"selection":"lowest held-out RawMath dev answer NLL subject to <=5% general-language validation regression","math_fraction":args.math_fraction,"language_replay_fraction":1-args.math_fraction,"language_replay_mix":"approximately 60% de / 40% en document-bounded blocks","answer_only_math_loss":True,"math_eos_in_loss":True,"args":vars(args),"baseline_language":baseline_language,"baseline_math_dev":baseline_math,"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__)}
    (out/"run.json").write_text(json.dumps(run,indent=2),encoding="utf8")
    def checkpoint():
        return {"model":model.state_dict(),"config":FIFTY.to_dict(),"optimizer":opt.state_dict(),"tokens":source_tokens+done,"source_cumulative_tokens":source_tokens,"stage_c_tokens":done,"step":step,"best_math_dev_nll":best_math,"baseline_language_macro":baseline_language["macro"],"stage_c":"rawmath_v1"}
    # Baseline remains selectable: specialization is never allowed to force a worse checkpoint.
    atomic_save(checkpoint(),out/"best_math.pt");atomic_save(checkpoint(),out/"latest.pt")
    start=time.perf_counter();update_seconds=0.0;math_sequences=language_sequences=math_target_tokens=0;stop="token_budget";torch.cuda.reset_peak_memory_stats()
    try:
        while done+args.seq*args.batch<=args.tokens:
            progress=done/max(1,args.tokens);lr=cosine(progress,args.peak_lr,args.min_lr)
            for g in opt.param_groups:g["lr"]=lr
            x,y,nm,nl,mt=mixed_batch(streams,replay,sampler,args.batch,args.math_fraction,args.seq,rr,device)
            torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);v=answer_only_loss(model,x,y)
            if not torch.isfinite(v):raise FloatingPointError("Non-finite Stage-C loss")
            v.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();dt=time.perf_counter()-t
            step+=1;inc=args.seq*args.batch;done+=inc;update_seconds+=dt;math_sequences+=nm;language_sequences+=nl;math_target_tokens+=mt
            row={"step":step,"stage_c_tokens":done,"cumulative_tokens":source_tokens+done,"train_loss":float(v),"lr":lr,"grad_norm":float(gn),"seconds":dt,"tokens_per_second":inc/max(dt,1e-9),"math_sequences":math_sequences,"language_replay_sequences":language_sequences,"math_target_tokens":math_target_tokens}
            if step%args.eval_every==0 or done+inc>args.tokens:
                m=dev_math_nll(model,device,args.seq);l=validate(model,val);row["math_dev"]=m;row["language_validation"]=l
                eligible=l["macro"]<=baseline_language["macro"]*1.05
                improved=eligible and m["nll"]<best_math-.003
                row["eligible"]=eligible;row["selected"]=improved
                if improved:
                    best_math=m["nll"];best_language=l["macro"];bad_regression=0;atomic_save(checkpoint(),out/"best_math.pt")
                elif l["macro"]>baseline_language["macro"]*1.10:bad_regression+=1
                else:bad_regression=0
                row["language_regression_evals"]=bad_regression;atomic_save(checkpoint(),out/"latest.pt");print(json.dumps(row),flush=True)
                if bad_regression>=3:stop="language_regression_abort";break
            with (out/"metrics.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
    finally:
        atomic_save(checkpoint(),out/"latest.pt")
        summary={"stop":stop,"complete":stop=="token_budget" and done+args.seq*args.batch>args.tokens,"stage_c_tokens":done,"source_cumulative_tokens":source_tokens,"cumulative_tokens":source_tokens+done,"math_sequences":math_sequences,"language_replay_sequences":language_sequences,"math_target_tokens":math_target_tokens,"best_math_dev_nll":best_math,"selected_language_macro":best_language,"baseline_language_macro":baseline_language["macro"],"optimizer_seconds":update_seconds,"optimizer_tokens_per_second":done/max(update_seconds,1e-9),"elapsed_seconds":time.perf_counter()-start,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),"best_sha256":sha(out/"best_math.pt"),"latest_sha256":sha(out/"latest.pt"),"solve_used":False}
        (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8");print(json.dumps(summary),flush=True)
    if stop!="token_budget":raise RuntimeError(f"Stage C stopped: {stop}")

if __name__=="__main__":main()
