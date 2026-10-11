from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from chat350_contract import EXPECTED_TOKENIZER_SHA256, validate_checkpoint_metadata
from chat350_data import validate_dataset_root
from chat350_sft_format import encode_messages
from model350 import FRONTIER_350, Neral350M
from sft_batching350 import length_bucket_batches, padding_stats
from sft_stratified350 import stratified_batches, cohort_summary

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)

def load_tokenizer(path: Path) -> Tokenizer:
    if sha256_file(path)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("locked 16K tokenizer digest mismatch")
    tok=Tokenizer.from_file(str(path))
    if tok.get_vocab_size()!=FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    expected={"<|bos|>":0,"<|eos|>":1,"<|sep|>":2,"<|pad|>":3}
    got={k:tok.token_to_id(k) for k in expected}
    if got!=expected: raise ValueError(f"special-token mapping mismatch: {got}")
    return tok

def list_shards(root: Path) -> list[Path]:
    shards=sorted(Path(root).glob("*.bin"))
    if not shards: raise FileNotFoundError(f"no replay shards in {root}")
    return shards

def replay_batch(shards, batch, seq, device, rng, vocab):
    xs=[]; ys=[]
    for _ in range(batch):
        p=rng.choice(shards); a=np.memmap(p,dtype=np.uint16,mode="r")
        if len(a)<seq+1: raise ValueError(f"replay shard too short: {p}")
        s=rng.randrange(0,len(a)-seq)
        z=np.asarray(a[s:s+seq+1],dtype=np.int64)
        if z.max(initial=0)>=vocab: raise ValueError(f"replay token outside vocabulary: {p}")
        xs.append(torch.from_numpy(z[:-1].copy())); ys.append(torch.from_numpy(z[1:].copy()))
    return torch.stack(xs).to(device,non_blocking=True),torch.stack(ys).to(device,non_blocking=True)

def collate(encoded, pad_id, device):
    width=max(len(x.input_ids) for x in encoded)
    x=torch.full((len(encoded),width),pad_id,dtype=torch.long)
    y=torch.full((len(encoded),width),-100,dtype=torch.long)
    for i,row in enumerate(encoded):
        x[i,:len(row.input_ids)]=torch.tensor(row.input_ids,dtype=torch.long)
        y[i,:len(row.targets)]=torch.tensor(row.targets,dtype=torch.long)
    return x.to(device,non_blocking=True),y.to(device,non_blocking=True)

def assistant_loss(model,x,y,eos_id,eos_weight):
    logits,_=model(x)
    flat_y=y.reshape(-1); mask=flat_y!=-100
    if not bool(mask.any()): raise ValueError("batch has no assistant targets")
    selected=logits.reshape(-1,logits.size(-1))[mask].float()
    targets=flat_y[mask]
    per=F.cross_entropy(selected,targets,reduction="none")
    w=torch.ones_like(per); w[targets==eos_id]=eos_weight
    return (per*w).sum()/w.sum().clamp_min(1.0)

@torch.inference_mode()
def eval_sft(model,encoded,batch_size,pad_id,eos_id,device):
    model.eval(); nll=0.0; tokens=0; eos_ok=0; eos_total=0
    for i in range(0,len(encoded),batch_size):
        x,y=collate(encoded[i:i+batch_size],pad_id,device)
        logits,_=model(x); fy=y.reshape(-1); mask=fy!=-100
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1))[mask].float(),fy[mask],reduction="sum")
        nll+=float(loss); tokens+=int(mask.sum())
        em=y==eos_id
        if bool(em.any()):
            pred=logits.argmax(-1); eos_ok+=int((pred[em]==eos_id).sum()); eos_total+=int(em.sum())
    model.train()
    return {"answer_nll":nll/max(1,tokens),"answer_tokens":tokens,
            "terminal_eos_top1_pct":100.0*eos_ok/max(1,eos_total)}

@torch.inference_mode()
def eval_replay(model,shards,batch,seq,batches,device,seed):
    rng=random.Random(seed); model.eval(); vals=[]
    for _ in range(batches):
        x,y=replay_batch(shards,batch,seq,device,rng,model.cfg.vocab_size)
        _,loss=model(x,y); vals.append(float(loss))
    model.train(); return sum(vals)/len(vals)

def lr_at(step,total,peak,floor,warm_frac):
    warm=max(1,round(total*warm_frac))
    if step<=warm: return peak*step/warm
    p=min(1.0,max(0.0,(step-warm)/max(1,total-warm)))
    return floor+(peak-floor)*0.5*(1+math.cos(math.pi*p))

def shuffled_batches(rows,batch,rng):
    order=list(range(len(rows))); rng.shuffle(order); cursor=0
    while True:
        picked=[]
        while len(picked)<batch:
            if cursor>=len(order): rng.shuffle(order); cursor=0
            take=min(batch-len(picked),len(order)-cursor)
            picked.extend(rows[order[j]] for j in range(cursor,cursor+take)); cursor+=take
        yield picked

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--init-from",required=True); ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--data",required=True); ap.add_argument("--replay-data",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--steps",type=int,default=300); ap.add_argument("--seq",type=int,default=512)
    ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--replay-batch",type=int,default=2)
    ap.add_argument("--replay-seq",type=int,default=512); ap.add_argument("--replay-every",type=int,default=4)
    ap.add_argument("--sft-weight",type=float,default=.90); ap.add_argument("--eos-weight",type=float,default=1.5)
    ap.add_argument("--peak-lr",type=float,default=2e-5); ap.add_argument("--min-lr",type=float,default=2e-6)
    ap.add_argument("--warmup-fraction",type=float,default=.05); ap.add_argument("--eval-every",type=int,default=50)
    ap.add_argument("--save-every",type=int,default=50); ap.add_argument("--eval-replay-batches",type=int,default=8)
    ap.add_argument("--seed",type=int,default=350_726); ap.add_argument("--gradient-checkpointing",action="store_true"); ap.add_argument("--length-bucket-size",type=int,default=64)
    ap.add_argument("--sampling-mode",choices=["bucketed","human-balanced"],default="bucketed")
    ap.add_argument("--human-per-batch",type=int,default=2)
    args=ap.parse_args()
    if args.seq>FRONTIER_350.max_seq_len or args.replay_seq>FRONTIER_350.max_seq_len: raise ValueError("sequence exceeds architecture")
    if min(args.steps,args.batch_size,args.replay_batch,args.replay_every,args.eval_every,args.save_every)<=0: raise ValueError("positive budgets required")
    if args.length_bucket_size<args.batch_size or args.length_bucket_size%args.batch_size: raise ValueError("length-bucket-size must be >= batch-size and divisible by batch-size")
    if not 0.5<=args.sft_weight<=1.0: raise ValueError("sft-weight must be 0.5..1")
    if not 1.0<=args.eos_weight<=3.0: raise ValueError("eos-weight must be 1..3")
    if not 0<args.min_lr<=args.peak_lr: raise ValueError("invalid learning rates")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")

    root=Path(args.data); manifest,splits=validate_dataset_root(root)
    # Exact conversation de-duplication alone misses identical prompts with
    # different answers across train and evaluation splits.
    from chat350_data import validate_prompt_holdouts
    prompt_holdout_audit = validate_prompt_holdouts(splits)
    tok=load_tokenizer(Path(args.tokenizer))
    train=[encode_messages(tok,r["messages"],max_seq_len=args.seq) for r in splits["train"]]
    train_is_human=[r["source"].startswith("oasst2-human") for r in splits["train"]]
    mixture=cohort_summary(train_is_human)
    if args.sampling_mode=="human-balanced" and not (0<args.human_per_batch<args.batch_size):
        raise ValueError("human-per-batch must be between 1 and batch-size-1")
    val=[encode_messages(tok,r["messages"],max_seq_len=args.seq) for r in splits["val"]]
    test=[encode_messages(tok,r["messages"],max_seq_len=args.seq) for r in splits["test"]]
    pad_id=int(tok.token_to_id("<|pad|>")); eos_id=int(tok.token_to_id("<|eos|>"))
    replay=list_shards(Path(args.replay_data))

    source_path=Path(args.init_from); source_sha=sha256_file(source_path)
    payload=torch.load(source_path,map_location="cpu",weights_only=True)
    validate_checkpoint_metadata(payload,min_processed_tokens=1_000_000_000)
    source_tokens=int(payload.get("processed_tokens",payload.get("source_processed_tokens",payload.get("tokens",0))))
    device=torch.device("cuda"); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.set_float32_matmul_precision("high")
    model=Neral350M().to(device=device,dtype=torch.bfloat16)
    model.load_state_dict(payload["model"],strict=True); del payload
    model.set_gradient_checkpointing(args.gradient_checkpointing); model.train()
    if model.parameter_report()["total"]!=349_825_152: raise RuntimeError("unexpected parameter count")

    baseline_sft=eval_sft(model,val,args.batch_size,pad_id,eos_id,device)
    baseline_replay=eval_replay(model,replay,args.replay_batch,args.replay_seq,args.eval_replay_batches,device,args.seed+91)
    opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),eps=1e-8,weight_decay=.03,fused=True)
    rng=random.Random(args.seed+1)
    batches=(stratified_batches(train,train_is_human,args.batch_size,rng,
              human_per_batch=args.human_per_batch,bucket_size=args.length_bucket_size)
              if args.sampling_mode=="human-balanced"
              else length_bucket_batches(train,args.batch_size,rng,bucket_size=args.length_bucket_size))
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("chat output directory must be fresh")
    out.mkdir(parents=True,exist_ok=True)
    run={"stage":"Arctus 350M chat SFT pilot","source_checkpoint_sha256":source_sha,"source_processed_tokens":source_tokens,
         "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"data_manifest":manifest,"baseline_sft":baseline_sft,
         "baseline_replay_loss":baseline_replay,"fresh_optimizer":True,"assistant_only":True,"args":vars(args),
         "gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),
         "sampling":{**mixture,"mode":args.sampling_mode,"human_per_batch":args.human_per_batch}}
    (out/"run.json").write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding="utf8")

    def model_payload(step, val_report=None):
        return {"model":model.state_dict(),"config":FRONTIER_350.to_dict(),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
                "source_checkpoint_sha256":source_sha,"source_processed_tokens":source_tokens,
                "chat_step":step,"data_manifest_sha256":manifest["manifest_sha256"],"validation":val_report}

    best=baseline_sft["answer_nll"]; best_eos=baseline_sft["terminal_eos_top1_pct"]
    start=time.perf_counter(); measured_tokens=0; update_seconds=0.0; total_sft_real=0; total_sft_padded=0; sampled_human=0; sampled_verified=0; torch.cuda.reset_peak_memory_stats()
    metrics=out/"metrics.jsonl"
    for step in range(1,args.steps+1):
        lr=lr_at(step,args.steps,args.peak_lr,args.min_lr,args.warmup_fraction)
        for g in opt.param_groups:g["lr"]=lr
        batch_rows=next(batches); pad=padding_stats(batch_rows); x,y=collate(batch_rows,pad_id,device); do_replay=step%args.replay_every==0
        total_sft_real+=pad["real_tokens"]; total_sft_padded+=pad["padded_tokens"]
        if args.sampling_mode=="human-balanced":
            sampled_human+=args.human_per_batch
            sampled_verified+=args.batch_size-args.human_per_batch
        opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); tick=time.perf_counter()
        sft=assistant_loss(model,x,y,eos_id,args.eos_weight)
        (sft*(args.sft_weight if do_replay else 1.0)).backward()
        replay_loss=None; compute_tokens=x.numel()
        if do_replay:
            rx,ry=replay_batch(replay,args.replay_batch,args.replay_seq,device,rng,model.cfg.vocab_size)
            _,rl=model(rx,ry); replay_loss=rl
            (rl*(1.0-args.sft_weight)).backward(); compute_tokens+=rx.numel()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize(); dt=time.perf_counter()-tick
        update_seconds+=dt; measured_tokens+=compute_tokens
        row={"step":step,"sft_loss":float(sft.detach()),"replay_loss":None if replay_loss is None else float(replay_loss.detach()),
             "lr":lr,"grad_norm":float(gn),"seconds":dt,"compute_tokens":compute_tokens,"tokens_per_second":compute_tokens/max(dt,1e-9),
             "sft_batch_width":pad["batch_width"],"sft_real_tokens":pad["real_tokens"],"sft_padded_tokens":pad["padded_tokens"],
             "sft_padding_tokens":pad["padding_tokens"],"sft_padding_fraction":pad["padding_fraction"],
             "sampling_mode":args.sampling_mode}
        if step%args.eval_every==0 or step==args.steps:
            vr=eval_sft(model,val,args.batch_size,pad_id,eos_id,device)
            rr=eval_replay(model,replay,args.replay_batch,args.replay_seq,args.eval_replay_batches,device,args.seed+91)
            replay_ok=rr<=baseline_replay*1.05
            improved=vr["answer_nll"]<best-0.002 and vr["terminal_eos_top1_pct"]>=best_eos and replay_ok
            row.update(validation=vr,replay_validation_loss=rr,replay_guard=replay_ok,selected=improved)
            if improved:
                best=vr["answer_nll"]; best_eos=vr["terminal_eos_top1_pct"]
                atomic_save(model_payload(step,vr),out/"best_chat.pt")
        if step%args.save_every==0 or step==args.steps:
            atomic_save({**model_payload(step,row.get("validation")),"optimizer":opt.state_dict(),
                         "rng_state":rng.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},
                        out/"latest_resume.pt")
        with metrics.open("a",encoding="utf8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        if step%10==0 or step==args.steps: print(json.dumps(row,ensure_ascii=False),flush=True)

    final_sft=eval_sft(model,val,args.batch_size,pad_id,eos_id,device)
    test_sft=eval_sft(model,test,args.batch_size,pad_id,eos_id,device)
    final_replay=eval_replay(model,replay,args.replay_batch,args.replay_seq,args.eval_replay_batches,device,args.seed+91)
    summary={"complete":True,"steps":args.steps,"source_checkpoint_sha256":source_sha,"source_processed_tokens":source_tokens,
             "baseline_sft":baseline_sft,"final_sft":final_sft,"test_sft":test_sft,"baseline_replay_loss":baseline_replay,
             "final_replay_loss":final_replay,"replay_regression_pct":100*(final_replay/baseline_replay-1),
             "best_answer_nll":best,"best_eos_top1_pct":best_eos,"optimizer_seconds":update_seconds,
             "mean_optimizer_tokens_per_second":measured_tokens/max(update_seconds,1e-9),
             "sft_real_tokens":total_sft_real,"sft_padded_tokens":total_sft_padded,"sft_padding_fraction":1.0-total_sft_real/max(1,total_sft_padded),
             "sampled_human_rows":sampled_human,"sampled_verified_rows":sampled_verified,"sampling_mode":args.sampling_mode,
             "elapsed_seconds":time.perf_counter()-start,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
             "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"best_checkpoint_exists":(out/"best_chat.pt").is_file()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
