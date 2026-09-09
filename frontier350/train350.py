from __future__ import annotations

"""Resumable Neral 350M Frontier trainer.

Design goals:
- local/self-hosted RTX training
- exact token accounting toward a 10B curriculum
- BF16 + AdamW + cosine decay + warmup
- gradient accumulation and activation checkpointing
- atomic checkpoints with optimizer/scheduler/RNG state
- JSONL live metrics consumable by Trackio bridge
- periodic validation and fail-fast numerical checks

Dataset input is a directory containing uint16 token shards (*.bin). Each shard must
contain byte-token IDs in [0, vocab_size). Benchmark holdouts must never be present.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch

from model350 import Neral350M, FRONTIER_350


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()


def list_shards(root: Path):
    shards = sorted(root.glob('*.bin'))
    if not shards:
        raise FileNotFoundError(f'no *.bin shards in {root}')
    return shards


def sample_batch(shards, batch, seq, device, rng):
    rows=[]
    targets=[]
    for _ in range(batch):
        p=rng.choice(shards)
        a=np.memmap(p,dtype=np.uint16,mode='r')
        if len(a) < seq+1:
            raise ValueError(f'shard too short: {p}')
        start=rng.randrange(0,len(a)-seq-1)
        chunk=np.asarray(a[start:start+seq+1],dtype=np.int64)
        if chunk.max(initial=0) >= FRONTIER_350.vocab_size:
            raise ValueError(f'token id outside vocabulary in {p}')
        rows.append(torch.from_numpy(chunk[:-1].copy()))
        targets.append(torch.from_numpy(chunk[1:].copy()))
    return torch.stack(rows).to(device,non_blocking=True), torch.stack(targets).to(device,non_blocking=True)


def cosine_lr(step,warmup,total,peak,min_ratio=0.08):
    if step < warmup:
        return peak * (step+1)/max(1,warmup)
    u=min(1.0,(step-warmup)/max(1,total-warmup))
    return peak*(min_ratio+(1-min_ratio)*0.5*(1+math.cos(math.pi*u)))


def evaluate(model, shards, seq, batch, batches, device, seed):
    rng=random.Random(seed)
    model.eval(); vals=[]
    with torch.no_grad():
        for _ in range(batches):
            x,y=sample_batch(shards,batch,seq,device,rng)
            _,loss=model(x,y)
            vals.append(float(loss))
    model.train()
    return sum(vals)/len(vals)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-data',required=True)
    ap.add_argument('--valid-data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--target-tokens',type=int,default=10_000_000_000)
    ap.add_argument('--seq',type=int,default=1024)
    ap.add_argument('--microbatch',type=int,default=2)
    ap.add_argument('--grad-accum',type=int,default=16)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--weight-decay',type=float,default=0.1)
    ap.add_argument('--warmup-frac',type=float,default=0.02)
    ap.add_argument('--eval-every',type=int,default=250)
    ap.add_argument('--save-every',type=int,default=500)
    ap.add_argument('--log-every',type=int,default=10)
    ap.add_argument('--eval-batches',type=int,default=16)
    ap.add_argument('--seed',type=int,default=35001)
    ap.add_argument('--resume',default='')
    args=ap.parse_args()

    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16-capable GPU required')
    device=torch.device('cuda')
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    metrics=out/'metrics.jsonl'
    train=list_shards(Path(args.train_data)); valid=list_shards(Path(args.valid_data))
    rng=random.Random(args.seed+1)

    model=Neral350M().to(device=device,dtype=torch.bfloat16)
    model.set_gradient_checkpointing(True)
    report=model.parameter_report()
    if not (330_000_000 <= report['total'] <= 370_000_000):
        raise RuntimeError(f'Frontier parameter budget violated: {report}')
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95),eps=1e-8,weight_decay=args.weight_decay,fused=True)

    tokens_per_update=args.seq*args.microbatch*args.grad_accum
    total_updates=math.ceil(args.target_tokens/tokens_per_update)
    warmup=max(1,int(total_updates*args.warmup_frac))
    step=0; processed=0; best=float('inf')

    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        step=int(ck['step']); processed=int(ck['processed_tokens']); best=float(ck.get('best_validation',best))
        torch.set_rng_state(ck['torch_rng']); random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        if torch.cuda.is_available() and 'cuda_rng' in ck: torch.cuda.set_rng_state_all(ck['cuda_rng'])

    model.train(); start=time.time(); last=time.time(); last_tokens=processed
    while processed < args.target_tokens:
        lr=cosine_lr(step,warmup,total_updates,args.lr)
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True); running=0.0
        for _ in range(args.grad_accum):
            x,y=sample_batch(train,args.microbatch,args.seq,device,rng)
            _,loss=model(x,y)
            if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss at step {step}')
            (loss/args.grad_accum).backward(); running += float(loss.detach())
            processed += x.numel()
            if processed >= args.target_tokens: break
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        if not math.isfinite(grad_norm): raise RuntimeError(f'non-finite grad norm at step {step}')
        opt.step(); step+=1

        if step % args.log_every == 0:
            now=time.time(); tps=(processed-last_tokens)/max(1e-9,now-last)
            row={'event':'train','step':step,'processed_tokens':processed,'target_tokens':args.target_tokens,'progress':processed/args.target_tokens,'loss':running/max(1,args.grad_accum),'lr':lr,'grad_norm':grad_norm,'tokens_per_second':tps,'elapsed_s':now-start,'gpu_mem_allocated_gb':torch.cuda.memory_allocated()/2**30,'gpu_mem_reserved_gb':torch.cuda.memory_reserved()/2**30}
            print(json.dumps(row),flush=True); append_jsonl(metrics,row); last=now; last_tokens=processed

        if step % args.eval_every == 0 or processed >= args.target_tokens:
            val=evaluate(model,valid,args.seq,max(1,args.microbatch),args.eval_batches,device,args.seed+999+step)
            ev={'event':'validation','step':step,'processed_tokens':processed,'validation_loss':val,'best_validation_loss':min(best,val)}
            print(json.dumps(ev),flush=True); append_jsonl(metrics,ev)
            if val < best:
                best=val
                atomic_save({'model':model.state_dict(),'step':step,'processed_tokens':processed,'validation_loss':val,'config':FRONTIER_350.to_dict()},out/'best_validation.pt')

        if step % args.save_every == 0 or processed >= args.target_tokens:
            state={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'processed_tokens':processed,'best_validation':best,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'config':FRONTIER_350.to_dict(),'target_tokens':args.target_tokens}
            atomic_save(state,out/'resume.pt')

    summary={'complete':True,'parameters':report,'processed_tokens':processed,'target_tokens':args.target_tokens,'steps':step,'best_validation_loss':best,'elapsed_s':time.time()-start}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
