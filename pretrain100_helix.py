"""From-scratch Neral-100M Helix pretraining skeleton with RTX-4070 OOM fallback."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from architectures100_variants import VARIANTS,EXPECTED
from model100 import Neral100M
from neral100_checkpoint import checkpoint_payload

def batch_from_stream(s,batch,seq,device,rng):
    if len(s)<=seq+1:raise ValueError('stream too short')
    starts=rng.integers(0,len(s)-seq-1,size=batch);x=np.stack([np.asarray(s[i:i+seq],dtype=np.int64) for i in starts]);y=np.stack([np.asarray(s[i+1:i+seq+1],dtype=np.int64) for i in starts]);return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)
def train_once(m,opt,s,seq,microbatch,grad_accum,device,rng):
    opt.zero_grad(set_to_none=True);total=0.;tokens=0
    for _ in range(grad_accum):
        x,y=batch_from_stream(s,microbatch,seq,device,rng)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=m(x);loss=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1))
        if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
        (loss/grad_accum).backward();total+=float(loss.detach());tokens+=x.numel()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();return total/grad_accum,tokens
def fallback_shapes(seq,micro):
    seen=set()
    while (seq,micro) not in seen:
        seen.add((seq,micro));yield seq,micro
        if micro>1:micro=max(1,micro//2)
        elif seq>512:seq=max(512,seq//2)
        else:break
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=sorted(VARIANTS),default='a1');ap.add_argument('--train-bin',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--seq',type=int,default=2048);ap.add_argument('--microbatch',type=int,default=1);ap.add_argument('--grad-accum',type=int,default=16);ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--seed',type=int,default=100968);a=ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA required')
    if min(a.steps,a.seq,a.microbatch,a.grad_accum)<=0:raise ValueError('positive budgets required')
    device=torch.device('cuda');torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_float32_matmul_precision('high');rng=np.random.default_rng(a.seed);stream=np.memmap(a.train_bin,dtype='<u2',mode='r');m=Neral100M(VARIANTS[a.variant]).to(device);m.set_gradient_checkpointing(True)
    if sum(p.numel() for p in m.parameters())!=EXPECTED[a.variant]:raise ValueError('parameter drift')
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1,fused=True);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tokens=0;effective=None;history=[]
    for step in range(1,a.steps+1):
        success=False
        for seq,micro in fallback_shapes(a.seq,a.microbatch):
            try:
                torch.cuda.empty_cache();t=time.perf_counter();loss,n=train_once(m,opt,stream,seq,micro,a.grad_accum,device,rng);tokens+=n;effective=(seq,micro);history.append({'step':step,'loss':loss,'seq':seq,'microbatch':micro,'tokens':tokens,'seconds':time.perf_counter()-t});success=True;break
            except torch.cuda.OutOfMemoryError:opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if not success:raise RuntimeError('100M OOM even at seq=512 microbatch=1')
        if step%100==0 or step==a.steps:print(json.dumps(history[-1]),flush=True)
    torch.save(checkpoint_payload(m,a.variant,tokens=tokens,step=a.steps,optimizer_state=opt.state_dict(),provenance={'train_bin':str(Path(a.train_bin).resolve()),'from_scratch':True,'oom_fallback':True}),out/'latest.pt');(out/'summary.json').write_text(json.dumps({'variant':a.variant,'parameters':EXPECTED[a.variant],'steps':a.steps,'tokens':tokens,'effective_shape':effective,'from_scratch':True,'foreign_weights':False},indent=2),encoding='utf8')
if __name__=='__main__':main()
