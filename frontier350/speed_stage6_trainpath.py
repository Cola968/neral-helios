from __future__ import annotations

"""Stage-6 speed-only sweep for Arctus 350M.

No checkpoint is loaded or written. The locked 349,825,152-parameter fused model
and exact 16,384 tokens/update contract are preserved. This stage isolates compiler
mode, larger microbatch, and host prefetch effects before any production promotion.
"""
import argparse, gc, json, math, queue, random, threading, time
from pathlib import Path
import numpy as np
import torch
from model350 import FRONTIER_350
from model350_fast import Neral350MFast

PARAMETERS=349_825_152
TOKENS_PER_UPDATE=16_384
STAGE5_BEST=20_585.0

class MMapPool:
    def __init__(self, root: Path):
        ps=sorted(root.glob('*.bin'))
        if not ps: raise FileNotFoundError(root)
        self.arr=[np.memmap(p,dtype=np.uint16,mode='r') for p in ps]
    def cpu_batch(self,b,s,rng):
        rows=np.empty((b,s+1),dtype=np.int64)
        for i in range(b):
            a=rng.choice(self.arr); st=rng.randrange(0,len(a)-s-1); rows[i]=a[st:st+s+1]
        if rows.max(initial=0)>=FRONTIER_350.vocab_size: raise ValueError('token outside vocab')
        return torch.from_numpy(rows).pin_memory()
    def batch(self,b,s,rng,device):
        z=self.cpu_batch(b,s,rng).to(device,non_blocking=True); return z[:,:-1],z[:,1:]

class PrefetchPool:
    def __init__(self,base,b,s,rng,depth=4):
        self.base,self.b,self.s,self.rng=base,b,s,rng; self.q=queue.Queue(depth); self.stop=False
        self.t=threading.Thread(target=self._work,daemon=True); self.t.start()
    def _work(self):
        while not self.stop:
            try:self.q.put(self.base.cpu_batch(self.b,self.s,self.rng),timeout=.1)
            except queue.Full: pass
    def batch(self,b,s,rng,device):
        z=self.q.get().to(device,non_blocking=True); return z[:,:-1],z[:,1:]
    def close(self): self.stop=True; self.t.join(timeout=1)

def accum(mb,seq):
    n=mb*seq
    if TOKENS_PER_UPDATE%n: raise ValueError('non exact accumulation')
    return TOKENS_PER_UPDATE//n

def updates(model,raw,opt,pool,mb,seq,ga,n,rng,dev):
    ls=torch.zeros((),device=dev); gn=torch.zeros((),device=dev)
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        for _ in range(ga):
            x,y=pool.batch(mb,seq,rng,dev); _,loss=model(x,y); (loss/ga).backward(); ls.add_(loss.detach().float())
        gn=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.0); opt.step()
    return ls,gn

def case(base,mb,mode,prefetch,seq,warm,n):
    torch.cuda.empty_cache(); gc.collect(); ga=accum(mb,seq); seed=36000+mb+(100 if prefetch else 0)+sum(map(ord,mode))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng=random.Random(seed); dev=torch.device('cuda')
    raw=Neral350MFast().to(dev,dtype=torch.bfloat16); raw.set_gradient_checkpointing(False)
    assert raw.parameter_report()['total']==PARAMETERS
    pool=PrefetchPool(base,mb,seq,rng,4) if prefetch else base
    try:
        model=torch.compile(raw,backend='inductor',mode=mode,fullgraph=True,dynamic=False)
        opt=torch.optim.AdamW(raw.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=True)
        torch.cuda.reset_peak_memory_stats(); wl,wg=updates(model,raw,opt,pool,mb,seq,ga,warm,rng,dev); torch.cuda.synchronize()
        if not math.isfinite(float(wl)) or not math.isfinite(float(wg)): raise RuntimeError('nonfinite warmup')
        torch.cuda.synchronize(); t=time.perf_counter(); ml,mg=updates(model,raw,opt,pool,mb,seq,ga,n,rng,dev); torch.cuda.synchronize(); sec=time.perf_counter()-t
        lv=float(ml/(n*ga)); gv=float(mg); tps=TOKENS_PER_UPDATE*n/sec
        return {'status':'ok','microbatch':mb,'grad_accum':ga,'compile_mode':mode,'prefetch':prefetch,'fullgraph':True,'dynamic':False,'e2e_tokens_per_second':tps,'vs_stage5_pct':(tps/STAGE5_BEST-1)*100,'loss':lv,'last_grad_norm':gv,'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gb':torch.cuda.max_memory_reserved()/2**30,'seconds':sec}
    except torch.cuda.OutOfMemoryError as e:
        return {'status':'oom','microbatch':mb,'grad_accum':ga,'compile_mode':mode,'prefetch':prefetch,'error':type(e).__name__}
    except Exception as e:
        return {'status':'failed','microbatch':mb,'grad_accum':ga,'compile_mode':mode,'prefetch':prefetch,'error':f'{type(e).__name__}:{e}'[:500]}
    finally:
        if prefetch: pool.close()
        try: del model,opt
        except: pass
        del raw; torch.cuda.empty_cache(); gc.collect()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seq',type=int,default=512); ap.add_argument('--updates',type=int,default=12); ap.add_argument('--warmup',type=int,default=4); a=ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise SystemExit('CUDA BF16 required')
    torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_math_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True); torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision('high')
    base=MMapPool(Path(a.train_data)); specs=[(8,'default',False),(8,'default',True),(8,'reduce-overhead',False),(8,'reduce-overhead',True),(16,'default',False),(16,'reduce-overhead',False),(8,'max-autotune',False)]
    rows=[]
    for s in specs:
        r=case(base,*s,a.seq,a.warmup,a.updates); rows.append(r); print(json.dumps(r),flush=True)
    ok=[r for r in rows if r['status']=='ok']; best=max(ok,key=lambda r:r['e2e_tokens_per_second']) if ok else None
    rep={'architecture':'frontier350-v1','parameters':PARAMETERS,'stage5_best_tps':STAGE5_BEST,'cases':rows,'best':best,'promotion_allowed':False,'measured_facts':'GPU timings above are measured end-to-end over optimizer updates; no checkpoint touched.','estimates':'None. Promotion requires checkpoint numerical + validation A/B.'}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(rep,indent=2),encoding='utf8'); print(json.dumps(rep),flush=True)
if __name__=='__main__': main()
