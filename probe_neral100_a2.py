from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from architectures100_variants import HELIX_A2, report
from model100 import Neral100M

def sync(): torch.cuda.synchronize()
def timed(fn,warmup=1,repeats=3):
    for _ in range(warmup): fn()
    sync(); vals=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); sync(); vals.append(time.perf_counter()-t)
    return min(vals),sum(vals)/len(vals)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='NERAL100_A2_GPU_PROBE.json');a=ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA required')
    device=torch.device('cuda');torch.set_float32_matmul_precision('high')
    model=Neral100M(HELIX_A2).to(device=device,dtype=torch.bfloat16).eval()
    r={'architecture':report('a2'),'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'training_performed':False,'weights_saved':False,'prefill':{}}
    for seq in (256,512,1024):
        x=torch.randint(0,260,(1,seq),device=device);torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): best,mean=timed(lambda:model(x)[0])
        r['prefill'][str(seq)]={'best_seconds':best,'mean_seconds':mean,'best_tokens_per_second':seq/best,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    prompt=torch.randint(0,260,(1,512),device=device);torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits,cache=model.forward_cached(prompt);sync();t=time.perf_counter();tok=logits[:,-1].argmax(-1,keepdim=True)
        for _ in range(24): logits,cache=model.forward_cached(tok,cache);tok=logits[:,-1].argmax(-1,keepdim=True)
        sync();seconds=time.perf_counter()-t
    r['decode']={'steps':24,'tokens_per_second':24/seconds,'seconds':seconds,'cache_heads':cache[0][0].size(1),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    model.train();model.set_gradient_checkpointing(True);x=torch.randint(0,260,(1,512),device=device);y=torch.randint(0,260,(1,512),device=device);model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();sync();t=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16): _,loss=model(x,y)
    loss.backward();sync();seconds=time.perf_counter()-t
    r['backward_checkpointed']={'sequence_length':512,'loss':float(loss.detach()),'finite':bool(torch.isfinite(loss)),'seconds':seconds,'tokens_per_second':512/seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
