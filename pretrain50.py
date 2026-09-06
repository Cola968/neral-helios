"""Token-budgeted, packed language pretraining pilot; no tokenizer in hot path."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from architectures import FIFTY
from model import Neral20M
from model50 import Neral50M
from experiment import atomic_save,sha,load_model

def streams(split):return {lang:np.memmap(f'pretrain50_data/{split}_{lang}.bin',dtype='<u2',mode='r') for lang in ['de','en']}
def batch(data,lang,b,s,rng):
    a=data[lang];starts=rng.integers(0,len(a)-s-1,size=b)
    z=np.stack([a[i:i+s+1].astype(np.int64) for i in starts]);t=torch.from_numpy(z).to('cuda');return t[:,:-1],t[:,1:]
def loss(model,x,y):
    with torch.autocast('cuda',dtype=torch.bfloat16):return model(x,y)[1]
@torch.inference_mode()
def validate(model,data,sequence=512,batches=12):
    model.eval();rng=np.random.default_rng(50013);result={}
    for lang in ['de','en']:
        total=0
        for _ in range(batches):x,y=batch(data,lang,8,sequence,rng);total+=float(loss(model,x,y))
        result[lang]=total/batches
    model.train();return {'domains':result,'macro':sum(result.values())/2,'bits_per_token':sum(result.values())/2/math.log(2),'tokens':batches*8*sequence*2,'note':'Fixed validation slices. Includes BOS/EOS, hence bits/token, not pure byte BPB.'}
def init():
    torch.manual_seed(50013);m=Neral50M(FIFTY);assert m.parameter_report()['trainable']==50_012_976
    target=Path('seed/neral50m.pt')
    if not target.exists():atomic_save({'model':m.state_dict(),'config':FIFTY.to_dict(),'step':0,'initialization_seed':50013,'source':'Own Neral architecture, new random initialization; no external weights'},target)
    return m.to('cuda')
def tune():
    data=streams('train');results=[]
    for seq,bs in [(256,16),(256,32),(256,64),(512,8),(512,16),(512,32),(1024,4),(1024,8),(1024,16)]:
        model=init();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True);rng=np.random.default_rng(51);torch.cuda.reset_peak_memory_stats()
        try:
            samples=[]
            for i in range(5):
                torch.cuda.synchronize();t=time.perf_counter();x,y=batch(data,'de',bs,seq,rng);opt.zero_grad(set_to_none=True);v=loss(model,x,y);v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();torch.cuda.synchronize()
                if not torch.isfinite(v):raise FloatingPointError('Nonfinite benchmark loss')
                if i>=2:samples.append(time.perf_counter()-t)
            r={'seq':seq,'batch':bs,'tokens_per_step':seq*bs,'seconds':sum(samples)/len(samples),'tokens_per_second':seq*bs*len(samples)/sum(samples),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'oom':False,'last_loss':float(v)}
        except torch.cuda.OutOfMemoryError:r={'seq':seq,'batch':bs,'oom':True}
        results.append(r);print(json.dumps(r),flush=True);del opt,model;torch.cuda.empty_cache()
    usable=[r for r in results if not r['oom'] and r['peak_reserved']<9.5*2**30];best=max(usable,key=lambda r:r['tokens_per_second'])
    model=init();x,y=batch(data,'de',best['batch'],best['seq'],np.random.default_rng(51))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:loss(model,x,y).backward()
    kernels=sorted({e.key for e in prof.key_averages() if 'attention' in e.key.lower()})
    report={'parameters':50012976,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'candidates':results,'selected':best,'attention_operators':kernels,'optimizations':{'packing':'enabled: 0 PAD tokens','pretokens':'uint16 memmap','BF16':True,'fused_adamw':True,'compile':'not enabled or benchmarked in this bounded Windows pilot; no speed claim','pinned_memory':'not enabled: not benchmarked','gradient_checkpointing':'not needed in measured memory budget'},'training_only_estimated_hours':{str(n):n/best['tokens_per_second']/3600 for n in [100_000_000,250_000_000,500_000_000,1_000_000_000]}}
    Path('reports/throughput50.json').write_text(json.dumps(report,indent=2));print(json.dumps({'selected':best,'attention':kernels}),flush=True)
def train(args):
    tuned=json.loads(Path('reports/throughput50.json').read_text())['selected'];seq,bs=tuned['seq'],tuned['batch'];data,val=streams('train'),streams('val');rng=np.random.default_rng(50013);torch.manual_seed(50013)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:raise ValueError('Fresh run directory required')
    out.mkdir(parents=True,exist_ok=True);model,p=load_model(args.resume or 'seed/neral50m.pt',torch.device('cuda'));model.train();opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.1,fused=True)
    hashes={f'{split}_{lang}':sha(f'pretrain50_data/{split}_{lang}.bin') for split in ['train','val'] for lang in ['de','en']}
    done=step=bad=0;best=float('inf')
    if args.resume:
        if p['hashes']!=hashes or p['seq']!=seq or p['batch']!=bs:raise ValueError('Resume corpus or packing mismatch')
        opt.load_state_dict(p['optimizer']);done=p['tokens'];step=p['step'];rng.bit_generator.state=json.loads(p['rng_numpy']);torch.set_rng_state(p['rng_torch']);torch.cuda.set_rng_state_all(p['rng_cuda']);best=p['best'];bad=p['bad']
    baseline=validate(model,val);best=min(best,baseline['macro']);run={'parameters':50012976,'seed':50013,'source':args.resume or 'seed/neral50m.pt','source_sha256':sha(args.resume or 'seed/neral50m.pt'),'data_hashes':hashes,'args':vars(args),'packing':tuned,'baseline':baseline,'promotion':'No default promotion based on loss. Free generation and manual quality gates required.','token_budget_scope':'Target cumulative tokens when resuming.','learning_rate_schedule':'Fixed LR with 10-update warmup, independent of eventual target budget.'}
    (out/'run.json').write_text(json.dumps(run,indent=2));stop='token_budget';start=time.perf_counter();train_seconds=0;counts={'de':0,'en':0};torch.cuda.reset_peak_memory_stats()
    def checkpoint():return {'model':model.state_dict(),'config':FIFTY.to_dict(),'optimizer':opt.state_dict(),'tokens':done,'step':step,'best':best,'bad':bad,'seq':seq,'batch':bs,'hashes':hashes,'rng_numpy':json.dumps(rng.bit_generator.state),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'source_sha256':run['source_sha256']}
    if not args.resume:atomic_save(checkpoint(),out/'best_validation.pt')
    print(json.dumps({'baseline':baseline,'budget':args.tokens}),flush=True)
    try:
        while done+bs*seq<=args.tokens:
            lang='de' if rng.random()<.55 else 'en';x,y=batch(data,lang,bs,seq,rng);torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=args.lr*min(1,(step+1)/10)
            v=loss(model,x,y)
            if not torch.isfinite(v):raise FloatingPointError('Nonfinite pretraining loss')
            v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-tick;train_seconds+=elapsed;step+=1;done+=bs*seq;counts[lang]+=bs*seq
            row={'step':step,'processed_tokens':done,'loss':float(v),'seconds':elapsed,'language':lang,'tokens_per_second':bs*seq/elapsed,'padding_tokens':0}
            if step%args.eval_every==0 or done+bs*seq>args.tokens:
                score=validate(model,val);row['validation']=score
                if score['macro']<best-.005:best=score['macro'];bad=0;atomic_save(checkpoint(),out/'best_validation.pt');row['selected_validation']=True
                else:bad+=1
                print(json.dumps(row),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if bad>=args.patience:stop='early_stopping';break
    except KeyboardInterrupt:stop='interrupted'
    except Exception:
        stop='failed';raise
    finally:
        atomic_save(checkpoint(),out/'latest.pt');summary={'stop':stop,'steps':step,'processed_tokens':done,'this_segment_tokens':sum(counts.values()),'language_tokens':counts,'elapsed_seconds':time.perf_counter()-start,'optimizer_seconds':train_seconds,'optimizer_tokens_per_second':sum(counts.values())/max(.001,train_seconds),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'best_validation':best,'latest_sha256':sha(out/'latest.pt'),'padding_tokens':0,'energy_kwh':None,'cost':None,'cost_note':'No measured integrated power or user electricity tariff; no invented energy/cost estimate.'};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tune',action='store_true');ap.add_argument('--tokens',type=int,default=1_000_000);ap.add_argument('--out',default='runs/neral50m_pilot_a');ap.add_argument('--resume');ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--eval-every',type=int,default=25);ap.add_argument('--patience',type=int,default=6);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Pilot requires measured BF16 CUDA environment')
    if args.tune:tune()
    else:train(args)
if __name__=='__main__':main()
