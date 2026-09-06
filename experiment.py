"""Reproducible assistant-only SFT of the supplied Neral weights, unchanged architecture."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict
import torch
import torch.nn.functional as F
from config import NeralConfig
from model import Neral20M
from tokenizer import ByteTokenizer
from architectures import expected_parameters

EXPECTED_PARAMS=20_587_008
BASE=Path(__file__).resolve().parent
TOKENIZER=ByteTokenizer()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_save(payload,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def read_records(path):return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
def encode_record(r,max_len=1024):
    prefix=TOKENIZER.encode('User: '+r['prompt'].strip()+'\nAssistant:',bos=True)
    answer=TOKENIZER.encode(' '+r['answer'].strip()+'\n',eos=True)
    ids=prefix+answer
    if len(ids)-1>max_len:raise ValueError(f"Record {r['id']} has {len(ids)} tokens, limit {max_len+1}; no silent truncation")
    targets=[-100]*len(prefix)+answer
    return torch.tensor(ids[:-1]),torch.tensor(targets[1:])
def batch(records,device):
    encoded=[encode_record(r) for r in records];n=max(len(x) for x,y in encoded)
    x=torch.zeros((len(records),n),dtype=torch.long);y=torch.full((len(records),n),-100,dtype=torch.long)
    for i,(a,b) in enumerate(encoded):x[i,:len(a)]=a;y[i,:len(b)]=b
    return x.to(device),y.to(device)
def amp(device):
    if device.type=='cuda':return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return nullcontext()
def load_model(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    cfg=NeralConfig(**payload.get('config',{}));expected=expected_parameters(cfg)
    if expected==50_012_976:
        from model50 import Neral50M
        model=Neral50M(cfg)
    else:model=Neral20M(cfg)
    assert model.parameter_report()['total']==expected
    state=payload['model'];assert all(torch.isfinite(v).all() for v in state.values()),'Nonfinite seed weights'
    assert torch.equal(state['tok_emb.weight'],state['lm_head.weight']),'Inconsistent tied weights'
    model.load_state_dict(state,strict=True)
    return model.to(device),payload
@torch.inference_mode()
def evaluate(model,records,device,batch_size=8):
    was_training=model.training;model.eval();result={}
    for domain in sorted({r['domain'] for r in records}):
        selected=[r for r in records if r['domain']==domain];total_loss=0.;tokens=0
        for start in range(0,len(selected),batch_size):
            x,y=batch(selected[start:start+batch_size],device)
            with amp(device):logits,_=model(x);loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum')
            total_loss+=float(loss);tokens+=int((y!=-100).sum())
        result[domain]={'loss':total_loss/tokens,'answer_tokens':tokens,'records':len(selected)}
    model.train(was_training)
    return {'domains':result,'macro_loss':sum(v['loss'] for v in result.values())/len(result),'definition':'Mean of domain mean answer-token cross-entropies (nats). Prompt/padding labels ignored; EOS/newline included. Not directly comparable with legacy all-token loss.'}
def eligible(candidate,baseline):
    return all(v['loss']<=baseline['domains'][d]['loss']*1.10+0.03 for d,v in candidate['domains'].items())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init-from',default='seed/neral20m.pt');ap.add_argument('--data',default='curriculum');ap.add_argument('--out',default='runs/pilot_01');ap.add_argument('--steps',type=int,default=160);ap.add_argument('--lr',type=float,default=3e-5);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--grad-accum',type=int,default=4);ap.add_argument('--eval-every',type=int,default=20);ap.add_argument('--patience',type=int,default=4);ap.add_argument('--min-delta',type=float,default=.005);ap.add_argument('--seed',type=int,default=96820);ap.add_argument('--resume');args=ap.parse_args()
    if min(args.steps,args.batch_size,args.grad_accum,args.eval_every,args.patience)<=0:raise ValueError('Positive budgets required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required: refusing accidental CPU training')
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_float32_matmul_precision('high');device=torch.device('cuda');rng=random.Random(args.seed)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Choose a fresh run directory; no silent overwrite')
    out.mkdir(parents=True,exist_ok=True)
    train=read_records(Path(args.data)/'train.jsonl');val=read_records(Path(args.data)/'val.jsonl')
    for r in train+val:encode_record(r)
    groups={d:[r for r in train if r['domain']==d] for d in sorted({r['domain'] for r in train})}; domains=list(groups)
    signature={s:sha(Path(args.data)/(s+'.jsonl')) for s in ['train','val']}
    model,payload=load_model(args.resume or args.init_from,device)
    capacity_only=os.environ.get('NERAL_TRAIN_NEW_CAPACITY')=='1'
    if capacity_only:
        if model.cfg.ffn_hidden!=3648:raise ValueError('Capacity-only training requires own widened model')
        for name,param in model.named_parameters():
            if '.mlp.' not in name:param.requires_grad_(False)
            elif name.endswith('down.weight'):
                mask=torch.ones_like(param);mask[:,:1536]=0;param.register_hook(lambda grad,m=mask:grad*m)
            else:
                mask=torch.ones_like(param);mask[:1536]=0;param.register_hook(lambda grad,m=mask:grad*m)
        if args.resume:raise ValueError('Capacity-only resume is not implemented; use a fresh controlled run')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,betas=(.9,.95),weight_decay=0 if capacity_only else .03,fused=True)
    scaler=torch.amp.GradScaler('cuda',enabled=not torch.cuda.is_bf16_supported())
    baseline=evaluate(model,val,device);best=baseline['macro_loss'];bad=0;start_step=0
    if args.resume:
        if payload['data_hashes']!=signature:raise ValueError('Resume data mismatch')
        for field in ['lr','steps','batch_size','grad_accum','seed','eval_every','patience','min_delta']:
            if payload['args'][field]!=getattr(args,field):raise ValueError('Resume config mismatch: '+field)
        optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler']);rng.setstate(payload['rng_python']);torch.set_rng_state(payload['rng_torch']);torch.cuda.set_rng_state_all(payload['rng_cuda']);baseline=payload['baseline'];best=payload['best_val'];bad=payload['bad_evals'];start_step=payload['step']
    provenance={'source_checkpoint':str(Path(args.init_from).resolve()),'source_sha256':sha(args.init_from),'data_hashes':signature,'architecture':model.cfg.to_dict(),'parameters':model.parameter_report()['total'],'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'precision':'bf16' if torch.cuda.is_bf16_supported() else 'fp16+GradScaler','seed':args.seed,'args':vars(args),'selection':'macro validation loss; <=10%+0.03 nats regression per domain vs baseline; min_delta; test never used','baseline':baseline}
    provenance['capacity_only']=capacity_only
    provenance['effective_updated_parameters']=19_464_192 if capacity_only else model.parameter_report()['total']
    if not args.resume:
        (out/'run.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        atomic_save({'model':model.state_dict(),'config':model.cfg.to_dict(),'step':0,'best_val':best,'source_sha256':provenance['source_sha256']},out/'best.pt')
    print(json.dumps({'event':'baseline','validation':baseline,'gpu':provenance['gpu'],'precision':provenance['precision']}),flush=True)
    def checkpoint(step):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'config':model.cfg.to_dict(),'step':step,'best_val':best,'bad_evals':bad,'baseline':baseline,'data_hashes':signature,'rng_python':rng.getstate(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'args':vars(args),'source_sha256':provenance['source_sha256']}
    torch.cuda.reset_peak_memory_stats();run_start=time.perf_counter();stop='max_steps';step=start_step
    try:
        for step in range(start_step+1,args.steps+1):
            model.train();torch.cuda.synchronize();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss_sum=0;target_tokens=0;input_tokens=0
            warmup=min(10,max(1,args.steps//10));lr=args.lr*min(1,step/warmup)*(.2+.8*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,args.steps-warmup))))
            for g in optimizer.param_groups:g['lr']=lr
            for _ in range(args.grad_accum):
                # Domain-balanced sampling, no duplicated rows stored on disk.
                rs=[rng.choice(groups[rng.choice(domains)]) for _ in range(args.batch_size)]
                x,y=batch(rs,device);target_tokens+=int((y!=-100).sum());input_tokens+=int((x!=0).sum())
                with amp(device):_,loss=model(x,y)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss/args.grad_accum).backward();loss_sum+=float(loss.detach())
            scaler.unscale_(optimizer);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            dt=time.perf_counter()-tick;record={'step':step,'train_loss':loss_sum/args.grad_accum,'lr':lr,'grad_norm':float(gn),'train_seconds':dt,'input_tokens':input_tokens,'answer_tokens':target_tokens,'input_tokens_per_second':input_tokens/dt,'peak_vram_bytes':torch.cuda.max_memory_allocated()}
            if step%args.eval_every==0 or step==args.steps:
                score=evaluate(model,val,device);ok=eligible(score,baseline);improved=ok and score['macro_loss']<best-args.min_delta
                if improved:best=score['macro_loss'];bad=0
                else:bad+=1
                record.update(validation=score,eligible=ok,selected=improved,bad_evals=bad)
                if improved:atomic_save(checkpoint(step),out/'best.pt')
                atomic_save(checkpoint(step),out/'latest.pt')
            with (out/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            if step%10==0:print(json.dumps(record),flush=True)
            if bad>=args.patience:stop='early_stopping';break
    except KeyboardInterrupt:stop='interrupted'
    finally:
        atomic_save(checkpoint(step),out/'latest.pt')
        summary={'stop_reason':stop,'completed_step':step,'best_validation_macro_loss':best,'elapsed_seconds':time.perf_counter()-run_start,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'best_sha256':sha(out/'best.pt')}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
