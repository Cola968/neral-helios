"""Own-weight V9 A/B: identical replay and frozen layers, optional gradient protection.

Defaults reproduce the validated 4 x 12 layout. Throughput experiments can pack adjacent
MAIN microbatches into larger physical forwards while leaving the optimizer-update token
budget, 25% replay share, three protected replay references, and loss weighting unchanged.
"""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from model350 import Neral350M,FRONTIER_350
from protection_v9 import configure_trainable,project_descent,bound_step
from train350 import atomic_save,append_jsonl
from train350_compiled import PersistentMMapPool,mask_budget_targets,update_token_counts
from recovery_identity import file_sha,object_sha

DOMAINS=('education','general','math_reasoning')


def vector(parameters,gradient=False):
    return torch.cat([(p.grad if gradient else p.detach()).reshape(-1) for p in parameters])


def assign(parameters,values,gradient=False):
    offset=0
    with torch.no_grad():
        for p in parameters:
            chunk=values[offset:offset+p.numel()].view_as(p)
            if gradient:p.grad=chunk.clone()
            else:p.copy_(chunk)
            offset+=p.numel()


def main():
    ap=argparse.ArgumentParser()
    for name in ('anchor','anchor-sha','main-data','replay','out'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--mode',choices=('replay_only','projected'),required=True)
    ap.add_argument('--new-tokens',type=int,default=2_000_000)
    ap.add_argument('--resume',default='')
    ap.add_argument('--max-updates',type=int,default=0)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--microbatches',type=int,default=12)
    ap.add_argument('--main-pack',type=int,default=1,help='Pack this many adjacent main microbatches into one larger forward; replay refs remain separate')
    ap.add_argument('--compile-mode',choices=('none','default','reduce-overhead','max-autotune-no-cudagraphs'),default='default')
    args=ap.parse_args()
    seq,batch,micros=512,args.batch_size,args.microbatches
    if batch<=0 or micros<4 or args.main_pack<=0:raise ValueError('positive batch/main-pack and >=4 microbatches required')
    tokens_per_update=seq*batch*micros
    if args.new_tokens<=0 or 0<args.new_tokens%tokens_per_update<3*seq*batch:
        raise ValueError('budget must leave room for all three replay references in the final update')
    if file_sha(args.anchor)!=args.anchor_sha:raise ValueError('anchor checksum changed')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('use explicit resume')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(35901);torch.cuda.manual_seed_all(35901)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    source=torch.load(args.anchor,map_location='cpu',weights_only=False)
    if source['config']!=FRONTIER_350.to_dict():raise ValueError('architecture mismatch')
    model=Neral350M().to('cuda',dtype=torch.bfloat16)
    model.load_state_dict(source['model'],strict=True)
    named=configure_trainable(model,4);params=[p for _,p in named];names=[n for n,_ in named]
    frozen_names=set(model.state_dict())-set(names)
    pools={'main':PersistentMMapPool(Path(args.main_data),FRONTIER_350.vocab_size)}
    pools.update({d:PersistentMMapPool(Path(args.replay)/d,FRONTIER_350.vocab_size) for d in DOMAINS})
    hashes={d:{p.name:file_sha(p) for p in pool.paths} for d,pool in pools.items()}
    main_manifest=json.loads((Path(args.main_data).parent/'manifest.json').read_text())
    if main_manifest['tokenizer']['tokenizer_sha256']!=source['tokenizer_sha256'] or main_manifest.get('benchmark_content_included') is not False:
        raise ValueError('main corpus identity/separation mismatch')
    if hashes['main']!={p['path']:p['sha256'] for p in main_manifest['train_files']}:
        raise ValueError('main corpus contains changed/unlisted shards')
    replay_manifest=json.loads((Path(args.replay)/'manifest.json').read_text())
    if replay_manifest['tokenizer_sha256']!=source['tokenizer_sha256']:raise ValueError('replay tokenizer mismatch')
    for d in DOMAINS:
        if hashes[d]!={r['path']:r['sha256'] for r in replay_manifest['domains'][d]['files']}:
            raise ValueError('replay shard identity differs')
    contract={'anchor_sha256':args.anchor_sha,'mode':args.mode,'new_tokens':args.new_tokens,'seed':35901,
              'seq':seq,'microbatch':batch,'microbatches':micros,'main_pack':args.main_pack,'tokens_per_update':tokens_per_update,
              'lr':7.5e-5,'trainable_names':names,'data_sha256':hashes,
              'implementation':{p:file_sha(Path(__file__).parent/p) for p in ('model350.py','protection_v9.py','train_guarded_v9.py')},
              'master_precision':'FP32','autocast':'BF16','compile_mode':args.compile_mode,
              'replay_fraction_nominal':3/micros,'trust_radius_relative':.0005}
    optimizer=torch.optim.AdamW(params,lr=7.5e-5,betas=(.9,.95),weight_decay=.1,fused=True)
    rng=random.Random(35902);done=step=0;chain='0'*64;counts={d:0 for d in pools}
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False);state=ck['v9_training_state']
        if state['contract']!=contract:raise ValueError('resume contract differs')
        with torch.no_grad():
            for name,p in named:p.copy_(state['masters'][name].to('cuda'))
        optimizer.load_state_dict(state['optimizer']);rng.setstate(state['sampler_rng'])
        done=state['new_tokens'];step=state['updates'];chain=state['batch_chain'];counts=state['source_tokens']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    run=model if args.compile_mode=='none' else torch.compile(model,fullgraph=True,dynamic=False,mode=args.compile_mode)
    total_parameters=sum(p.numel() for p in model.parameters())
    if total_parameters!=349_825_152:raise ValueError('parameter contract changed')
    startup={'event':'startup','contract':contract,'parameters':total_parameters,
             'trainable_parameters':sum(p.numel() for p in params),'frozen_parameters':total_parameters-sum(p.numel() for p in params),
             'start_tokens':source['processed_tokens'],'new_tokens_already_done':done,'gpu':torch.cuda.get_device_name(0)}
    append_jsonl(out/'metrics.jsonl',startup);print(json.dumps(startup),flush=True)
    start=time.perf_counter();start_done=done;start_step=step
    def sample(domain,count,physical_batch):
        nonlocal chain
        arrays=pools[domain].arrays
        rows=np.empty((physical_batch,seq+1),dtype=np.int64)
        for i in range(physical_batch):
            array=rng.choice(arrays);offset=rng.randrange(len(array)-seq)
            rows[i]=array[offset:offset+seq+1]
        if rows.min()<0 or rows.max()>=FRONTIER_350.vocab_size:raise ValueError('invalid token id')
        chain=hashlib.sha256(bytes.fromhex(chain)+domain.encode()+count.to_bytes(8,'little')+rows.tobytes()).hexdigest()
        whole=torch.from_numpy(rows).pin_memory().to('cuda',non_blocking=True)
        return whole[:,:-1],mask_budget_targets(whole[:,1:],count)
    def save():
        state={k:v.detach().to(device='cpu',dtype=torch.bfloat16) for k,v in model.state_dict().items()}
        if any(not torch.equal(state[k],source['model'][k]) for k in frozen_names):
            raise RuntimeError('frozen weights changed')
        ck=dict(source);ck.update(model=state,processed_tokens=int(source['processed_tokens'])+done,
                                 step=int(source['step'])+step,best_validation=float('inf'),validation_identity='')
        ck['optimizer']={**source['optimizer'],'state':{}}
        ck['v9_training_state']={'contract':contract,'masters':{n:p.detach().cpu() for n,p in named},
            'optimizer':optimizer.state_dict(),'sampler_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'new_tokens':done,'updates':step,'batch_chain':chain,'source_tokens':counts}
        ck['v9_provenance']={'anchor_sha256':args.anchor_sha,'mode':args.mode,'frozen_weights_unchanged':True,
                             'data_sha256':hashes,'new_training_tokens':done,'batch_chain':chain}
        atomic_save(ck,out/'resume.pt')
    if not args.resume:save()
    while done<args.new_tokens:
        tick=time.perf_counter();token_counts=update_token_counts(args.new_tokens-done,seq*batch,micros)
        total=sum(token_counts);references=[];weighted=None;loss_value=0.
        for domain,count in zip(DOMAINS,token_counts[:3]):
            optimizer.zero_grad(set_to_none=True);x,y=sample(domain,count,batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,loss=run(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite replay loss')
            loss.backward();ref=vector(params,True).detach();references.append(ref)
            if weighted is None:weighted=ref*(count/total)
            else:weighted.add_(ref,alpha=count/total)
            loss_value+=float(loss)*count/total
        optimizer.zero_grad(set_to_none=True);assign(params,weighted,True)
        main_counts=token_counts[3:]
        for i in range(0,len(main_counts),args.main_pack):
            group=main_counts[i:i+args.main_pack];count=sum(group);physical_batch=batch*len(group)
            x,y=sample('main',count,physical_batch)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,loss=run(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite main loss')
            (loss*(count/total)).backward();loss_value+=float(loss)*count/total
        gradient=vector(params,True);grad_norm=float(gradient.norm())
        diagnostics={}
        if args.mode=='projected':
            gradient,info=project_descent(gradient,references);diagnostics['gradient_projection']=info
            assign(params,gradient,True)
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        before=vector(params);optimizer.step()
        if args.mode=='projected':
            descent=before-vector(params)
            descent,info=project_descent(descent,references)
            descent,factor=bound_step(descent,before,.0005)
            assign(params,before-descent)
            diagnostics.update(step_projection=info,step_radius_scale=factor,
                               reference_descent_dots=[float(torch.dot(descent,r)) for r in references])
        if not all(torch.isfinite(p).all() for p in params):raise FloatingPointError('nonfinite parameter after update')
        done+=total;step+=1
        for i,count in enumerate(token_counts):counts[DOMAINS[i] if i<3 else 'main']+=count
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        row={'event':'train','step':step,'processed_tokens':int(source['processed_tokens'])+done,
             'new_tokens':done,'loss':loss_value,'grad_norm':grad_norm,'lr':7.5e-5,'seconds':elapsed,
             'tokens_per_second':total/elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'batch_size':batch,'microbatches':micros,
             'main_pack':args.main_pack,'max_main_physical_batch':batch*min(args.main_pack,max(1,micros-3)),
             'tokens_per_update':total,'batch_chain':chain,**diagnostics}
        append_jsonl(out/'metrics.jsonl',row)
        if step%5==0 or done==args.new_tokens:print(json.dumps(row),flush=True)
        if step%20==0 or done==args.new_tokens:save()
        if args.max_updates and step-start_step>=args.max_updates:save();break
    summary={'complete':done==args.new_tokens,'new_tokens':done,'processed_tokens':int(source['processed_tokens'])+done,
             'updates':step,'elapsed_session':time.perf_counter()-start,'tokens_this_session':done-start_done,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'batch_size':batch,'microbatches':micros,'main_pack':args.main_pack,'tokens_per_update':tokens_per_update,
             'batch_chain':chain,'source_tokens':counts,'checkpoint_sha256':file_sha(out/'resume.pt'),
             'frozen_weights_unchanged':True,'promoted':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
