"""Neral-50M B2: broader, provenance-tracked continued pretraining.

B2 consumes ``pretrain50_b2_data`` produced by prepare_pretrain50_b2.py. By default it
never replays a train block: if the local B2 corpus cannot supply enough unique blocks to
reach the cumulative target, training fails rather than silently repeating data.
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

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from pretrain50 import loss
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch

EXPECTED_PARAMS=50_012_976
STAGE_VERSION=2
SEED=50100


def hash_file(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_streams(root: Path,split: str):
    return {lang:np.memmap(root/f'{split}_{lang}.bin',dtype='<u2',mode='r') for lang in ('de','en')}


def load_documents(root: Path):
    path=root/'documents.jsonl'
    rows=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    if not rows:raise ValueError(f'Empty document index: {path}')
    return rows


def data_hashes(root: Path):
    required=['train_de.bin','train_en.bin','val_de.bin','val_en.bin','documents.jsonl','manifest.json']
    out={}
    for name in required:
        p=root/name
        if not p.is_file():raise FileNotFoundError(p)
        out[name]=hash_file(p)
    return out


def build_train_schedule(rows,seq,seed=SEED):
    schedule=document_blocks(rows,'de',seq,'train')+document_blocks(rows,'en',seq,'train')
    random.Random(seed).shuffle(schedule)
    de=sum(lang=='de' for lang,_ in schedule);total=len(schedule)
    return schedule,{
        'seed':seed,'unique_document_bounded_blocks':total,'tokens_available':total*seq,
        'de_blocks':de,'en_blocks':total-de,'de_fraction':de/max(1,total),
        'document_boundary_crossing':False,'replay':False,
    }


def lr_for_update(update,total_updates,peak,floor,warmup_fraction):
    if total_updates<=0:return floor
    warmup=max(1,round(total_updates*warmup_fraction))
    if update<warmup:return peak*(update+1)/warmup
    progress=(update-warmup)/max(1,total_updates-warmup-1)
    progress=min(1.0,max(0.0,progress))
    return floor+(peak-floor)*0.5*(1+math.cos(math.pi*progress))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--resume',required=True,help='Stage-B best/latest or interrupted B2 latest checkpoint')
    ap.add_argument('--data',default='pretrain50_b2_data')
    ap.add_argument('--old-data',default='pretrain50_data',help='Frozen B1 validation distribution for forgetting guard')
    ap.add_argument('--out',default='runs/neral50m_b2_100m')
    ap.add_argument('--target-cumulative-tokens',type=int,default=100_000_000)
    ap.add_argument('--seq',type=int,default=256);ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--peak-lr',type=float,default=1e-4);ap.add_argument('--min-lr',type=float,default=1e-5)
    ap.add_argument('--warmup-fraction',type=float,default=.02);ap.add_argument('--eval-every',type=int,default=250)
    ap.add_argument('--validation-blocks',type=int,default=192)
    args=ap.parse_args()
    if min(args.seq,args.batch,args.eval_every,args.validation_blocks)<=0:raise ValueError('positive seq/batch/eval/validation required')
    if args.seq>FIFTY.max_seq_len:raise ValueError('sequence exceeds model context')
    if not 0<args.min_lr<=args.peak_lr:raise ValueError('invalid learning rates')
    if not 0<=args.warmup_fraction<=.2:raise ValueError('warmup_fraction must be 0..0.2')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('B2 requires BF16 CUDA')

    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda');root=Path(args.data);old_root=Path(args.old_data)
    streams=load_streams(root,'train');val=load_streams(root,'val');docs=load_documents(root);hashes=data_hashes(root)
    old_val=load_streams(old_root,'val');old_docs=load_documents(old_root)
    old_hashes={name:hash_file(old_root/name) for name in ('val_de.bin','val_en.bin','documents.jsonl')}
    schedule,schedule_meta=build_train_schedule(docs,args.seq)
    val_schedule,val_meta=build_validation_schedule(docs,args.seq,args.validation_blocks,SEED+17)
    old_schedule,old_meta=build_validation_schedule(old_docs,args.seq,min(96,args.validation_blocks),SEED+29)

    model,payload=load_model(args.resume,device)
    if model.parameter_report()['trainable']!=EXPECTED_PARAMS:raise ValueError('B2 accepts exact Neral-50M only')
    model.train();source_sha=sha(args.resume)

    if payload.get('stage_b2_version')==STAGE_VERSION:
        if payload.get('stage_b2_hashes')!=hashes:raise ValueError('B2 corpus mismatch on resume')
        if payload.get('stage_b2_seq')!=args.seq or payload.get('stage_b2_batch')!=args.batch:raise ValueError('B2 shape mismatch on resume')
        if payload.get('stage_b2_schedule_seed')!=SEED:raise ValueError('B2 schedule seed mismatch on resume')
        if int(payload.get('stage_b2_target_cumulative_tokens',args.target_cumulative_tokens))!=args.target_cumulative_tokens:raise ValueError('B2 target changed on interrupted resume; start a new stage for a larger milestone')
        cursor=int(payload['stage_b2_cursor']);source_tokens=int(payload['source_cumulative_tokens'])
        best=float(payload['best_b2_validation']);baseline_old=float(payload['baseline_old_validation'])
        regression_evals=int(payload.get('regression_evals',0));old_regression_evals=int(payload.get('old_regression_evals',0))
        resume_optimizer=True
    else:
        cursor=0;source_tokens=int(payload.get('tokens',0));best=float('inf');regression_evals=0;old_regression_evals=0;resume_optimizer=False
        baseline_old=float('nan')

    if source_tokens>=args.target_cumulative_tokens and cursor==0:
        raise ValueError(f'source checkpoint already has {source_tokens:,} tokens >= target')
    target_stage_tokens=args.target_cumulative_tokens-source_tokens
    target_blocks=math.ceil(target_stage_tokens/args.seq)
    if target_blocks>len(schedule):
        raise ValueError(f'B2 needs {target_blocks:,} unique blocks but corpus supplies {len(schedule):,}; add data instead of replaying it')
    target_blocks=min(target_blocks,len(schedule));target_updates=math.ceil(target_blocks/args.batch)
    if not 0<=cursor<=target_blocks:raise ValueError('invalid B2 resume cursor')
    start_cursor=cursor

    opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),weight_decay=.1,fused=True)
    if resume_optimizer:
        if 'optimizer' not in payload:raise ValueError('Interrupted B2 checkpoint is missing optimizer state')
        opt.load_state_dict(payload['optimizer'])

    current_b2=validate_document_bounded(model,val,val_schedule,args.seq,device)
    current_old=validate_document_bounded(model,old_val,old_schedule,args.seq,device)
    if not resume_optimizer:
        best=current_b2['macro'];baseline_old=current_old['macro']

    out=Path(args.out)
    if out.exists() and any(out.iterdir()) and not resume_optimizer:raise FileExistsError('Fresh B2 output directory required')
    out.mkdir(parents=True,exist_ok=True)
    if resume_optimizer and not (out/'best_validation.pt').is_file():
        raise FileNotFoundError('Interrupted B2 resume requires the original best_validation.pt in the output directory')
    run={
        'stage':'B2 broader continued pretraining','stage_b2_version':STAGE_VERSION,'parameters':EXPECTED_PARAMS,
        'source_checkpoint':str(Path(args.resume).resolve()),'source_sha256':source_sha,'source_cumulative_tokens':source_tokens,
        'resume_cursor':cursor,'target_cumulative_tokens':args.target_cumulative_tokens,'target_stage_tokens':target_stage_tokens,'target_blocks':target_blocks,
        'data_root':str(root.resolve()),'data_hashes':hashes,'old_validation_hashes':old_hashes,'schedule':schedule_meta,
        'validation_schedule':val_meta,'old_validation_schedule':old_meta,'current_b2_validation_at_start':current_b2,
        'current_old_validation_at_start':current_old,'selection_old_baseline_macro':baseline_old,
        'selection':'lowest B2 document-bounded validation macro subject to <=5% regression on frozen B1 validation baseline',
        'optimizer':'fresh AdamW at new stage; optimizer state restored only for interrupted B2','lr_schedule':{'type':'linear warmup + cosine','peak':args.peak_lr,'floor':args.min_lr,'warmup_fraction':args.warmup_fraction},
        'replay_policy':'No train-block replay. Insufficient unique data is a hard error.','benchmark_policy':'Frozen free/v3/v4/RawMath benchmark required after stage; no promotion from loss alone.',
        'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'args':vars(args),
    }
    (out/'run.json').write_text(json.dumps(run,indent=2),encoding='utf8')

    def checkpoint():
        return {'model':model.state_dict(),'optimizer':opt.state_dict(),'config':FIFTY.to_dict(),'tokens':source_tokens+cursor*args.seq,
                'source_cumulative_tokens':source_tokens,'stage_b2_version':STAGE_VERSION,'stage_b2_cursor':cursor,
                'stage_b2_seq':args.seq,'stage_b2_batch':args.batch,'stage_b2_hashes':hashes,'stage_b2_schedule_seed':SEED,
                'stage_b2_target_cumulative_tokens':args.target_cumulative_tokens,'best_b2_validation':best,
                'baseline_old_validation':baseline_old,'regression_evals':regression_evals,
                'old_regression_evals':old_regression_evals,'source_sha256':source_sha}

    if cursor==0:
        atomic_save(checkpoint(),out/'best_validation.pt');atomic_save(checkpoint(),out/'latest.pt')

    start=time.perf_counter();update_seconds=0.;stop='target_reached';step=math.ceil(cursor/args.batch);torch.cuda.reset_peak_memory_stats()
    try:
        while cursor<target_blocks:
            items=schedule[cursor:min(target_blocks,cursor+args.batch)]
            x,y=make_batch(streams,items,args.seq,device);lr=lr_for_update(step,target_updates,args.peak_lr,args.min_lr,args.warmup_fraction)
            for g in opt.param_groups:g['lr']=lr
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True);v=loss(model,x,y)
            if not torch.isfinite(v):stop='nonfinite_loss';raise FloatingPointError('non-finite B2 loss')
            v.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();dt=time.perf_counter()-tick
            cursor+=len(items);step+=1;update_seconds+=dt
            row={'step':step,'stage_b2_cursor':cursor,'target_blocks':target_blocks,'stage_b2_tokens':cursor*args.seq,'cumulative_tokens':source_tokens+cursor*args.seq,
                 'train_loss':float(v),'lr':lr,'grad_norm':float(gn),'seconds':dt,'tokens_per_second':len(items)*args.seq/max(dt,1e-9)}
            if step%args.eval_every==0 or cursor==target_blocks:
                score=validate_document_bounded(model,val,val_schedule,args.seq,device);old_score=validate_document_bounded(model,old_val,old_schedule,args.seq,device)
                eligible=old_score['macro']<=baseline_old*1.05;improved=eligible and score['macro']<best-.002
                row['b2_validation']=score;row['old_validation']=old_score;row['selection_eligible']=eligible;row['selected']=improved
                if improved:
                    best=score['macro'];regression_evals=0;old_regression_evals=0;atomic_save(checkpoint(),out/'best_validation.pt')
                else:
                    regression_evals=regression_evals+1 if score['macro']>best*1.20 else 0
                    old_regression_evals=old_regression_evals+1 if old_score['macro']>baseline_old*1.10 else 0
                atomic_save(checkpoint(),out/'latest.pt');print(json.dumps(row),flush=True)
                if regression_evals>=3:stop='b2_quality_abort';break
                if old_regression_evals>=3:stop='forgetting_abort';break
            with (out/'metrics.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    finally:
        atomic_save(checkpoint(),out/'latest.pt')
        segment_tokens=(cursor-start_cursor)*args.seq
        summary={'stop':stop,'complete':stop=='target_reached' and cursor>=target_blocks,'stage_b2_blocks':cursor,'stage_b2_tokens':cursor*args.seq,
                 'this_segment_tokens':segment_tokens,'source_cumulative_tokens':source_tokens,'cumulative_tokens':source_tokens+cursor*args.seq,
                 'target_cumulative_tokens':args.target_cumulative_tokens,'best_b2_validation':best,'baseline_old_validation':baseline_old,
                 'elapsed_seconds':time.perf_counter()-start,'optimizer_seconds':update_seconds,
                 'optimizer_tokens_per_second':segment_tokens/max(update_seconds,1e-9),
                 'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
                 'best_sha256':sha(out/'best_validation.pt'),'latest_sha256':sha(out/'latest.pt'),'instruction_tuning_performed':False,'train_block_replay':False}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary),flush=True)
    if stop!='target_reached':raise RuntimeError(f'B2 stopped: {stop}')


if __name__=='__main__':main()
