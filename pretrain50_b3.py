"""Neral-50M B3: disjoint 300M/500M long-context continued-pretraining phases.

B2 already supplies seq=256 training. B3 phase A advances to seq=512 and ~300M cumulative
tokens; phase B advances to seq=1024 and ~500M cumulative tokens. The prepared corpus stores
phase A and B in disjoint documents, and this trainer refuses train-block replay. Each phase is
run separately so a frozen benchmark can be reviewed before the next phase is started.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch

from architectures import FIFTY
from experiment import atomic_save, load_model, sha
from loss_telemetry50 import ce_to_bits, ce_to_perplexity, eval_due, should_select, validation_derived
from pretrain50 import loss
from pretrain50_stage_b import document_blocks, build_validation_schedule, validate_document_bounded, make_batch

EXPECTED_PARAMS=50_012_976
STAGE_VERSION=1
BASE_SEED=50300


def hash_file(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_streams(root: Path,split: str):
    return {lang:np.memmap(root/f'{split}_{lang}.bin',dtype='<u2',mode='r') for lang in ('de','en')}


def load_docs(root: Path):
    p=root/'documents.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
    if not rows:raise ValueError('empty B3 document index')
    return rows


def data_hashes(root: Path,phase: str):
    names=[f'train_{phase}_de.bin',f'train_{phase}_en.bin','val_de.bin','val_en.bin','documents.jsonl','manifest.json']
    return {name:hash_file(root/name) for name in names}


def phase_schedule(docs,phase,seq,seed):
    split=f'train_{phase}'
    schedule=document_blocks(docs,'de',seq,split)+document_blocks(docs,'en',seq,split)
    random.Random(seed).shuffle(schedule)
    if not schedule:raise ValueError(f'no B3 phase-{phase} document blocks')
    return schedule


def lr_for(step,total,peak,floor,warmup_fraction):
    warm=max(1,round(total*warmup_fraction))
    if step<warm:return peak*(step+1)/warm
    p=(step-warm)/max(1,total-warm-1);p=min(1.,max(0.,p))
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*p))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('a','b'),required=True);ap.add_argument('--resume',required=True)
    ap.add_argument('--data',default='pretrain50_b3_data');ap.add_argument('--old-data',default='pretrain50_b2_data');ap.add_argument('--out')
    ap.add_argument('--target-cumulative-tokens',type=int);ap.add_argument('--seq',type=int);ap.add_argument('--batch',type=int)
    ap.add_argument('--peak-lr',type=float);ap.add_argument('--min-lr',type=float);ap.add_argument('--warmup-fraction',type=float,default=.02)
    ap.add_argument('--eval-every',type=int,default=200);ap.add_argument('--validation-blocks',type=int,default=192)
    ap.add_argument('--early-eval-every',type=int,default=100,help='Denser validation during early warmup; does not alter optimizer steps')
    ap.add_argument('--early-eval-until',type=int,default=1000,help='Last update eligible for denser early validation')
    ap.add_argument('--selection-min-delta',type=float,default=.0005,help='Minimum guarded validation CE improvement saved as best')
    a=ap.parse_args()
    defaults={'a':{'target':300_000_000,'seq':512,'batch':16,'peak':6e-5,'floor':6e-6},'b':{'target':500_000_000,'seq':1024,'batch':8,'peak':3e-5,'floor':3e-6}}[a.phase]
    target=a.target_cumulative_tokens or defaults['target'];seq=a.seq or defaults['seq'];batch=a.batch or defaults['batch'];peak=a.peak_lr or defaults['peak'];floor=a.min_lr or defaults['floor']
    if a.phase=='a' and seq!=512:raise ValueError('B3 phase A is frozen at seq=512 for comparable curriculum evidence')
    if a.phase=='b' and seq!=1024:raise ValueError('B3 phase B is frozen at seq=1024 for comparable curriculum evidence')
    if seq>FIFTY.max_seq_len or min(seq,batch,a.eval_every,a.validation_blocks,a.early_eval_every)<=0:raise ValueError('invalid B3 shape/budget')
    if a.early_eval_until<0 or a.selection_min_delta<0:raise ValueError('invalid B3 evaluation/selection settings')
    if not 0<floor<=peak:raise ValueError('invalid learning rates')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('B3 requires BF16 CUDA')

    seed=BASE_SEED+(0 if a.phase=='a' else 1);torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda');root=Path(a.data);old_root=Path(a.old_data);docs=load_docs(root);streams=load_streams(root,f'train_{a.phase}');val=load_streams(root,'val')
    old_val=load_streams(old_root,'val');old_docs=load_docs(old_root);hashes=data_hashes(root,a.phase)
    schedule=phase_schedule(docs,a.phase,seq,seed);val_schedule,val_meta=build_validation_schedule(docs,seq,a.validation_blocks,seed+17);old_schedule,old_meta=build_validation_schedule(old_docs,seq,min(96,a.validation_blocks),seed+29)
    model,payload=load_model(a.resume,device)
    if model.parameter_report()['trainable']!=EXPECTED_PARAMS:raise ValueError('B3 accepts exact Neral-50M only')
    model.train();source_sha=sha(a.resume)

    resume_same=payload.get('stage_b3_version')==STAGE_VERSION and payload.get('stage_b3_phase')==a.phase
    if resume_same:
        if payload.get('stage_b3_hashes')!=hashes or payload.get('stage_b3_seq')!=seq or payload.get('stage_b3_batch')!=batch:raise ValueError('B3 resume corpus/shape mismatch')
        if int(payload.get('stage_b3_target_cumulative_tokens'))!=target:raise ValueError('B3 resume target changed')
        cursor=int(payload['stage_b3_cursor']);source_tokens=int(payload['source_cumulative_tokens']);best=float(payload['best_b3_validation']);baseline_old=float(payload['baseline_old_validation']);resume_optimizer=True
        best_step=payload.get('best_b3_step');best_tokens=payload.get('best_b3_tokens');best_old=payload.get('best_b3_old_validation')
    else:
        cursor=0;source_tokens=int(payload.get('tokens',0));best=float('inf');baseline_old=float('nan');resume_optimizer=False
        best_step=None;best_tokens=None;best_old=None
    if source_tokens>=target and cursor==0:raise ValueError(f'checkpoint already has {source_tokens:,} tokens >= B3 target {target:,}')
    stage_tokens=target-source_tokens;target_blocks=math.ceil(stage_tokens/seq)
    if target_blocks>len(schedule):raise ValueError(f'B3 phase {a.phase} needs {target_blocks:,} unique blocks but supplies {len(schedule):,}; add data, never replay')
    total_updates=math.ceil(target_blocks/batch);start_cursor=cursor

    opt=torch.optim.AdamW(model.parameters(),lr=peak,betas=(.9,.95),weight_decay=.1,fused=True)
    if resume_optimizer:
        if 'optimizer' not in payload:raise ValueError('B3 interrupted checkpoint missing optimizer')
        opt.load_state_dict(payload['optimizer'])
    current=validate_document_bounded(model,val,val_schedule,seq,device);old_current=validate_document_bounded(model,old_val,old_schedule,seq,device)
    if not resume_optimizer:
        best=current['macro'];baseline_old=old_current['macro'];best_step=0;best_tokens=source_tokens;best_old=old_current['macro']
    out=Path(a.out or f'runs/neral50m_b3{a.phase}_{target//1_000_000}m')
    if out.exists() and any(out.iterdir()) and not resume_optimizer:raise FileExistsError('fresh B3 output required')
    out.mkdir(parents=True,exist_ok=True)
    run={'stage':f'B3{a.phase} disjoint long-context continued pretraining','stage_b3_version':STAGE_VERSION,'phase':a.phase,'parameters':EXPECTED_PARAMS,
         'source_checkpoint':str(Path(a.resume).resolve()),'source_sha256':source_sha,'source_cumulative_tokens':source_tokens,'target_cumulative_tokens':target,
         'seq':seq,'batch':batch,'unique_blocks_available':len(schedule),'target_blocks':target_blocks,'data_hashes':hashes,
         'validation_schedule':val_meta,'old_validation_schedule':old_meta,
         'selection':f'lowest B3 validation macro improving by >= {a.selection_min_delta:g} CE subject to <=5% regression on frozen B2 validation baseline',
         'evaluation':{'normal_every':a.eval_every,'early_every':a.early_eval_every,'early_until':a.early_eval_until},
         'replay_policy':'No train-block replay. Phase A/B documents are disjoint and B2 document hashes were excluded during corpus build.',
         'benchmark_policy':'Run frozen free/v3/v4/RawMath/BPB/context/NIS after this phase. Do not automatically start the next phase.',
         'lr':{'peak':peak,'floor':floor,'warmup_fraction':a.warmup_fraction},'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__)}
    (out/'run.json').write_text(json.dumps(run,indent=2),encoding='utf8')

    def checkpoint():
        return {'model':model.state_dict(),'optimizer':opt.state_dict(),'config':FIFTY.to_dict(),'tokens':source_tokens+cursor*seq,'source_cumulative_tokens':source_tokens,
                'stage_b3_version':STAGE_VERSION,'stage_b3_phase':a.phase,'stage_b3_cursor':cursor,'stage_b3_seq':seq,'stage_b3_batch':batch,'stage_b3_hashes':hashes,
                'stage_b3_target_cumulative_tokens':target,'best_b3_validation':best,'baseline_old_validation':baseline_old,'source_sha256':source_sha,
                'best_b3_step':best_step,'best_b3_tokens':best_tokens,'best_b3_old_validation':best_old}
    if cursor==0:atomic_save(checkpoint(),out/'best_validation.pt');atomic_save(checkpoint(),out/'latest.pt')
    step=math.ceil(cursor/batch);start=time.perf_counter();update_seconds=0.;bad=old_bad=0;stop='target_reached';torch.cuda.reset_peak_memory_stats()
    try:
        while cursor<target_blocks:
            items=schedule[cursor:min(target_blocks,cursor+batch)];x,y=make_batch(streams,items,seq,device);lr=lr_for(step,total_updates,peak,floor,a.warmup_fraction)
            for g in opt.param_groups:g['lr']=lr
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True);v=loss(model,x,y)
            if not torch.isfinite(v):stop='nonfinite_loss';raise FloatingPointError('non-finite B3 loss')
            v.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();dt=time.perf_counter()-tick
            cursor+=len(items);step+=1;update_seconds+=dt;row={'step':step,'cursor':cursor,'cumulative_tokens':source_tokens+cursor*seq,'train_loss':float(v),'lr':lr,'grad_norm':float(gn),'seconds':dt,'tokens_per_second':len(items)*seq/max(dt,1e-9)}
            if eval_due(step,a.eval_every,a.early_eval_every,a.early_eval_until) or cursor==target_blocks:
                score=validate_document_bounded(model,val,val_schedule,seq,device);old_score=validate_document_bounded(model,old_val,old_schedule,seq,device);eligible=old_score['macro']<=baseline_old*1.05
                improved=should_select(score['macro'],best,a.selection_min_delta,eligible=eligible)
                row.update(validation=score,old_validation=old_score,validation_derived=validation_derived(score),old_validation_derived=validation_derived(old_score),eligible=eligible,selected=improved)
                if improved:
                    best=score['macro'];best_step=step;best_tokens=source_tokens+cursor*seq;best_old=old_score['macro'];bad=old_bad=0;atomic_save(checkpoint(),out/'best_validation.pt')
                else:
                    bad=bad+1 if score['macro']>best*1.20 else 0;old_bad=old_bad+1 if old_score['macro']>baseline_old*1.10 else 0
                atomic_save(checkpoint(),out/'latest.pt');print(json.dumps(row),flush=True)
                if bad>=3:stop='quality_abort';break
                if old_bad>=3:stop='forgetting_abort';break
            with (out/'metrics.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    finally:
        atomic_save(checkpoint(),out/'latest.pt');segment=(cursor-start_cursor)*seq;summary={'phase':a.phase,'stop':stop,'complete':stop=='target_reached' and cursor>=target_blocks,
            'source_cumulative_tokens':source_tokens,'this_segment_tokens':segment,'cumulative_tokens':source_tokens+cursor*seq,'target_cumulative_tokens':target,'seq':seq,'batch':batch,
            'best_validation':best,'best_validation_perplexity':ce_to_perplexity(best),'best_validation_bits_per_token':ce_to_bits(best),
            'best_validation_step':best_step,'best_validation_cumulative_tokens':best_tokens,'best_old_validation_at_selection':best_old,
            'baseline_old_validation':baseline_old,'selection_min_delta':a.selection_min_delta,'optimizer_seconds':update_seconds,'optimizer_tokens_per_second':segment/max(update_seconds,1e-9),
            'elapsed_seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'best_sha256':sha(out/'best_validation.pt'),'latest_sha256':sha(out/'latest.pt'),'train_block_replay':False}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary),flush=True)
    if stop!='target_reached':raise RuntimeError(f'B3 phase {a.phase} stopped: {stop}')

if __name__=='__main__':main()