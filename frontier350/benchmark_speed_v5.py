"""Isolated production A/B harness. No promotion, controller or state writes.

Runs only after the protected GitHub run has completed and the GPU is idle.
All datasets and checkpoints are explicit, hash-verified inputs. Audits and timed
runs are separate; repeated trials use the original production trainer as A.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from recovery_identity import file_sha
from run_scale_v4 import core_gate, bi_gate, scaled_recipe, RECIPES, valid_budget

ROOT = Path(__file__).resolve().parent
PRODUCTION_SHA = '7da6a7e906a17f0acb66c6c0c6d120207f335a63'
PROTECTED_RUN = 34755193140
TPU = 24576
CANDIDATES = {
    'fused_replay_pack9': ['--fused-execution','--fuse-replay'],
    'fused_replay_bulk_pack9': ['--fused-execution','--fuse-replay','--bulk-transfer'],
    **{f'fused_replay_bulk_pack{p}': ['--fused-execution','--fuse-replay','--bulk-transfer',
                                    '--execution-pack',str(p)] for p in (10,11,12)},
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def verify_production_sources():
    expected=read(ROOT/'speed_v5_production_blobs.json')
    for name,sha in expected.items():
        # Git checkout may convert LF to CRLF on Windows.
        data=(ROOT/name).read_text(encoding='utf-8').encode('utf-8')
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=sha:raise ValueError('pinned production source differs: '+name)
    return expected


def require_completed_run():
    token=os.environ.get('GITHUB_TOKEN')
    if not token:raise RuntimeError('GITHUB_TOKEN required for live protected-run check')
    req=urllib.request.Request(
        f'https://api.github.com/repos/Cola968/neral-helios/actions/runs/{PROTECTED_RUN}',
        headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=30) as response:run=json.load(response)
    if run.get('id')!=PROTECTED_RUN or run.get('status')!='completed':
        raise RuntimeError('protected Scale V4 run is active; no GPU work allowed')
    return {'id':run['id'],'status':run['status'],'conclusion':run.get('conclusion'),
            'head_sha':run['head_sha']}


def require_idle_gpu():
    require_completed_run()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    if apps:raise RuntimeError('GPU has compute processes or cannot prove idle: '+apps)
    info=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,temperature.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
    if len(info)!=1:raise RuntimeError('requires one isolated RTX 4070')
    name,free,temp=[s.strip() for s in info[0].split(',')]
    if 'RTX 4070' not in name or int(free)<9000 or int(temp)>=85:
        raise RuntimeError('GPU identity, free memory or temperature preflight failed')
    return info[0]


def compare_audits(a, b):
    rows=[]
    for step in range(1,4):
        x=torch.load(a/f'audit_{step:03d}.pt',map_location='cpu',weights_only=False)
        y=torch.load(b/f'audit_{step:03d}.pt',map_location='cpu',weights_only=False)
        if x['batch_chain']!=y['batch_chain'] or x['grads'].keys()!=y['grads'].keys():
            raise ValueError('sample or trainable-parameter identity mismatch')
        total=delta=0.;maximum=0.
        for name,g in x['grads'].items():
            h=y['grads'][name]
            if not torch.isfinite(g).all() or not torch.isfinite(h).all():
                raise ValueError('nonfinite gradient')
            d=(g-h).double();total+=float(g.double().square().sum());delta+=float(d.square().sum())
            maximum=max(maximum,float(d.abs().max()))
        relative=math.sqrt(delta)/max(math.sqrt(total),1e-12)
        loss_delta=abs(float(x['loss'])-float(y['loss']))
        if not x['sampled_logits'] or x['sampled_logits'].keys()!=y['sampled_logits'].keys():
            raise ValueError('sampled logit inputs differ')
        logit_delta=logit_total=0.;logit_max=0.
        for key,g in x['sampled_logits'].items():
            h=y['sampled_logits'][key]
            if not torch.isfinite(g).all() or not torch.isfinite(h).all():raise ValueError('nonfinite logits')
            d=(g-h).double();logit_delta+=float(d.square().sum());logit_total+=float(g.double().square().sum())
            logit_max=max(logit_max,float(d.abs().max()))
        logit_relative=math.sqrt(logit_delta)/max(math.sqrt(logit_total),1e-12)
        rows.append({'step':step,'loss_abs_delta':loss_delta,'gradient_relative_l2':relative,
            'gradient_max_abs_delta':maximum,'sampled_logit_max_abs_delta':logit_max,
            'sampled_logit_relative_l2':logit_relative,
            'pass':math.isfinite(loss_delta) and relative<=.005 and loss_delta<=.002
                   and logit_relative<=.005 and logit_max<=.125})
    ca=torch.load(a/'resume.pt',map_location='cpu',weights_only=False)
    cb=torch.load(b/'resume.pt',map_location='cpu',weights_only=False)
    masters_a=ca['v9_training_state']['masters'];masters_b=cb['v9_training_state']['masters']
    if masters_a.keys()!=masters_b.keys():raise ValueError('master parameter names differ')
    if not all(torch.isfinite(p).all() for p in masters_b.values()):raise ValueError('nonfinite masters')
    master_delta=max(float((p-masters_b[n]).abs().max()) for n,p in masters_a.items())
    return {'steps':rows,'master_max_abs_delta':master_delta,
            'pass':all(r['pass'] for r in rows) and master_delta<=2e-5}


def summarize_trial(out, warmup, expected_tokens, expected_updates, wall):
    rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines() if s.strip()]
    startup=next(r for r in rows if r.get('event')=='startup')
    train=[r for r in rows if r.get('event')=='train']
    summary=read(out/'summary.json')
    if len(train)!=expected_updates or summary.get('new_tokens')!=expected_tokens or summary.get('frozen_weights_unchanged') is not True:
        raise ValueError('incomplete run, token mismatch or frozen-weight check missing')
    for r in train:
        if not all(math.isfinite(r[k]) for k in ('loss','grad_norm','seconds','tokens_per_second')):
            raise ValueError('nonfinite training metrics')
    rates=[r['tokens_per_second'] for r in train[warmup:]]
    median=statistics.median(rates);p10=sorted(rates)[math.floor((len(rates)-1)*.1)]
    allocated=max(r['peak_allocated_bytes'] for r in train);reserved=max(r['peak_reserved_bytes'] for r in train)
    return {'median_tps':median,'p10_tps':p10,'samples':len(rates),'wall_seconds':wall,
        'end_to_end_tps_including_compile_checkpoint':expected_tokens/wall,
        'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
        'stable':p10>=.9*median,'memory_pass':allocated<=10.25*2**30 and reserved<=11*2**30,
        'batch_chain':summary['batch_chain'],'source_tokens':summary['source_tokens'],
        'contract':startup['contract'],'frozen_weights_unchanged':True}


def same_contract(a,b):
    keys=('anchor_sha256','mode','new_tokens','seed','seq','microbatch','microbatches',
          'tokens_per_update','lr','lr_end','lr_schedule','replay_boost','trainable_names','data_sha256')
    return a['batch_chain']==b['batch_chain'] and a['source_tokens']==b['source_tokens'] and all(
        a['contract'][k]==b['contract'][k] for k in keys)


def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite quality evidence')
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    if isinstance(value,list):
        for v in value:finite_tree(v)


def main():
    ap=argparse.ArgumentParser()
    for key in ('assets','anchor','anchor-sha','out'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--recipe',choices=('balanced','retention'),default='balanced')
    ap.add_argument('--stage-tokens',type=int,default=5_000_000)
    ap.add_argument('--pressure',type=int,choices=range(4),default=0)
    ap.add_argument('--updates',type=int,default=64)
    ap.add_argument('--warmup',type=int,default=8)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--canary-tokens',type=int,default=5_013_504)
    args=ap.parse_args()
    if not 32<=args.updates<=256 or not 4<=args.warmup<=args.updates//4 or not 3<=args.repeats<=5:
        raise ValueError('insufficient warmup, duration or repetitions')
    if not 5_000_000<=args.canary_tokens<=25_000_000 or valid_budget(args.canary_tokens)!=args.canary_tokens:
        raise ValueError('canary needs >=5M tokens and a legal unchanged token budget')
    if not 5_000_000<=args.stage_tokens<=250_000_000:raise ValueError('invalid Scale V4 stage size')
    assets=Path(args.assets).resolve();out=Path(args.out).resolve();anchor=Path(args.anchor).resolve()
    verify_production_sources()
    if out==assets or assets in out.parents or out in assets.parents or anchor==out or out in anchor.parents:
        raise ValueError('output overlaps protected inputs')
    require_idle_gpu()  # Before output creation, CUDA initialization, or training.
    if file_sha(anchor)!=args.anchor_sha:raise ValueError('anchor identity mismatch')
    out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(out).free<60*2**30:raise RuntimeError('requires 60 GiB isolated scratch space')
    env=dict(os.environ)
    env.update(TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_MODULE_LOADING='LAZY')
    # Read-only hashes detect accidental writes; never attempt to restore live state.
    state_paths=list(assets.glob('frontier350_scale_v*_state.json'))
    protected={str(p):file_sha(p) for p in [anchor,*state_paths]}
    recipe=scaled_recipe(next(r for r in RECIPES if r['name']==args.recipe),args.stage_tokens,args.pressure)
    report={'schema':1,'status':'running','production_commit':PRODUCTION_SHA,'protected_run':PROTECTED_RUN,
        'anchor_sha256':args.anchor_sha,'recipe':recipe,'production_adoption':False,
        'measured':{},'estimates':{},'target_tps':[47000,50000],
        'implementation_sha256':{p.name:file_sha(p) for p in [ROOT/'train_guarded_v9.py',ROOT/'train_speed_v5.py',ROOT/'speed_execution_v5.py',Path(__file__)]}}
    write(out/'report.json',report)

    def launch(name,flags,budget,updates,audit=False,original=False):
        require_idle_gpu()
        target=out/name
        cmd=[sys.executable,'-u',str(ROOT/('train_guarded_v9.py' if original else 'train_speed_v5.py')),
            '--anchor',str(anchor),'--anchor-sha',args.anchor_sha,
            '--main-data',str(assets/'frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/train'),
            '--replay',str(assets/'frontier350_replay_v9_20260912'),'--out',str(target),
            '--mode','replay_only','--new-tokens',str(budget),'--max-updates',str(updates),
            '--batch-size','4','--microbatches','12','--main-pack','9','--fast-replay-only',
            '--compile-mode','default','--lr',str(recipe['lr']),'--lr-end',str(recipe['lr_end']),
            '--replay-boost',str(recipe['replay_boost'])]
        if not original:cmd+=['--protected-root',str(assets),*flags]
        if audit:cmd+=['--audit-gradients']
        started=time.perf_counter()
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        return target,time.perf_counter()-started

    def quality(checkpoint,tag):
        require_idle_gpu()
        tokens=int(torch.load(checkpoint,map_location='cpu',weights_only=False)['processed_tokens'])
        sha=file_sha(checkpoint);tokenizer=assets/'frontier350_assets/tokenizer.json'
        specs={
            'validation':('eval_validation_loss350.py',['--valid-data',str(assets/'frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid'),
                '--seq','512','--batch','4','--batches','32','--seed','351337']),
            'dev':('eval_recovery_dev_v2.py',['--tokenizer',str(tokenizer),'--expected-tokens',str(tokens)]),
            'bilingual':('eval_bilingual_holdout_v1.py',['--tokenizer',str(tokenizer),'--expected-tokens',str(tokens),
                '--cases',str(assets/'frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05/bilingual_holdout.json')])}
        result={}
        for key,(script,extra) in specs.items():
            path=out/f'{tag}_{key}.json'
            with (out/f'{tag}_{key}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(ROOT/script),'--checkpoint',str(checkpoint),
                    '--out',str(path),*extra],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
            result[key]=read(path);finite_tree(result[key])
            if result[key].get('checkpoint_sha256')!=sha or result[key].get('checkpoint_processed_tokens')!=tokens:
                raise ValueError('quality evidence checkpoint identity differs')
        return result

    try:
        reference,_=launch('audit_reference',[],3*TPU,3,audit=True)
        control,control_wall=launch('audit_production_control',[],3*TPU,3,original=True)
        control_summary=summarize_trial(control,0,3*TPU,3,control_wall)
        # This checks the instrumented reference against the unmodified production
        # program. Neither diagnostic run is counted as a speed measurement.
        ref_ck=torch.load(reference/'resume.pt',map_location='cpu',weights_only=False)
        ctl_ck=torch.load(control/'resume.pt',map_location='cpu',weights_only=False)
        ref_state=ref_ck['v9_training_state'];ctl_state=ctl_ck['v9_training_state']
        if ref_state['batch_chain']!=ctl_state['batch_chain'] or ref_state['source_tokens']!=ctl_state['source_tokens']:
            raise ValueError('instrumented baseline differs from production sampling')
        control_delta=max(float((p-ctl_state['masters'][n]).abs().max()) for n,p in ref_state['masters'].items())
        if not math.isfinite(control_delta) or control_delta>2e-5:raise ValueError('instrumented reference differs from production updates')
        report['measured']['audit_control_master_max_abs_delta']=control_delta
        del ref_ck,ctl_ck,ref_state,ctl_state
        audits={};eligible=[]
        for name,flags in CANDIDATES.items():
            try:
                candidate,_=launch('audit_'+name,flags,3*TPU,3,audit=True)
                audits[name]=compare_audits(reference,candidate)
                if audits[name]['pass']:eligible.append(name)
            except (subprocess.CalledProcessError,ValueError) as exc:
                audits[name]={'pass':False,'error':str(exc)}
        report['measured']['numerical_audits']=audits;write(out/'report.json',report)
        if not eligible:raise RuntimeError('no combined candidate passed numerical gates')
        timing={'production':[],**{name:[] for name in eligible}}
        for repeat in range(args.repeats):
            # Alternate order to expose clock/thermal/order bias; no simultaneous GPU jobs.
            order=['production',*eligible]
            if repeat%2:order.reverse()
            for name in order:
                target,wall=launch(f'timing_{repeat}_{name}',CANDIDATES.get(name,[]),
                    args.updates*TPU,args.updates,original=name=='production')
                timing[name].append(summarize_trial(target,args.warmup,args.updates*TPU,args.updates,wall))
            for name in eligible:
                if not same_contract(timing['production'][-1],timing[name][-1]):raise ValueError('timed A/B contract mismatch')
            report['measured']['timing']=timing;write(out/'report.json',report)
        baseline=statistics.median(r['median_tps'] for r in timing['production'])
        if not all(r['stable'] and r['memory_pass'] for r in timing['production']):raise ValueError('production baseline unstable')
        winners=[n for n in eligible if all(r['stable'] and r['memory_pass'] for r in timing[n])
            and statistics.median(r['median_tps'] for r in timing[n])>baseline]
        if not winners:raise RuntimeError('no stable measured speed improvement')
        best=max(winners,key=lambda n:statistics.median(r['median_tps'] for r in timing[n]))
        report['best_candidate']=best
        qa=quality(anchor,'anchor')
        canary_updates=math.ceil(args.canary_tokens/TPU)
        pa,wa=launch('canary_production',[],args.canary_tokens,canary_updates,original=True)
        pb,wb=launch('canary_candidate',CANDIDATES[best],args.canary_tokens,canary_updates)
        sa=summarize_trial(pa,args.warmup,args.canary_tokens,canary_updates,wa)
        sb=summarize_trial(pb,args.warmup,args.canary_tokens,canary_updates,wb)
        if not same_contract(sa,sb):raise ValueError('long canary contract mismatch')
        a=quality(pa/'resume.pt','canary_production');b=quality(pb/'resume.pt','canary_candidate')
        gates={}
        for label,left,right in [('production_vs_anchor',qa,a),('candidate_vs_anchor',qa,b),('candidate_vs_production',a,b)]:
            core,failures=core_gate(left,right);bilingual=bi_gate(left['bilingual'],right['bilingual'],right['dev'])
            gates[label]={'core':core,'failures':failures,'bilingual':bilingual}
        report['measured']['canary']={'production':sa,'candidate':sb,'gates':gates}
        report['all_retention_gates_pass']=all(g['core'] and g['bilingual'] for g in gates.values())
        improved=(b['validation']['validation_loss']<qa['validation']['validation_loss'] or
            any(b['dev']['categories'][c]['accuracy_pct']>qa['dev']['categories'][c]['accuracy_pct'] for c in qa['dev']['categories']) or
            b['bilingual']['overall_accuracy_pct']>qa['bilingual']['overall_accuracy_pct'])
        report['candidate_strict_improvement_vs_anchor']=improved
        report['candidate_safe_quality_gate_pass']=bool(gates['candidate_vs_anchor']['core'] and
            gates['candidate_vs_anchor']['bilingual'] and improved)
        report['target_measured']=bool(report['all_retention_gates_pass'] and sa['stable'] and sb['stable']
            and sa['memory_pass'] and sb['memory_pass'] and sb['median_tps']>=47000
            and sb['p10_tps']>=.9*47000 and sb['median_tps']>sa['median_tps'])
        report['status']='completed'
        report['required_next_step']='Review evidence; existing SAFE strict-improvement promotion rules still apply. No automatic adoption.'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        unchanged=all(Path(p).is_file() and file_sha(p)==sha for p,sha in protected.items())
        report['protected_inputs_unchanged']=unchanged
        if not unchanged:report.update(status='failed',error='protected input changed during experiment')
        write(out/'report.json',report)
        if not unchanged:raise RuntimeError('protected input changed; stop and inspect, never restore automatically')


if __name__=='__main__':main()
