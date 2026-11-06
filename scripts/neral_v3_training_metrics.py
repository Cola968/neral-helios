#!/usr/bin/env python3
"""Read-only, loss/VRAM/adapter-SHA report for local NERAL v3 80-step A/B.

Never trains, loads model weights into memory, downloads, or exports prompts,
completions, secrets or adapter bytes. Outputs scalar-only metrics and SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

TRAIN_STEPS=80
BASE_REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
EXPECTED_PARAMS={16:40_370_176,32:80_740_352}
SUITES={'validation64':64,'math80':80,'phrasing36':36,'holdout20':20}
MODELS={'base','rank16_v3','rank32_v3'}
MAX_JSON_BYTES=1024*1024
MAX_ADAPTER_BYTES=512*1024*1024
HASH_PATTERN=re.compile(r'[a-f0-9]{64}\Z')


def _json(path:Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_JSON_BYTES:
        raise ValueError('Missing, symlinked or oversized private report')
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(path:Path, limit=MAX_ADAPTER_BYTES):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError('Missing, symlinked or oversized sealed input')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _float(value,name, *, zero_allowed=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(name+' must be finite')
    if value<0 or (not zero_allowed and value==0):
        raise ValueError(name+' must be positive')
    return round(float(value),5)


def _mib(v):
    if v is None:return None
    return round(_float(v,'VRAM bytes')/1024**2,2)


def summarize_rank(folder:Path,rank:int,expected_data_sha:str):
    if rank not in EXPECTED_PARAMS:raise ValueError('Unsupported LoRA rank')
    folder=Path(folder)
    if folder.is_symlink():raise ValueError('Symlinked rank directory')
    report_path=folder/'training-report.json'
    if not report_path.is_file():
        return {'rank':rank,'status':'missing','steps_completed':0,'adapter_sha256':None}
    doc=_json(report_path)
    if doc.get('steps')!=TRAIN_STEPS or doc.get('lora_rank')!=rank or doc.get('neral_trainable_parameters')!=EXPECTED_PARAMS[rank]:
        raise ValueError('Unapproved trainer rank/steps/parameter evidence')
    state=doc.get('status')
    if state not in ('completed','failed'):
        raise ValueError('Unknown or unfinished training status')
    steps=doc.get('optimizer_steps_completed')
    if steps is not None and (type(steps) is not int or not 0<=steps<=TRAIN_STEPS):
        raise ValueError('Invalid optimizer step report')
    if state=='completed' and steps!=TRAIN_STEPS:
        raise ValueError('Completed train without exactly 80 optimizer steps')
    curve=[]; seen=set()
    for entry in doc.get('logs',[]):
        if not isinstance(entry,dict):raise ValueError('Invalid loss log')
        if 'loss' not in entry:continue
        step=entry.get('step')
        if not isinstance(step,int) or isinstance(step,bool) or not 1<=step<=TRAIN_STEPS or step in seen:
            raise ValueError('Invalid or repeated loss step')
        seen.add(step)
        loss=_float(entry['loss'],'loss',zero_allowed=True)
        curve.append({'step':step,'loss':loss})
    curve.sort(key=lambda item:item['step'])
    # On exception, original QLoRA may not have reached the final metrics line.
    # A logged step is observed evidence but not an optimizer completion proof.
    reported_steps=steps if steps is not None else (max(seen) if seen else None)
    step_times=doc.get('optimizer_step_seconds')
    if step_times is None:
        step_times=[]
    if not isinstance(step_times,list) or len(step_times)>TRAIN_STEPS:
        raise ValueError('Invalid optimizer step time series')
    step_times=[_float(v,'optimizer step seconds') for v in step_times]
    sampled=_mib(doc.get('device_used_peak_sampled_bytes'))
    reserved=_mib(doc.get('peak_reserved_bytes'))
    allocated=_mib(doc.get('peak_allocated_bytes'))
    elapsed=doc.get('elapsed_seconds')
    result={
        'rank':rank,'status':state,'steps_completed':reported_steps,
        'steps_completed_proven':steps is not None, 'trainable_parameters':EXPECTED_PARAMS[rank],
        'loss_curve':curve,'logged_loss_points':len(curve),
        'loss_start':curve[0]['loss'] if curve else None,
        'loss_end':curve[-1]['loss'] if curve else None,
        'step_times_seconds':step_times,
        'step_times_count':len(step_times),
        'step_time_avg_seconds':round(sum(step_times)/len(step_times),4) if step_times else None,
        'elapsed_seconds':_float(elapsed,'elapsed seconds') if elapsed is not None else None,
        'peak_vram_sampled_mib':sampled,
        'peak_torch_allocated_mib':allocated,
        'peak_torch_reserved_mib':reserved,
        'adapter_sha256':None,
        'loss_reporting_complete':len(curve)==TRAIN_STEPS,
        'step_time_reporting_complete':len(step_times)==TRAIN_STEPS,
    }
    if state=='completed':
        manifest=_json(folder/'neral-adapter.json')
        config=_json(folder/'adapter_config.json')
        sealed=manifest.get('files',{}).get('adapter_model.safetensors')
        if not isinstance(sealed,str) or not HASH_PATTERN.fullmatch(sealed):
            raise ValueError('Adapter SHA256 missing from signed source manifest')
        if manifest.get('dataset_sha256')!=expected_data_sha or manifest.get('base_revision')!=BASE_REVISION or manifest.get('release_validated') is not False:
            raise ValueError('Wrong adapter data lineage, base model or release flag')
        if (config.get('r')!=rank or config.get('lora_alpha')!=rank or
                config.get('peft_type')!='LORA'):
            raise ValueError('Adapter config rank/type mismatch')
        actual=_digest(folder/'adapter_model.safetensors')
        if actual!=sealed:
            raise ValueError('adapter SHA256 differs from sealed manifest')
        result['adapter_sha256']=actual
    return result


def collect(run_root:Path,source_root:Path):
    run_root=Path(run_root);source_root=Path(source_root)
    if (run_root.is_symlink() or source_root.is_symlink() or
            not run_root.is_dir() or not source_root.is_dir()):
        raise ValueError('Local experiment directories missing or linked')
    source=_json(source_root/'summary.json')
    train=source.get('splits',{}).get('train',{})
    val=source.get('splits',{}).get('validation',{})
    if (source.get('status')!='NERAL_V3_TWO_LINE_ALGEBRA_REVIEW_ONLY' or
            train.get('count')!=320 or train.get('converted_equations')!=142 or
            val.get('count')!=64 or val.get('converted_equations')!=28):
        raise ValueError('Wrong two-line data source inventory')
    train_sha=_digest(source_root/'train.jsonl',limit=MAX_JSON_BYTES)
    if train_sha!=train.get('data_sha256'):
        raise ValueError('Training source SHA256 changed')
    val_sha=val.get('data_sha256')
    if not isinstance(val_sha,str) or not HASH_PATTERN.fullmatch(val_sha):
        raise ValueError('Unpinned validation SHA256')
    if _digest(source_root/'validation.jsonl',limit=MAX_JSON_BYTES)!=val_sha:
        raise ValueError('Validation source SHA256 changed')
    ranks={f'rank{rank}':summarize_rank(run_root/f'rank{rank}',rank,train_sha)
           for rank in (16,32)}
    statuses={rank['status'] for rank in ranks.values()}
    if statuses=={'missing'}:status='NO_TRAINING_REPORTS'
    elif statuses=={'completed'}:status='BOTH_TRAINED_SCORES_PENDING'
    else:status='PARTIAL_OR_FAILED_TRAINING'
    scores=None
    evalfile=run_root/'evaluation'/'summary.json'
    if evalfile.exists() or evalfile.is_symlink():
        report=_json(evalfile)
        if (report.get('status')!='NERAL_V3_RANK16_RANK32_DEV_COMPARISON' or
                report.get('weights_promoted') is not False or
                report.get('independent_external_benchmark') is not False):
            raise ValueError('Invalid evaluation status or release claim')
        if statuses!={'completed'}:
            raise ValueError('Comparison present before both ranks finished')
        if (report.get('adapter_16_sha256')!=ranks['rank16']['adapter_sha256'] or
                report.get('adapter_32_sha256')!=ranks['rank32']['adapter_sha256'] or
                report.get('validation_sha256')!=val_sha):
            raise ValueError('Evaluation adapter/source digest mismatch')
        raw_scores=report.get('scores')
        if not isinstance(raw_scores,dict) or set(raw_scores)!=MODELS:
            raise ValueError('Missing baseline or LoRA evaluation variant')
        scores={}
        for model, record in raw_scores.items():
            if not isinstance(record,dict) or set(record)!=set(SUITES):
                raise ValueError('Invalid development suite coverage')
            scores[model]={}
            for suite,n in SUITES.items():
                value=record[suite]
                if not isinstance(value,dict):raise ValueError('Missing score')
                correct=value.get('correct');formatted=value.get('format_valid')
                if (value.get('total')!=n or not all(type(v) is int and 0<=v<=n
                       for v in (correct,formatted))):
                    raise ValueError('Malformed suite count')
                scores[model][suite]={'correct':correct,'format_valid':formatted,'total':n}
        status='TRAINED_AND_DEVELOPMENT_EVALUATED'
    return {
        'schema':1,'status':status,'base_revision':BASE_REVISION,
        'source':{'train_count':320,'validation_count':64,
                  'training_sha256':train_sha,'validation_sha256':val_sha,
                  'train_equations_two_line':142,'validation_equations_two_line':28},
        'ranks':ranks,'evaluation_available':scores is not None,
        'scores':scores,'model_comparison_method':'single-NF4-base-with-disable_adapter() baseline',
        'development_suites':['validation64','math80','phrasing36','holdout20'],
        'known_36_development_cases_not_blind':True,
        'fresh_20_author_written_cases_not_external_benchmark':True,
        'independent_external_benchmark':False,
        'release_validated':False,'weights_promoted':False,
        'private_prompts_uploaded':False,'adapter_weights_uploaded':False,
    }



def discover_source(run_root:Path,search_root:Path) -> Path:
    """Find sealed local v3 data by adapter-proven dataset SHA, not filename guesses.

    The report-only Action may not receive original workflow_dispatch inputs in
    the workflow_run event payload; matching audited source bytes is safer.
    """
    if search_root.is_symlink() or not search_root.is_dir():
        raise ValueError('Untrusted local source discovery root')
    sealed=[]
    for rank in (16,32):
        p=run_root/f'rank{rank}'/'neral-adapter.json'
        if p.is_file() and not p.is_symlink():
            digest=_json(p).get('dataset_sha256')
            if isinstance(digest,str) and HASH_PATTERN.fullmatch(digest):
                sealed.append(digest)
    if not sealed or len(set(sealed))!=1:
        raise ValueError('No unique signed adapter data SHA to discover reviewed source')
    candidates=[]
    for folder in sorted(search_root.glob('math-v3-two-line-*-*')):
        if folder.is_symlink() or not folder.is_dir():continue
        if not re.fullmatch(r'math-v3-two-line-\d{8,14}-\d+',folder.name):continue
        if len(candidates)>=50:raise ValueError('Too many locally matching source directories')
        try:
            doc=_json(folder/'summary.json')
            tok=_json(folder/'token-report.json')
            if (doc.get('status')=='NERAL_V3_TWO_LINE_ALGEBRA_REVIEW_ONLY' and
                tok.get('status')=='NERAL_V3_TWO_LINE_TOKEN_GATE_PASS' and
                doc.get('splits',{}).get('train',{}).get('data_sha256')==sealed[0] and
                _digest(folder/'train.jsonl',limit=MAX_JSON_BYTES)==sealed[0]):
                candidates.append(folder)
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
            continue
    if len(candidates)!=1:
        raise ValueError('Cannot uniquely identify sealed 320/64 source ('+str(len(candidates))+')')
    return candidates[0]


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--source-search-root',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.out.is_symlink() or args.out.exists() or not args.out.parent.is_dir():
        parser.error('New report file required; no overwrite')
    source=args.source_root
    if source is None:
        if args.source_search_root is None:
            parser.error('Provide explicit audited source or local discovery root')
        source=discover_source(args.run_root,args.source_search_root)
    report=collect(args.run_root,source)
    if len(json.dumps(report).encode('utf-8'))>32768:
        parser.error('Report exceeds scalar-only payload limit')
    with args.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({'status':report['status'],'ranks':{
        name:{'status':r['status'],'steps':r['steps_completed'],'sha256':r['adapter_sha256']}
        for name,r in report['ranks'].items()},'evaluation_available':report['evaluation_available']}),flush=True)


if __name__=='__main__':
    main()