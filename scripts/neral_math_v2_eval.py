#!/usr/bin/env python3
"""Offline Qwen/base/old/new numerical comparison. Never promotes or uploads outputs."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import gc
import hashlib
import json
from pathlib import Path
import time
import sys

def ensure_pinned_model_runtime(trainer_root: Path) -> None:
    """Pin immutable trainer imports before loading Qwen, PEFT or adapter validators.

    The Windows Actions control checkout contains a different model_catalog.py,
    which previously shadowed the pinned trainer and broke evaluation after
    120 successful optimization steps.
    """
    original = Path(trainer_root)
    root = original.resolve()
    if original.is_symlink() or not root.is_dir():
        raise ValueError('Pinned trainer folder missing or symlinked')
    for name in ('model_catalog.py', 'qwen_local_runtime.py', 'neral_adapter_candidate.py'):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Pinned trainer module is missing or symlinked')
    for name in ('model_catalog', 'qwen_local_runtime', 'neral_adapter_candidate'):
        if name in sys.modules:
            path = getattr(sys.modules[name], '__file__', None)
            if not path or Path(path).resolve() != root / (name + '.py'):
                raise ValueError('Unpinned model runtime already loaded; refusing')
    sys.path.insert(0, str(root))

from scripts.neral_math_eval import CASES, SYSTEM, _load_model, normalized_number
from scripts.neral_math_v2_builder import EVAL_SEED, generate, norm, TRAIN_SEED


def cases():
    independent=generate(EVAL_SEED,each=8,eval_style=True)
    training_prompts={norm(x['prompt']) for x in generate(TRAIN_SEED)}
    original={norm(x['prompt']) for x in CASES}
    seen=set()
    derived=[]
    for i,x in enumerate(independent,1):
        key=norm(x['prompt'])
        if key in training_prompts or key in original or key in seen:
            raise ValueError('Generated evaluation prompt leaked into training or earlier cases')
        seen.add(key)
        derived.append({'id':f'novel-{i:03d}','prompt':x['prompt'],'expected':x['expected'],'family':x['family']})
    return [{**x,'family':'frozen16'} for x in CASES]+derived


def evaluate_text(raw,expected):
    candidate=normalized_number(raw)
    if candidate is None:return {'format_valid':False,'correct':False}
    try:return {'format_valid':True,'correct':Fraction(candidate)==Fraction(expected)}
    except (ZeroDivisionError,ValueError):return {'format_valid':True,'correct':False}


def score(results):
    required={'base','old_rank32','new_rank16_v2'}
    if set(results)!=required:raise ValueError('Expected three compared models')
    expected={r['id']:r for r in cases()}
    result={'schema':1,'status':'MATH_V2_DEVELOPMENT_COMPARISON',
            'question_count':len(expected),'frozen_dev_cases':len(CASES),
            'new_generated_dev_cases':len(expected)-len(CASES),
            'independent_holdout':False,'general_improvement_proven':False,
            'weights_promoted':False,'scores':{}}
    for label,rows in results.items():
        if len(rows)!=len(expected) or set(x['id'] for x in rows)!=set(expected):raise ValueError('Missing/duplicate comparison outputs')
        families=defaultdict(lambda:{'total':0,'correct':0,'format_valid':0})
        correct=format_valid=0
        for row in rows:
            ex=expected[row['id']]
            typ=ex['family']
            families[typ]['total']+=1
            families[typ]['correct']+=int(row['correct'])
            families[typ]['format_valid']+=int(row['format_valid'])
            correct+=int(row['correct']);format_valid+=int(row['format_valid'])
        result['scores'][label]={'correct':correct,'format_valid':format_valid,'total':len(rows),
                                  'by_family':dict(sorted(families.items()))}
    new=result['scores']['new_rank16_v2']['correct']
    result['delta_new_vs_base']=new-result['scores']['base']['correct']
    result['delta_new_vs_old']=new-result['scores']['old_rank32']['correct']
    result['positive_gain_on_dev_suite']=result['delta_new_vs_base']>0 and result['delta_new_vs_old']>0
    return result


def _run(label,model,tokenizer):
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval(); rows=[]
    for item in cases():
        prompt=[{'role':'system','content':SYSTEM},{'role':'user','content':item['prompt']}]
        ids=tokenizer.apply_chat_template(prompt,chat_template=CHAT_TEMPLATE,
            tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            out=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                max_new_tokens=48,pad_token_id=tokenizer.pad_token_id)
        raw=tokenizer.decode(out[0,ids.shape[-1]:],skip_special_tokens=True)
        result=evaluate_text(raw,item['expected'])
        rows.append({'id':item['id'],**result,'answer_sha256':hashlib.sha256(raw.encode()).hexdigest()})
        print(f'{label} {item["id"]}: format={result["format_valid"]} correct={result["correct"]}',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('base','old-r32','new-r16','out','summary','trainer-root'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args()
    ensure_pinned_model_runtime(a.trainer_root)
    if a.out.exists() or a.summary.exists():p.error('Output must be fresh')
    for directory in (a.base,a.old_r32,a.new_r16):
        if directory.is_symlink() or not directory.is_dir():p.error('Trusted local directories only')
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter as validate_r16
    from neral_adapter_candidate_r32 import validate_adapter as validate_r32
    snap=validate_hf_snapshot(a.base,verify_hashes=True)
    old=validate_r32(a.old_r32,snap)
    new=validate_r16(a.new_r16,snap)
    if old['release_validated'] or new['release_validated']:p.error('Experimental adapters only')
    import torch
    from transformers import Qwen2TokenizerFast
    from peft import PeftModel
    if not torch.cuda.is_available():p.error('CUDA unavailable')
    free,_=torch.cuda.mem_get_info(0)
    if free<9_000*1024**2:p.error('Less than 9000 MiB free VRAM')
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(a.base),local_files_only=True,token=False,trust_remote_code=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    a.out.mkdir(exist_ok=False)
    results={}
    for label,adapter in [('base',None),('old_rank32',a.old_r32),('new_rank16_v2',a.new_r16)]:
        m=_load_model(a.base)
        try:
            if adapter is not None:m=PeftModel.from_pretrained(m,str(adapter),is_trainable=False,local_files_only=True)
            results[label]=_run(label,m,tokenizer)
        finally:
            del m;gc.collect();torch.cuda.empty_cache()
    (a.out/'private-details.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    report=score(results)
    report['base_revision']=snap['revision']
    report['old_adapter_sha256']=old['files']['adapter_model.safetensors']
    report['new_adapter_sha256']=new['files']['adapter_model.safetensors']
    a.summary.parent.mkdir(exist_ok=True)
    a.summary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
