#!/usr/bin/env python3
"""Offline 512-token Qwen chat-boundary audit of pinned NERAL-v3 320/64 files.

No model weights loaded, no GPU training, no truncation and no private texts
in output report. Uses exact original pinned QLoRA tokenize_examples().
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

PINNED_DATA={
 'train':{'count':320,
          'sha256':'8ad8de4ebc3aee1bf7376bd97f3520000b795a6bb756d364de8c795a01f52ca0',
          'review_sha256':'0bc2c761d9968cd5f815a9b50314ea2a2cc1188821910165593316ca8adac391'},
 'validation':{'count':64,
          'sha256':'166ecff1e05088dadb3beb692541373c5293221df6e7b8a099b6a921c3128ee5',
          'review_sha256':'84613114481d9485b342c62bfc3359961e1d33d476934f03edc120fcaf108afe'},
}
PINNED_BASE_REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
MODEL_ID='Qwen/Qwen2.5-Coder-7B-Instruct'


def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def summarize_masks(examples:list[dict],max_len:int)->dict:
    if not examples or max_len!=512:
        raise ValueError('Unsupported number of examples or token window')
    lengths=[];completion=0;context=0
    for number,row in enumerate(examples,1):
        ids=row.get('input_ids');mask=row.get('completion_mask')
        if (not isinstance(ids,list) or not isinstance(mask,list) or
            len(ids)!=len(mask) or not 0<len(ids)<=max_len):
            raise ValueError(f'Sample {number}: missing or overlength token sequence')
        if any(v not in (0,1) for v in mask):
            raise ValueError('Only binary completion masks allowed')
        first=next((i for i,v in enumerate(mask) if v==1),None)
        if first is None or first==0 or mask[:first]!=[0]*first or mask[first:]!=[1]*(len(mask)-first):
            raise ValueError('Completion must be nonempty contiguous tail, prompt fully masked')
        lengths.append(len(ids));completion+=len(mask)-first;context+=first
    order=sorted(lengths)
    return {'count':len(examples),'min_tokens':order[0],
            'median_tokens':order[(len(order)-1)//2],
            'p95_tokens':order[math.ceil(.95*len(order))-1],
            'max_tokens':order[-1],'tokens_total':sum(lengths),
            'context_tokens':context,'completion_tokens':completion,
            'overflow':0,'completion_only':True,'max_length':512}


def load_verified(path:Path,review:Path,expected:dict):
    for p in (path,review):
        if p.is_symlink() or not p.is_file() or p.stat().st_size>1024*1024:
            raise ValueError('Untrusted local file or size exceeds original trainer cap')
    if (sha(path.read_bytes())!=expected['sha256'] or
        sha(review.read_bytes())!=expected['review_sha256']):
        raise ValueError('Not the pinned 320/64 split content')
    return True


def load_pinned_trainer(folder:Path):
    original=Path(folder);folder=folder.resolve()
    if original.is_symlink() or not folder.is_dir():
        raise ValueError('Untrusted immutable trainer root')
    script=folder/'scripts'/'neral_qlora_pilot.py'
    runtime=folder/'qwen_local_runtime.py'
    catalog=folder/'model_catalog.py'
    if any(p.is_symlink() or not p.is_file() for p in (script,runtime,catalog)):
        raise ValueError('Pinned QLoRA trainer and tokenizer runtime missing')
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('neral_pinned_qlora_token_preflight',script)
    if spec is None or spec.loader is None:
        raise ValueError('Pinned QLoRA source cannot be loaded')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(*,root:Path,base:Path,train:Path,train_review:Path,
          validation:Path,validation_review:Path,out:Path):
    out=Path(out)
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError('Output folder must be new and local')
    source={'train':(train,train_review),'validation':(validation,validation_review)}
    for name,(path,review) in source.items():
        load_verified(Path(path),Path(review),PINNED_DATA[name])
    pilot=load_pinned_trainer(root)
    from model_catalog import validate_hf_snapshot
    snapshot=validate_hf_snapshot(Path(base),verify_hashes=True)
    if (snapshot.get('revision')!=PINNED_BASE_REVISION or
        snapshot.get('base_model')!=MODEL_ID):
        raise ValueError('Unexpected Qwen snapshot')
    from transformers import Qwen2TokenizerFast
    tokenizer=Qwen2TokenizerFast.from_pretrained(
        str(base),local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    report={'schema':1,'status':'NERAL_V3_PINNED_TOKEN_GATE_PASS',
            'base_revision':PINNED_BASE_REVISION,'window':512,
            'training_performed':False,'adapter_created':False,'release_validated':False,
            'independent_holdout':False,'splits':{}}
    for name,(path,review) in source.items():
        pilot.verify_review(path,review)
        rows=pilot.read_examples(path)
        if len(rows)!=PINNED_DATA[name]['count']:
            raise ValueError('Sample count differs from frozen v3 split')
        token_rows=pilot.tokenize_examples(rows,tokenizer,512)
        stats=summarize_masks(token_rows,512)
        stats['data_sha256']=PINNED_DATA[name]['sha256']
        stats['review_sha256']=PINNED_DATA[name]['review_sha256']
        report['splits'][name]=stats
    out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('trainer-root','base','train','train-review','validation',
                 'validation-review','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    result=audit(root=a.trainer_root,base=a.base,train=a.train,
                 train_review=a.train_review,validation=a.validation,
                 validation_review=a.validation_review,out=a.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
