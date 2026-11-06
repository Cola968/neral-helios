#!/usr/bin/env python3
"""Offline frozen 16-question math development A/B/C: base, old R32, new math R16.

Evaluates answer-only arithmetic numerically; no generated code executed. Reports
only counts and hashes to GitHub. This small development suite is NOT a release gate.
"""
from __future__ import annotations
import argparse
import gc
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time

CASES=[
 {'id':'add-de-1','prompt':'Addiere 317 und 586. Antworte ausschließlich mit einer Zahl.','expected':'903'},
 {'id':'add-en-2','prompt':'Add 468 and 729. Respond with only the number.','expected':'1197'},
 {'id':'sub-de-1','prompt':'Was ergibt 963 minus 418? Antworte nur mit der Zahl.','expected':'545'},
 {'id':'sub-en-2','prompt':'Compute 802 minus 569. Only the number.','expected':'233'},
 {'id':'mul-de-1','prompt':'Multipliziere 37 mit 24. Nur die Zahl.','expected':'888'},
 {'id':'mul-en-2','prompt':'What is 18 multiplied by 47? Only the number.','expected':'846'},
 {'id':'percent-de-1','prompt':'Wie viel sind 15 Prozent von 380? Schreibe ausschließlich den Zahlenwert.','expected':'57'},
 {'id':'percent-en-2','prompt':'Find 40 percent of 125. Answer with the number only.','expected':'50'},
 {'id':'force-de-1','prompt':'F = m mal a. Für m = 13 kg und a = 7 m/s²: Wie groß ist F in Newton? Nur die Zahl.','expected':'91'},
 {'id':'force-en-2','prompt':'Force equals mass times acceleration. Mass 8 kg, acceleration 11 m/s². How many newtons? Give only a number.','expected':'88'},
 {'id':'equation-de-1','prompt':'Löse 5x + 7 = 82. Gib ausschließlich den Zahlenwert von x an.','expected':'15'},
 {'id':'equation-en-2','prompt':'Solve 9x - 8 = 64. Reply with the numeric value of x only.','expected':'8'},
 {'id':'sequence-de-1','prompt':'Zahlenfolge: 7, 12, 17, 22. Welche Zahl folgt? Nur die Zahl.','expected':'27'},
 {'id':'sequence-en-2','prompt':'Sequence: 3, 10, 17, 24. What comes next? Only the number.','expected':'31'},
 {'id':'fraction-de-1','prompt':'Was ist 5/8 mal 4? Gib nur das vollständig gekürzte Ergebnis als Zahl oder Bruch an.','expected':'5/2'},
 {'id':'fraction-en-2','prompt':'Compute 3/7 multiplied by 14. Only the final simplified number.','expected':'6'},
]
SYSTEM='You are an arithmetic assistant. Follow the user output format exactly. Do not make up calculations or tool executions.'


def normalized_number(s:str)->str|None:
    if not isinstance(s,str) or not re.fullmatch(r'[+-]?\d+(?:[.,]\d+)?(?:/\d+)?',s.strip()):return None
    return s.strip().replace(',','.')


def summary(results:dict)->dict:
    if set(results)!={'base','old_rank32','new_math_rank16'}:raise ValueError('All three comparators required')
    n=len(CASES)
    if any(len(v)!=n for v in results.values()):
        raise ValueError('Missing expected task runs')
    scores={label:{'correct':sum(bool(x['correct']) for x in rows),'format_valid':sum(bool(x['format_valid']) for x in rows),'total':len(rows)} for label,rows in results.items()}
    new=scores['new_math_rank16']['correct'];old=scores['old_rank32']['correct'];base=scores['base']['correct']
    return {'schema':1,'status':'SMALL_MATH_DEV_DIAGNOSTIC','question_count':n,
            'scores':scores,'math_correct_delta_new_vs_base':new-base,
            'math_correct_delta_new_vs_old':new-old,
            'general_improvement_proven':False,'weight_promotion_allowed':False,
            'note':'Strict numeric-only answers; 16 hand-authored math dev cases. Not blinded or sufficient for general improvement.'}


def _load_model(base:Path):
    import torch
    from transformers import BitsAndBytesConfig,Qwen2ForCausalLM
    return Qwen2ForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,token=False,
        use_safetensors=True,attn_implementation='sdpa',device_map={'':0},dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))


def _eval_model(label,model,tokenizer,output:Path):
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval();data=[]
    for case in CASES:
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':case['prompt']}]
        encoded=tokenizer.apply_chat_template(messages,chat_template=CHAT_TEMPLATE,
            tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
        t=time.monotonic()
        with torch.inference_mode():
            gen=model.generate(encoded,attention_mask=torch.ones_like(encoded),
                do_sample=False,max_new_tokens=48,pad_token_id=tokenizer.pad_token_id)
        raw=tokenizer.decode(gen[0,encoded.shape[-1]:],skip_special_tokens=True)
        v=normalized_number(raw)
        correct=False
        if v is not None:
            try:correct=Fraction(v)==Fraction(case['expected'])
            except (ZeroDivisionError,ValueError):pass
        data.append({'id':case['id'],'correct':correct,'format_valid':v is not None,
                     'answer_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                     'seconds':round(time.monotonic()-t,2)})
        print(f'{label}/{case["id"]}: numeric-format={v is not None}; correct={correct}',flush=True)
    return data


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--old-r32',required=True,type=Path)
    p.add_argument('--new-r16',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--summary',required=True,type=Path)
    args=p.parse_args(argv)
    if args.out.exists() or args.summary.exists() or args.out.is_symlink():p.error('Output already exists')
    for v in (args.base,args.old_r32,args.new_r16):
        if v.is_symlink() or not v.is_dir():p.error('Trusted local model folders only')
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter as r16_validator
    from neral_adapter_candidate_r32 import validate_adapter as r32_validator
    snap=validate_hf_snapshot(args.base,verify_hashes=True)
    new_manifest=r16_validator(args.new_r16,snap)
    old_manifest=r32_validator(args.old_r32,snap)
    if new_manifest['release_validated'] or old_manifest['release_validated']:
        p.error('Research-only adapters expected')
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():p.error('BF16 CUDA required')
    free,_=torch.cuda.mem_get_info(0)
    if free<9_000*1024*1024:p.error('Insufficient GPU VRAM')
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(args.base),local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    args.out.mkdir(exist_ok=False)
    result={}
    for label,adapter in [('base',None),('old_rank32',args.old_r32),('new_math_rank16',args.new_r16)]:
        model=_load_model(args.base)
        try:
            if adapter is not None:model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
            result[label]=_eval_model(label,model,tokenizer,args.out)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    (args.out/'details.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    s=summary(result)
    s.update(base_revision=snap['revision'],
        new_adapter_sha256=new_manifest['files']['adapter_model.safetensors'],
        old_adapter_sha256=old_manifest['files']['adapter_model.safetensors'])
    args.summary.parent.mkdir(exist_ok=True)
    args.summary.write_text(json.dumps(s,indent=2)+'\n',encoding='utf8')
    print(json.dumps(s,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

