#!/usr/bin/env python3
"""Single-GPU, no-tools Rank16/Rank32 QLoRA development comparison.

Qwen baseline is measured using PeftModel.disable_adapter(); each rank's LoRA
is evaluated with the exact same underlying model, with one adapter loaded at
a time. No weights merged, no generated code executed, no automatic promotion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import gc
import hashlib
import json
from math import comb
from pathlib import Path
import re

from scripts.neral_math_v2_eval import cases as math80, ensure_pinned_model_runtime
from scripts.neral_math_eval import SYSTEM as MATH_SYSTEM, _load_model, normalized_number
from scripts.neral_v2_generalization_probe import CASES as GENERALIZATION36, SYSTEM as GEN_SYSTEM

MODEL_KEYS=('base','rank16_v3','rank32_v3')
SYSTEM='Du bist NERAL. Gib exakt das verlangte Antwortformat zurück.'
# This is a separate, newly constructed engineering holdout, NOT an external
# standardized or blinded benchmark, and it is not used to fit the adapters.
HOLDOUT=(
 {'id':'h01','kind':'numeric','prompt':'12 plus 7, anschließend minus 2. Nur Zahl.','expected':'17'},
 {'id':'h02','kind':'numeric','prompt':'What is 19 multiplied by 3, then minus 8? Give only the result.','expected':'49'},
 {'id':'h03','kind':'numeric','prompt':'72 geteilt durch 9, dann plus 6. Nur Zahl.','expected':'14'},
 {'id':'h04','kind':'numeric','prompt':'How much is 15 percent of 240? Output a number only.','expected':'36'},
 {'id':'h05','kind':'numeric','prompt':'Berechne zwei Drittel von 81. Nur Zahl.','expected':'54'},
 {'id':'h06','kind':'numeric','prompt':'Multiply the sum of 17 and 5 by 4. Number only.','expected':'88'},
 {'id':'h07','kind':'numeric','prompt':'90 km pro Stunde, zwei Stunden: Strecke in km? Nur Zahl.','expected':'180'},
 {'id':'h08','kind':'numeric','prompt':'Five buckets of 12 liters, then remove 7 liters. Just number.','expected':'53'},
 {'id':'h09','kind':'algebra','prompt':'Löse 4x + 7 = 39. Gib zwei Zeilen aus: ax = Zahl, dann x = Zahl.','expected':'4x = 32\nx = 8'},
 {'id':'h10','kind':'algebra','prompt':'Solve -5x + 10 = 35. Exactly 2 lines: ax = number; x = number.','expected':'-5x = 25\nx = -5'},
 {'id':'h11','kind':'algebra','prompt':'Löse 6(x - 2) = 48. Zeige ax = Zahl und darunter x = Zahl.','expected':'6x = 60\nx = 10'},
 {'id':'h12','kind':'algebra','prompt':'Solve 3x - 9 = 12. Two lines, ax = number then x = number.','expected':'3x = 21\nx = 7'},
 {'id':'h13','kind':'algebra','prompt':'Löse -2(x + 3) = 10. Nur zwei Zeilen: ax = Zahl; x = Zahl.','expected':'-2x = 16\nx = -8'},
 {'id':'h14','kind':'algebra','prompt':'Solve 7x + 14 = 0, two lines: ax = number then x = number.','expected':'7x = -14\nx = -2'},
 {'id':'h15','kind':'instruction','prompt':'Schreibe exakt diese Zeichenfolge: NERAL-PRUEFUNG','expected':'NERAL-PRUEFUNG'},
 {'id':'h16','kind':'instruction','prompt':'Return exactly this string: AIR-27','expected':'AIR-27'},
 {'id':'h17','kind':'instruction','prompt':'Antworte mit exakt einem Wort: überprüft','expected':'überprüft'},
 {'id':'h18','kind':'instruction','prompt':'Output exactly OK-409, with no quotes.','expected':'OK-409'},
 {'id':'h19','kind':'instruction','prompt':'Gib exakt die Zeichenfolge abc-123 zurück.','expected':'abc-123'},
 {'id':'h20','kind':'instruction','prompt':'Reply with exactly verified_72.','expected':'verified_72'},
)


def validate_suite(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate or empty held-out evaluation suite')
    for r in rows:
        if (set(r)!={'id','kind','prompt','expected'} or
                r['kind'] not in ('numeric','algebra','instruction') or
                not isinstance(r['prompt'],str) or not 0<len(r['prompt'])<=600):
            raise ValueError('Invalid holdout schema')
        if r['kind']=='algebra' and r['expected'].count('\n')!=1:
            raise ValueError('Two-line holdout contract violated')
    return True


def suite_score(row,raw):
    if not isinstance(raw,str):
        return False,False
    raw=raw.strip()
    if row['kind']=='numeric':
        value=normalized_number(raw)
        if value is None:
            return False,False
        try:
            return Fraction(value)==Fraction(row['expected']),True
        except (ZeroDivisionError,ValueError):
            return False,True
    if row['kind']=='algebra':
        # Exact format AND algebra, already independently computed for the
        # manually authored holdout cases. Other equivalent formatting is
        # recorded as a failure for this strict contract test.
        valid=bool(re.fullmatch(r'-?\d+x = -?\d+\nx = -?\d+(?:/\d+)?',raw))
        return raw==row['expected'], valid
    return raw==row['expected'],raw==row['expected']



def exact_mcnemar_p(improvements: int, regressions: int) -> float:
    """Two-sided exact paired sign/McNemar test, without external dependencies.

    Scores are evidence for this *development* set only. A low p-value alone
    cannot establish general improvement, independence or release quality.
    """
    if (type(improvements) is not int or type(regressions) is not int
            or improvements < 0 or regressions < 0):
        raise ValueError('Paired discordant counts must be nonnegative integers')
    discordant = improvements + regressions
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(improvements, regressions) + 1))
    return min(1.0, (2 * tail) / (2 ** discordant))


def _paired_counts(baseline: list[dict], candidate: list[dict]) -> dict:
    """Compare matched case IDs, never different case order or aggregate totals."""
    base = {r['id']: r for r in baseline}
    other = {r['id']: r for r in candidate}
    if set(base) != set(other):
        raise ValueError('Paired comparison requires matched case ids')
    improvements = sum(not base[i]['correct'] and other[i]['correct'] for i in base)
    regressions = sum(base[i]['correct'] and not other[i]['correct'] for i in base)
    format_gains = sum(not base[i]['format_valid'] and other[i]['format_valid'] for i in base)
    format_losses = sum(base[i]['format_valid'] and not other[i]['format_valid'] for i in base)
    return {
        'total': len(base),
        'improvements': improvements,
        'regressions': regressions,
        'net_gain': improvements - regressions,
        'discordant': improvements + regressions,
        'unchanged_correct': sum(base[i]['correct'] and other[i]['correct'] for i in base),
        'unchanged_incorrect': sum(not base[i]['correct'] and not other[i]['correct'] for i in base),
        'format_improvements': format_gains,
        'format_regressions': format_losses,
        'exact_mcnemar_p': exact_mcnemar_p(improvements, regressions),
    }


def summarize(results,counts):
    if set(results)!=set(MODEL_KEYS):
        raise ValueError('All three model variants required')
    out={}
    for model in MODEL_KEYS:
        if set(results[model])!=set(counts):
            raise ValueError('Missing evaluation suite')
        out[model]={}
        for suite,n in counts.items():
            rows=results[model][suite]
            if (not isinstance(rows, list) or len(rows)!=n or
                    any(not isinstance(r, dict) or not isinstance(r.get('id'), str)
                        or not r['id'] or len(r['id']) > 128 for r in rows)):
                raise ValueError('Incomplete or malformed inference results')
            if len({r['id'] for r in rows})!=n:
                raise ValueError('Incomplete or duplicate inference results')
            if any(type(r.get('correct')) is not bool or
                   type(r.get('format_valid')) is not bool for r in rows):
                raise ValueError('Paired evaluation requires boolean scores')
            out[model][suite]={'correct':sum(x['correct'] for x in rows),
                               'format_valid':sum(x['format_valid'] for x in rows),
                               'total':n}
    paired={model: {suite: _paired_counts(results['base'][suite], results[model][suite])
                    for suite in counts}
            for model in ('rank16_v3','rank32_v3')}
    return {'schema':1,'status':'NERAL_V3_RANK16_RANK32_DEV_COMPARISON',
            'scores':out,
            'paired_vs_base':paired,
            'development_suite_80':True,'prior_used_suite_36':True,
            'within_generator_validation':True,
            'fresh_in_repo_heldout20':True,
            'independent_external_benchmark':False,
            'general_improvement_proven':False,
            'weights_promoted':False,
            'note':'Not a general quality or release result; strict formatting and developer-authored tests.'}


def build_suites(val_rows,train_rows):
    validate_suite(HOLDOUT)
    seen={x['prompt'][1]['content'].casefold().strip() for x in train_rows+val_rows}
    for r in HOLDOUT:
        if r['prompt'].casefold().strip() in seen:
            raise ValueError('Holdout prompt overlaps with v3 dataset')
    suites={'validation64':[],'math80':[],'phrasing36':[],'holdout20':[]}
    for i,row in enumerate(val_rows,1):
        assistant=row['completion'][0]['content']
        typ='algebra' if ('\n' in assistant and re.match(r'-?\d+x\s*=',assistant)) else (
            'numeric' if normalized_number(assistant) is not None else 'instruction')
        suites['validation64'].append({'id':f'v{i:03d}','kind':typ,
            'messages':row['prompt'],'expected':assistant})
    for item in math80():
        suites['math80'].append({'id':item['id'],'kind':'numeric',
            'messages':[{'role':'system','content':MATH_SYSTEM},
                        {'role':'user','content':item['prompt']}],
            'expected':item['expected']})
    for case_id,kind,prompt,expected in GENERALIZATION36:
        suites['phrasing36'].append({'id':case_id,'kind':'numeric' if kind=='math' else 'instruction',
            'messages':[{'role':'system','content':GEN_SYSTEM},
                        {'role':'user','content':prompt}],'expected':expected})
    for row in HOLDOUT:
        suites['holdout20'].append({'id':row['id'],'kind':row['kind'],
            'messages':[{'role':'system','content':SYSTEM},
                        {'role':'user','content':row['prompt']}],
            'expected':row['expected']})
    return suites


def evaluate_variant(model,tokenizer,suites):
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    all_rows={}
    for name,cases in suites.items():
        gathered=[]
        for case in cases:
            ids=tokenizer.apply_chat_template(case['messages'],chat_template=CHAT_TEMPLATE,
                tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                out=model.generate(ids,attention_mask=torch.ones_like(ids),
                    do_sample=False,max_new_tokens=144,pad_token_id=tokenizer.pad_token_id)
            raw=tokenizer.decode(out[0,ids.shape[-1]:],skip_special_tokens=True)
            del out, ids
            correct,fmt=suite_score(case,raw)
            gathered.append({'id':case['id'],'correct':bool(correct),'format_valid':bool(fmt),
                             'answer_sha256':hashlib.sha256(raw.encode('utf8')).hexdigest(),
                             'local_raw':raw[:500]})
        all_rows[name]=gathered
        print(name+': cases='+str(len(gathered))+' correct='+str(sum(x['correct'] for x in gathered)),flush=True)
    return all_rows


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('base','rank16','rank32','trainer-root','data-root','out'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--variant',choices=MODEL_KEYS,default=None)
    p.add_argument('--combine-results',action='store_true')
    a=p.parse_args(argv)
    if a.variant and a.combine_results:
        p.error('Choose variant or combine, not both')
    if a.out.exists() or a.out.is_symlink() or not a.out.parent.is_dir():
        p.error('Evaluation directory must be new')
    ensure_pinned_model_runtime(a.trainer_root)
    import torch
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter as validate_r16
    from neral_adapter_candidate_r32 import validate_adapter as validate_r32
    from transformers import Qwen2TokenizerFast
    from peft import PeftModel
    snapshot=validate_hf_snapshot(a.base,verify_hashes=True)
    if snapshot['revision']!='c03e6d358207e414f1eca0bb1891e29f1db0e242':
        p.error('Pinned Qwen base revision missing')
    v16=validate_r16(a.rank16,snapshot);v32=validate_r32(a.rank32,snapshot)
    if v16['release_validated'] or v32['release_validated']:
        p.error('No production release adapter can be used in development comparison')
    from scripts.neral_v3_two_line_token_audit import audit as _audit
    # For this evaluator, provenance review runs using the immutable original
    # tokenizer/trainer, but data itself is never modified.
    from scripts.neral_v3_token_audit import load_pinned_trainer
    pilot=load_pinned_trainer(a.trainer_root)
    train=a.data_root/'train.jsonl';validation=a.data_root/'validation.jsonl'
    pilot.verify_review(train,a.data_root/'train.review.json')
    pilot.verify_review(validation,a.data_root/'validation.review.json')
    train_rows=pilot.read_examples(train);val_rows=pilot.read_examples(validation)
    if len(train_rows)!=320 or len(val_rows)!=64:
        p.error('Expected reviewed 320/64 split')
    suites=build_suites(val_rows,train_rows)
    if len(suites['math80'])!=80 or len(suites['phrasing36'])!=36:
        p.error('Known development suites changed')
    if a.combine_results:
        from scripts.neral_v3_eval_isolation import collect_variant_outputs
        val_sha=hashlib.sha256(validation.read_bytes()).hexdigest()
        results=collect_variant_outputs(a.out.parent,{k:len(v) for k,v in suites.items()},
            v16['files']['adapter_model.safetensors'],
            v32['files']['adapter_model.safetensors'],val_sha)
        # No CUDA model is ever loaded when combining child-process outputs.
        report=summarize(results,{k:len(v) for k,v in suites.items()})
        report['adapter_16_sha256']=v16['files']['adapter_model.safetensors']
        report['adapter_32_sha256']=v32['files']['adapter_model.safetensors']
        report['validation_sha256']=val_sha
        a.out.mkdir(exist_ok=False)
        (a.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(a.base),local_files_only=True,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('CUDA BF16 required')
    a.out.mkdir(exist_ok=False)
    if a.variant:
        from scripts.neral_v3_eval_isolation import clean_cuda, prepare_cuda
        model=None
        prepare_cuda(torch)
        try:
            model=_load_model(a.base)
            if a.variant!='base':
                adapter=a.rank16 if a.variant=='rank16_v3' else a.rank32
                model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
            results=evaluate_variant(model,tokenizer,suites)
            record={'model':a.variant,'results':results,
                'adapter16_sha256':v16['files']['adapter_model.safetensors'],
                'adapter32_sha256':v32['files']['adapter_model.safetensors'],
                'validation_sha256':hashlib.sha256(validation.read_bytes()).hexdigest()}
            (a.out/'local-outputs.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print('Completed variant '+a.variant+'; raw answers kept on Windows only',flush=True)
        finally:
            if model is not None:
                del model
            clean_cuda(torch)
        return
    results={}
    for rank,label in [(16,'rank16_v3'),(32,'rank32_v3')]:
        torch.cuda.empty_cache()
        free,_=torch.cuda.mem_get_info(0)
        if free<10000*1024**2:
            p.error('GPU free memory below 10000 MiB')
        model=_load_model(a.base)
        try:
            adapter=a.rank16 if rank==16 else a.rank32
            model=PeftModel.from_pretrained(model,str(adapter),
                is_trainable=False,local_files_only=True)
            if rank==16:
                with model.disable_adapter():
                    results['base']=evaluate_variant(model,tokenizer,suites)
            results[label]=evaluate_variant(model,tokenizer,suites)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    report=summarize(results,{k:len(v) for k,v in suites.items()})
    report['adapter_16_sha256']=v16['files']['adapter_model.safetensors']
    report['adapter_32_sha256']=v32['files']['adapter_model.safetensors']
    report['validation_sha256']=hashlib.sha256(validation.read_bytes()).hexdigest()
    # Raw outputs never leave this local Windows directory.
    (a.out/'local-outputs.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (a.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
