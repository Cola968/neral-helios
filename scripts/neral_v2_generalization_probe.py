#!/usr/bin/env python3
"""Additional read-only generalization diagnostic for the NERAL math-v2 candidate.

Public, manually authored developmental checks. No generated code executed; neither
results nor raw model responses should be treated as an independent release holdout.
No model download, training, adapter promotion or secrets upload.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import gc
import hashlib
import json
import re
from pathlib import Path

# Use different user phrasings, multi-step expressions and non-math formatting
# unlike the arithmetic generator's narrow training templates.
CASES = [
 ('multi-de-1','math','Berechne 36 mal die Differenz aus 7 und 4. Antworte nur mit der Zahl.','108'),
 ('multi-en-2','math','What is (28 + 17) divided by 5? Respond with a single number.','9'),
 ('multi-de-3','math','Was ist das Doppelte von 48, anschließend um 19 vermindert? Nur Zahl.','77'),
 ('multi-en-4','math','Half of 246, then add 17. Output the number only.','140'),
 ('multi-de-5','math','Rechne 15 mal 12 minus 25. Gib allein das Ergebnis zurück.','155'),
 ('multi-en-6','math','Take 8 squared and subtract 7. Give just the result.','57'),
 ('fraction-de-1','math','Ein Viertel plus ein Achtel ergibt welchen gekürzten Bruch? Nur Bruch.','3/8'),
 ('fraction-en-2','math','Compute two thirds minus one sixth. Return the simplified fraction only.','1/2'),
 ('fraction-de-3','math','Wie viel ist fünf Sechstel von 24? Nur Zahlenwert.','20'),
 ('fraction-en-4','math','One third of 27 plus one ninth of 27. Just the number.','12'),
 ('fraction-de-5','math','Rechne 7/12 plus 1/6. Als vollständig gekürzten Bruch.','3/4'),
 ('fraction-en-6','math','Divide 9/10 by 3. Reduced fraction only.','3/10'),
 ('signed-de-1','math','Berechne minus sieben multipliziert mit sechs. Nur die Zahl.','-42'),
 ('signed-en-2','math','What is -36 divided by 9? Return only the integer.','-4'),
 ('signed-de-3','math','Was ergibt -9 plus 4 mal 5? Nur Zahl.','11'),
 ('signed-en-4','math','Solve for x: 3x + 12 = 0. Answer just x.','-4'),
 ('word-de-1','math','Ein Zug fährt 150 Kilometer in zwei Stunden. Wie viele km/h im Mittel? Nur die Zahl.','75'),
 ('word-en-2','math','A box costs 80 euros; a discount reduces it by 15%. What is the final euro amount? Just the number.','68'),
 ('word-de-3','math','Ein Rechteck ist 13 cm lang und 9 cm breit. Wie groß ist der Umfang in cm? Nur Zahl.','44'),
 ('word-en-4','math','Convert 2.5 hours into minutes. Number only.','150'),
 ('word-de-5','math','Ein Fahrzeug benötigt 6 Liter auf 100 km. Wie viele Liter für 250 km? Nur Zahl.','15'),
 ('word-en-6','math','The sum of the integers from 1 through 10 is what number?','55'),
 ('general-de-1','math','Welche positive Zahl ist die Quadratwurzel aus 144? Nur Zahl.','12'),
 ('general-en-2','math','Compute 2 to the fifth power. Only the number.','32'),
 ('text-de-1','instruction','Antworte ausschließlich mit dem Wort BEREIT.','BEREIT'),
 ('text-en-2','instruction','Reply with exactly READY and nothing else.','READY'),
 ('text-de-3','instruction','Gib genau dieses Wort aus: Fenster','Fenster'),
 ('text-en-4','instruction','Say only the word cobalt.','cobalt'),
 ('text-de-5','instruction','Schreibe exakt JA, ohne weitere Zeichen.','JA'),
 ('text-en-6','instruction','Output only YES.','YES'),
 ('text-de-7','instruction','Antworte nur mit dem Text: abgeschlossen','abgeschlossen'),
 ('text-en-8','instruction','Return exactly: complete','complete'),
 ('text-de-9','instruction','Nur das Wort Sonnenlicht ausgeben.','Sonnenlicht'),
 ('text-en-10','instruction','Respond using exactly: verified','verified'),
 ('text-de-11','instruction','Schreibe genau OK ohne Satzzeichen.','OK'),
 ('text-en-12','instruction','Your entire response must be: final','final'),
]
SYSTEM = ('Du bist NERAL. Gib die verlangte Ausgabe exakt zurück. '
          'Mathematikaufgaben benötigen nur den verlangten Zahlenwert. '
          'Andere Aufgaben benötigen genau die ausdrücklich verlangte Antwort.')


def checked_cases():
    seen = set()
    for case_id, kind, prompt, expected in CASES:
        if case_id in seen or kind not in ('math', 'instruction') or not prompt or not expected:
            raise ValueError('Duplicate or invalid probe case')
        seen.add(case_id)
        if kind == 'math':
            Fraction(expected)
    return CASES


def grade(raw: str, kind: str, expected: str) -> dict:
    if kind == 'math':
        # Complete strict numeric answer; never award a hit to an explanation.
        clean = raw.strip() if isinstance(raw,str) else ""
        x = clean.replace(",", ".") if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?(?:/\d+)?", clean) else None
        if x is None:
            return {'format_valid':False,'correct':False}
        try:
            return {'format_valid':True,'correct':Fraction(x)==Fraction(expected)}
        except (ValueError, ZeroDivisionError):
            return {'format_valid':True,'correct':False}
    return {'format_valid':raw.strip()==expected,'correct':raw.strip()==expected}


def summary(results: dict) -> dict:
    expected = {x[0]:(x[1],x[3]) for x in checked_cases()}
    if set(results) != {'base','old_rank32','new_rank16_v2'}:
        raise ValueError('All three model comparisons required')
    scores = {}
    for label, rows in results.items():
        if len(rows)!=len(expected) or {x.get('id') for x in rows}!=set(expected):
            raise ValueError('Missing/duplicate probe responses')
        by_type=defaultdict(lambda:{'total':0,'correct':0,'format_valid':0})
        for row in rows:
            kind,_=expected[row['id']]
            by_type[kind]['total'] += 1
            by_type[kind]['correct'] += int(row['correct'] is True)
            by_type[kind]['format_valid'] += int(row['format_valid'] is True)
        scores[label]={'total':len(rows),
                       'correct':sum(v['correct'] for v in by_type.values()),
                       'by_type':dict(sorted(by_type.items()))}
    return {'schema':1,'status':'ADDITIONAL_PUBLIC_GENERALIZATION_DEV_CHECK',
            'cases':len(expected), 'independent_holdout':False,
            'general_improvement_proven':False, 'weights_promoted':False,
            'scores':scores}


def _run(label, model, tokenizer):
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    rows=[]
    for cid,kind,prompt,expected in checked_cases():
        turns=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
        ids=tokenizer.apply_chat_template(turns,chat_template=CHAT_TEMPLATE,
                 tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            completion=model.generate(ids,attention_mask=torch.ones_like(ids),
                do_sample=False,max_new_tokens=48,pad_token_id=tokenizer.pad_token_id)
        raw=tokenizer.decode(completion[0,ids.shape[-1]:],skip_special_tokens=True)
        check=grade(raw,kind,expected)
        rows.append({'id':cid,**check,'answer_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                     'local_answer':raw[:350]})
        print(f'{label}/{cid}: format={check["format_valid"]}, correct={check["correct"]}',flush=True)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False)
    for k in ('base','old-r32','new-r16','out','summary','trainer-root'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(argv)
    from scripts.neral_math_v2_eval import ensure_pinned_model_runtime
    ensure_pinned_model_runtime(a.trainer_root)
    if a.out.exists() or a.summary.exists() or a.out.is_symlink():
        p.error('Fresh external output required')
    for folder in (a.base,a.old_r32,a.new_r16):
        if folder.is_symlink() or not folder.is_dir():p.error('Trusted model folders required')
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate import validate_adapter as check_r16
    from neral_adapter_candidate_r32 import validate_adapter as check_r32
    s=validate_hf_snapshot(a.base,verify_hashes=True)
    old=check_r32(a.old_r32,s)
    new=check_r16(a.new_r16,s)
    if old['release_validated'] or new['release_validated']:p.error('Research adapters only')
    import torch
    from transformers import Qwen2TokenizerFast
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('BF16 CUDA unavailable')
    free,_=torch.cuda.mem_get_info(0)
    if free < 9_000*1024**2:p.error('Insufficient VRAM')
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(a.base),local_files_only=True,token=False,trust_remote_code=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    a.out.mkdir(exist_ok=False)
    results={}
    from scripts.neral_math_eval import _load_model
    for label, adapter in [('base',None),('old_rank32',a.old_r32),('new_rank16_v2',a.new_r16)]:
        model=_load_model(a.base)
        try:
            if adapter:model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
            results[label]=_run(label,model,tokenizer)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    # Local machine-only details to diagnose catastrophic format regressions.
    (a.out/'local_details.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report=summary(results)
    report['base_revision']=s['revision']
    report['old_adapter_sha256']=old['files']['adapter_model.safetensors']
    report['new_adapter_sha256']=new['files']['adapter_model.safetensors']
    a.summary.parent.mkdir(parents=True,exist_ok=True)
    a.summary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    # Only scalar summary is printed/uploaded. Raw answers stay private on host.
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()