#!/usr/bin/env python3
"""Draft-only balanced NERAL core + verified algebra QLoRA pack; never trains.

Uses SHA-pinned existing Math-v2 local dataset, preserves original reviewed
rows, independently checks 100 NEW equation variants and avoids dev prompts.
R32 is a separate architecture experiment, not silently enabled here.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re

from scripts.neral_math_v2_builder import sha, norm, generate, TRAIN_SEED, NUMBER_ONLY_SYSTEM

EXPECTED_V2_DATA = '3e3b1e680d87706e075e656f15bf2c615bcf9dce9e4d8ccb23bc6b4659f98dfc'
EXPECTED_V2_REVIEW = '981e6ab4428316aff5d47bc3f63a18216b6e70a4cb902609637076b46cccf38a'
CORE,LEGACY,ORIGINAL_EQ,NEW_EQ,OTHER=86,40,70,100,88
SIZE=CORE+LEGACY+ORIGINAL_EQ+NEW_EQ+OTHER
SEED=61120264868


def build_plan():
    return {'schema':1,'status':'MATH_V3_DRAFT_REVIEW_ONLY','examples':SIZE,
            'core':CORE,'legacy_explanations':LEGACY,'previous_equations':ORIGINAL_EQ,
            'new_verified_equations':NEW_EQ,'other_math':OTHER,
            'core_percent':round(100*CORE/SIZE,2),'training_performed':False,
            'release_approved':False,'rank32_not_trained':True}


def new_equations(seed=SEED):
    rng=random.Random(seed);out=[];seen=set()
    for i in range(NEW_EQ):
        a=rng.choice([n for n in range(-17,18) if abs(n)>=2])
        x=rng.randint(-35,75); b=rng.randint(-55,80)
        mode='explain' if i%4==0 else 'numeric'
        variant=i%3
        if variant==0:
            formula=f'{a}x + ({b}) = {a*x+b}'
            intermediate=f'{a}x = {a*x}'
        elif variant==1:
            k=abs(b)+1
            formula=f'{a}x - {k} = {a*x-k}'
            intermediate=f'{a}x = {a*x}'
        else:
            formula=f'{a}(x + ({b})) = {a*(x+b)}'
            intermediate=f'x + ({b}) = {x+b}'
        lang='de' if i%2==0 else 'en'
        if mode=='numeric':
            prompt=(f'Bestimme den Wert von x aus {formula}. Nur den Zahlenwert.' if lang=='de'
                    else f'Find x for the equation {formula}. Give the number only.')
            completion=str(x)
        else:
            prompt=(f'Loese {formula}. Zeige einen Umformungsschritt und schliesse mit x = Zahl ab.' if lang=='de'
                    else f'Solve {formula}. Show an algebraic transformation, then end with x = number.')
            completion=f'{intermediate}; x = {x}'
        v={'formula':formula,'prompt':prompt,'expected':str(x),'completion':completion,
           'variant':variant,'mode':mode,'language':lang}
        if norm(prompt) in seen or not proof_equation(v):raise ValueError('Generated unverified/duplicate equation')
        seen.add(norm(prompt));out.append(v)
    return out


def proof_equation(row):
    """Independently parse the actual equation, re-solve with exact fractions."""
    expr=row['formula']
    p=[r'(-?\d+)x \+ \((-?\d+)\) = (-?\d+)',
       r'(-?\d+)x - (\d+) = (-?\d+)',
       r'(-?\d+)\(x \+ \((-?\d+)\)\) = (-?\d+)'][row['variant']]
    m=re.fullmatch(p,expr)
    if not m:return False
    a,b,c=map(int,m.groups())
    if not a:return False
    ans=(Fraction(c-b,a) if row['variant']==0 else
         Fraction(c+b,a) if row['variant']==1 else Fraction(c,a)-b)
    return ans.denominator==1 and str(ans.numerator)==row['expected']


def _read_checked(data:Path,review:Path):
    if any(p.is_symlink() for p in (data,review)):raise ValueError('Symlinked source')
    if sha(data.read_bytes())!=EXPECTED_V2_DATA or sha(review.read_bytes())!=EXPECTED_V2_REVIEW:
        raise ValueError('Math v2 source SHA-256 mismatch')
    manifest=json.loads(review.read_text(encoding='utf8'))
    lines=data.read_text(encoding='utf8').splitlines()
    if len(lines)!=744 or len(manifest.get('rows',[]))!=744 or manifest['dataset_sha256']!=EXPECTED_V2_DATA:
        raise ValueError('Unreviewed or unexpected v2 input')
    seen=set()
    by_line={v['line']:v for v in manifest['rows']}
    if set(by_line)!=set(range(1,745)):raise ValueError('Incomplete origin review')
    for i,line in enumerate(lines,1):
        r=json.loads(line)
        if [m['role'] for m in r['messages']]!=['system','user','assistant']:
            raise ValueError('Unreviewed message schema')
        if by_line[i]['row_sha256']!=sha(line.encode('utf8')):raise ValueError('Line review SHA mismatch')
        if not by_line[i].get('evidence') or not by_line[i].get('finding'):
            raise ValueError('Missing review evidence')
    expected=generate(TRAIN_SEED)
    for i,item in enumerate(expected,184):
        r=json.loads(lines[i]);a=r['messages']
        if a[1]['content']!=item['prompt'] or a[2]['content']!=item['expected']:
            raise ValueError('Original numeric curriculum unexpectedly changed')
    return lines,manifest


def prepare(v2_data:Path,v2_review:Path,source:Path,out:Path,holdout:Path):
    if out.exists() or out.is_symlink() or not out.parent.is_dir():raise ValueError('Output must be fresh')
    lines,manifest=_read_checked(v2_data,v2_review)
    from scripts.neral_math_v2_eval import cases as math_dev
    from scripts.neral_v2_generalization_probe import checked_cases
    protected={norm(x['prompt']) for x in math_dev()}
    protected.update(norm(v[2]) for v in checked_cases())
    from scripts.neral_math_v2_builder import load_holdout_prompts
    protected.update(load_holdout_prompts(holdout))
    select=[];observed=set();sources=dict(manifest['sources'])
    key='scripts/neral_math_v3_balanced_pack.py'
    if key in sources or source.is_symlink() or not source.is_file():raise ValueError('New reviewed source missing')
    sources[key]=sha(source.read_bytes())
    def add(text,previous=None,reason=''):
        row=json.loads(text);prompt=norm(row['messages'][1]['content'])
        if prompt in observed or prompt in protected:raise ValueError('Duplicate/held-out prompt')
        observed.add(prompt)
        evidence=(dict(previous) if previous is not None else
            {'evidence':['deterministic-new-equation','fraction-exact-independent-recalculation'],
             'finding':'Algebra and requested output format independently verified; no human review.'})
        evidence.update(line=len(select)+1,row_sha256=sha(text.encode('utf8')))
        select.append((text,evidence,reason))
    # Source order: old verified math 0:98, reviewed core 98:184, numeric 184:744.
    for i in range(98,184):add(lines[i],manifest['rows'][i],'core')
    legacy=sorted(range(98),key=lambda k:sha(lines[k].encode('utf8')))[:LEGACY]
    for i in legacy:add(lines[i],manifest['rows'][i],'legacy')
    generated=generate(TRAIN_SEED)
    eq_indices=[184+i for i,x in enumerate(generated) if x['family']=='equation']
    if len(eq_indices)!=ORIGINAL_EQ:raise ValueError('Incorrect original equation count')
    for i in eq_indices:add(lines[i],manifest['rows'][i],'original_equation')
    for row in new_equations():
        if not proof_equation(row):raise ValueError('New algebra failed proof')
        t={'messages':[{'role':'system','content':NUMBER_ONLY_SYSTEM},
            {'role':'user','content':row['prompt']},
            {'role':'assistant','content':row['completion']}]}
        text=json.dumps(t,ensure_ascii=False,separators=(',',':'))
        add(text,reason='new_equation_'+row['mode'])
    families=[f for f in ('add','subtract','multiply','percent','force','sequence','fraction')]
    count=0
    for k,family in enumerate(families):
        n=13 if k<4 else 12
        choices=[184+i for i,x in enumerate(generated) if x['family']==family]
        choices=sorted(choices,key=lambda idx:sha(lines[idx].encode('utf8')))[:n]
        for idx in choices:add(lines[idx],manifest['rows'][idx],'other_'+family)
        count+=len(choices)
    if count!=OTHER or len(select)!=SIZE:raise ValueError('Incorrect balanced pack inventory')
    data=('\n'.join(x[0] for x in select)+'\n').encode('utf8')
    reviews=[x[1] for x in select]
    result={'schema':1,'dataset_sha256':sha(data),'sources':sources,'rows':reviews,
            'reviewer':'Inherited pinned reviews, exact algebra verified; no release approval.'}
    review_text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if len(data)>1024*1024 or len(review_text.encode())>1024*1024:raise ValueError('QLoRA size gate exceeded')
    out.mkdir(exist_ok=False)
    (out/'train.jsonl').write_bytes(data)
    (out/'train.review.json').write_text(review_text,encoding='utf8',newline='\n')
    report=build_plan();report['source_v2_sha256']=EXPECTED_V2_DATA
    report['candidate_sha256']=sha(data);report['review_sha256']=sha(review_text.encode())
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('v2-data','v2-review','source','out','holdout'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.v2_data,a.v2_review,a.source,a.out,a.holdout),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
