#!/usr/bin/env python3
"""Offline audited NERAL math-v2: legacy math + existing reviewed 86 + exact-answer curriculum.

Never trains or uploads text. All synthetic answers are independently recomputed
from the stated arithmetic operation. Generation and dev-evaluation use distinct
seeds, templates and values. Development evaluation is not a frozen public holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re

LEGACY_SHA='8c2452cdb4b70d0d7538b98af0710bbf7554eaf0dbd6997d8dc8bb12a74b7809'
LEGACY_REVIEW_SHA='e2dab7c809c47be0b51c4b07bb359c35b66b4328ffc847bfec1f3975a7549e88'
NUMBER_ONLY_SYSTEM='Du bist NERAL, basierend auf Qwen. Befolge exakt das verlangte Antwortformat. Rechne präzise und erfinde keine Tools.'
TRAIN_SEED=9686112026
EVAL_SEED=6006112026
COUNT_PER_FAMILY=70
FAMILIES=('add','subtract','multiply','percent','force','equation','sequence','fraction')


def sha(raw: bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def norm(prompt: str)->str:
    return ' '.join(prompt.casefold().split())


def answer(value: Fraction|int)->str:
    v=Fraction(value)
    return str(v.numerator) if v.denominator==1 else f'{v.numerator}/{v.denominator}'


def make_problem(kind: str, rng: random.Random, lang: str, *, eval_style=False):
    """Structured math facts; independently verifiable answer text."""
    if kind=='add':
        a,b=rng.randint(101,4000),rng.randint(102,4000)
        expression=f'{a} + {b}'; expected=Fraction(a+b)
    elif kind=='subtract':
        a,b=rng.randint(202,5500),rng.randint(201,4200)
        expression=f'{a} - {b}';expected=Fraction(a-b)
    elif kind=='multiply':
        a,b=rng.randint(13,92),rng.randint(12,91)
        expression=f'{a} * {b}';expected=Fraction(a*b)
    elif kind=='percent':
        p=rng.choice((5,10,15,20,25,30,40,50,75));base=20*rng.randint(6,120)
        expression=f'{p}% of {base}';expected=Fraction(p*base,100)
    elif kind=='force':
        m,a=rng.randint(6,140),rng.randint(3,30)
        expression=f'F = {m} * {a}';expected=Fraction(m*a)
    elif kind=='equation':
        a=rng.randint(2,17);x=rng.randint(-15,75);b=rng.randint(-50,120)
        expression=f'{a}x + ({b}) = {a*x+b}';expected=Fraction(x)
    elif kind=='sequence':
        start=rng.randint(-20,175);step=rng.randint(2,25)
        expression=', '.join(str(start+i*step) for i in range(4))
        expected=Fraction(start+4*step)
    elif kind=='fraction':
        n,d,k=rng.randint(2,19),rng.randint(2,15),rng.randint(2,12)
        expression=f'{n}/{d} * {k}';expected=Fraction(n*k,d)
    else:raise ValueError('Unrecognized family')
    if eval_style:
        patterns={
          'de':{'add':f'Addiere {expression}. Gib nur das Ergebnis an.',
                'subtract':f'Wie lautet {expression}? Nur das Ergebnis.',
                'multiply':f'Berechne {expression}. Schreibe ausschließlich eine Zahl.',
                'percent':f'Bestimme {expression}. Antworte nur mit einer Zahl.',
                'force':f'Newton: {expression}. Welcher Zahlenwert ergibt sich? Nur die Zahl.',
                'equation':f'Finde x: {expression}. Nur der Zahlenwert.',
                'sequence':f'Führe die Folge {expression} fort. Nur die nächste Zahl.',
                'fraction':f'Berechne {expression}. Nur den gekürzten Bruch oder die Zahl.'},
          'en':{'add':f'What is {expression}? Just the result.',
                'subtract':f'Evaluate {expression}. Give only one number.',
                'multiply':f'Multiply: {expression}. Only the numeric result.',
                'percent':f'How much is {expression}? Numeric response only.',
                'force':f'Physics: {expression}. What numeric force? Number only.',
                'equation':f'Find x in {expression}. Just its value.',
                'sequence':f'Continue {expression}. Return only the next value.',
                'fraction':f'Simplify {expression}. Give just the final rational number.'}}
    else:
        patterns={
          'de':{'add':f'Rechne {expression}. Antworte ausschließlich mit der Zahl.',
                'subtract':f'Berechne {expression}. Nur die Zahl.',
                'multiply':f'Wie viel ergibt {expression}? Schreibe nur das Ergebnis.',
                'percent':f'Berechne {expression}. Nur den Zahlenwert.',
                'force':f'Bestimme den Zahlenwert der Kraft: {expression}. Nur die Zahl.',
                'equation':f'Löse die Gleichung {expression}. Antworte nur mit x als Zahl.',
                'sequence':f'Welche Zahl folgt auf {expression}? Nur die Zahl.',
                'fraction':f'Berechne {expression}. Gib nur den gekürzten Bruch oder die Zahl aus.'},
          'en':{'add':f'Calculate {expression}. Answer with the number only.',
                'subtract':f'Compute {expression}. Only the number.',
                'multiply':f'What is {expression}? Return just the result.',
                'percent':f'Calculate {expression}. Respond with the number.',
                'force':f'Calculate force: {expression}. Just the number.',
                'equation':f'Solve {expression}. Answer with x as a number only.',
                'sequence':f'What comes after {expression}? Just one number.',
                'fraction':f'Compute {expression}. Return the reduced fraction or number only.'}}
    prompt=patterns[lang][kind]
    return {'family':kind,'prompt':prompt,'expected':answer(expected),'formula':expression,'language':lang}


def generate(seed:int,each:int=COUNT_PER_FAMILY,*,eval_style=False)->list[dict]:
    if each<1 or each>100:raise ValueError('Sample quota out of range')
    rng=random.Random(seed)
    seen=set();cases=[]
    for kind in FAMILIES:
        for i in range(each):
            lang='de' if i%2==0 else 'en'
            for tries in range(50):
                x=make_problem(kind,rng,lang,eval_style=eval_style)
                if norm(x['prompt']) not in seen:break
            else:raise ValueError('Excess generated duplicates')
            seen.add(norm(x['prompt']))
            cases.append(x)
    return cases


def json_line(row:dict)->str:
    return json.dumps(row,ensure_ascii=False,separators=(',',':'))


def to_qlora(row:dict)->dict:
    return {'messages':[{'role':'system','content':NUMBER_ONLY_SYSTEM},
                        {'role':'user','content':row['prompt']},
                        {'role':'assistant','content':row['expected']}]}


def validate_expected(case:dict)->bool:
    """Separate parser and rational arithmetic check of structured derivation."""
    family,expr=case['family'],case['formula']
    nums=[int(x) for x in re.findall(r'-?\d+',expr)]
    try:
        if family=='add':v=Fraction(nums[0]+nums[1])
        elif family=='subtract':v=Fraction(nums[0]-nums[1])
        elif family=='multiply':v=Fraction(nums[0]*nums[1])
        elif family=='percent':v=Fraction(nums[0]*nums[1],100)
        elif family=='force':v=Fraction(nums[0]*nums[1])
        elif family=='equation':
            m=re.fullmatch(r'(-?\d+)x \+ \((-?\d+)\) = (-?\d+)',expr)
            if not m:return False
            a,b,c=map(int,m.groups());v=Fraction(c-b,a)
        elif family=='sequence':
            if len(nums)!=4 or nums[2]-nums[1]!=nums[1]-nums[0] or nums[3]-nums[2]!=nums[1]-nums[0]:return False
            v=Fraction(nums[3]+nums[1]-nums[0])
        elif family=='fraction':v=Fraction(nums[0]*nums[2],nums[1])
        else:return False
        return answer(v)==case['expected']
    except (IndexError,ValueError,ZeroDivisionError):return False


def read_and_verify(dataset:Path, review_path:Path, *,expected_data_sha:str|None=None,expected_review_sha:str|None=None):
    if dataset.is_symlink() or review_path.is_symlink():raise ValueError('Symlinked input')
    data=dataset.read_bytes();rbytes=review_path.read_bytes()
    if expected_data_sha and sha(data)!=expected_data_sha:raise ValueError('Legacy data SHA mismatch')
    if expected_review_sha and sha(rbytes)!=expected_review_sha:raise ValueError('Legacy review SHA mismatch')
    review=json.loads(rbytes); lines=data.decode('utf8').splitlines()
    if review.get('dataset_sha256')!=sha(data) or len(lines)!=len(review.get('rows',[])):raise ValueError('Review coverage mismatch')
    by_line={v['line']:v for v in review['rows']}
    if len(by_line)!=len(lines):raise ValueError('Duplicate review lines')
    for i,line in enumerate(lines,1):
        if by_line[i]['row_sha256']!=sha(line.encode('utf8')):raise ValueError('Row hash mismatch')
    return lines,review


def build(legacy:Path,legacy_review:Path,pack:Path,pack_review:Path,out:Path,source_in_trainer:Path,protected_prompts:set[str]):
    if out.exists() or out.is_symlink() or not out.parent.is_dir():raise ValueError('Output exists/parent unavailable')
    old,rm=read_and_verify(legacy,legacy_review,expected_data_sha=LEGACY_SHA,expected_review_sha=LEGACY_REVIEW_SHA)
    if len(old)!=98:raise ValueError('Not the sealed 98-row legacy curriculum')
    stable,pr=read_and_verify(pack,pack_review)
    if len(stable)!=86:raise ValueError('Not the reviewed 86-row bundle')
    cases=generate(TRAIN_SEED)
    eval_cases=generate(EVAL_SEED,each=12,eval_style=True)
    if {norm(x['prompt']) for x in cases} & {norm(x['prompt']) for x in eval_cases}:raise ValueError('Synthetic evaluation leakage')
    selected=[]; reviews=[]; seen=set(); sources={**rm['sources'],**pr['sources']}
    source_key='scripts/neral_math_v2_builder.py'
    if source_key in sources or source_in_trainer.is_symlink() or not source_in_trainer.is_file():raise ValueError('Source placement invalid')
    sources[source_key]=sha(source_in_trainer.read_bytes())
    def add(line:str,evidence:list[str],finding:str):
        row=json.loads(line)
        if set(row)!={'messages'}:raise ValueError('Non-QLoRA row')
        messages=row['messages']
        if [m['role'] for m in messages]!=['system','user','assistant']:raise ValueError('Bad role order')
        prompt=norm(messages[1]['content'])
        if prompt in seen or prompt in protected_prompts:raise ValueError('Duplication or frozen holdout conflict')
        seen.add(prompt)
        selected.append(line)
        reviews.append({'line':len(selected),'row_sha256':sha(line.encode('utf8')),'kind':'source_verified',
                       'evidence':evidence,'finding':finding})
    for i,line in enumerate(old):
        r=rm['rows'][i];add(line,list(r['evidence']),r['finding'])
    for i,line in enumerate(stable):
        r=pr['rows'][i];add(line,list(r['evidence']),r['finding'])
    for case in cases:
        if not validate_expected(case):raise ValueError('Independent arithmetic disagreement')
        line=json_line(to_qlora(case))
        add(line,['deterministic_math_family:'+case['family'],'independent_fraction_or_integer_recalculation'],
            'Numeric completion independently checked against structured operation; no human review claimed.')
    if len(selected)!=744:raise ValueError('Unexpected training size')
    data='\n'.join(selected)+'\n'
    review={'schema':1,'reviewer':'Reused reviewed legacy + 86 sources, deterministic independent arithmetic; not human certified',
            'human_semantic_review':False,'dataset_sha256':sha(data.encode('utf8')),
            'sources':sources,'rows':reviews}
    if len(data.encode('utf8'))>1024*1024:raise ValueError('Trainer dataset size exceeded')
    review_text=json.dumps(review,ensure_ascii=False,indent=2)+'\n'
    if len(review_text.encode('utf8'))>1024*1024:raise ValueError('Trainer review size exceeded')
    out.mkdir(exist_ok=False)
    (out/'train.jsonl').write_bytes(data.encode('utf8'))
    (out/'train.review.json').write_text(review_text,encoding='utf8',newline='\n')
    summary={'schema':1,'status':'MATH_V2_PREPARED','legacy_math':98,'reviewed_project':86,
             'independently_checked_numeric':len(cases),'total':len(selected),
             'data_sha256':review['dataset_sha256'],'review_sha256':sha(review_text.encode('utf8')),
             'approved_for_training':False,'training_performed':False,'promoted':False,
             'independent_holdout':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    return summary


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for x in ('legacy','legacy-review','pack','pack-review','out','source-in-trainer','holdout'):
        p.add_argument('--'+x,required=True,type=Path)
    a=p.parse_args()
    holdout=json.loads(a.holdout.read_text(encoding='utf8'))
    protected={norm(c['prompt']) for c in holdout}
    from scripts.neral_math_eval import CASES
    protected.update(norm(c['prompt']) for c in CASES)
    result=build(a.legacy,a.legacy_review,a.pack,a.pack_review,a.out,a.source_in_trainer,protected)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
