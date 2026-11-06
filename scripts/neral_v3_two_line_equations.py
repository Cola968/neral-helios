#!/usr/bin/env python3
"""Convert verified v3 algebra into exact two-line transformations; no training.

Only 170 known algebra rows may change; all other 214 examples remain verbatim.
Input split hashes are frozen; new files and provenance remain local on Windows.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from scripts.neral_math_v2_builder import generate, TRAIN_SEED
from scripts.neral_math_v3_balanced_pack import new_equations
from scripts.neral_v3_split import verify_original_rows
from solve import solve

SOURCE_PINNED={
    'train':{
        'count':320,
        'sha256':'8ad8de4ebc3aee1bf7376bd97f3520000b795a6bb756d364de8c795a01f52ca0',
        'review_sha256':'0bc2c761d9968cd5f815a9b50314ea2a2cc1188821910165593316ca8adac391',
    },
    'validation':{
        'count':64,
        'sha256':'166ecff1e05088dadb3beb692541373c5293221df6e7b8a099b6a921c3128ee5',
        'review_sha256':'84613114481d9485b342c62bfc3359961e1d33d476934f03edc120fcaf108afe',
    },
}
EXPECTED_SOURCE_LINES=384
EXPECTED_TRAIN_EQUATIONS=142
EXPECTED_VALIDATION_EQUATIONS=28

P_ONE=re.compile(r'(-?\d+)x \+ \((-?\d+)\) = (-?\d+)')
P_TWO=re.compile(r'(-?\d+)x - (\d+) = (-?\d+)')
P_THREE=re.compile(r'(-?\d+)\(x \+ \((-?\d+)\)\) = (-?\d+)')


def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def source_section(i:int)->str:
    if type(i) is not int or not 1<=i<=EXPECTED_SOURCE_LINES:
        raise ValueError('Unrecognized original source line')
    if 127<=i<=196:return 'equation_old'
    if 197<=i<=296:return 'equation_new'
    return 'other'


def replacement_for_source_line(i:int):
    if source_section(i)=='equation_old':
        old=[x for x in generate(TRAIN_SEED) if x['family']=='equation']
        case=old[i-127];formula=case['formula']
        expected=case['expected'];completion=expected
        lang=case['language'];prompt=case['prompt']
    elif source_section(i)=='equation_new':
        case=new_equations()[i-197];formula=case['formula']
        expected=case['expected'];completion=case['completion']
        lang=case['language'];prompt=case['prompt']
    else:raise ValueError('Only the 170 equation source lines may change')

    a=b=c=None
    if m:=P_ONE.fullmatch(formula):
        a,b,c=map(int,m.groups()); target=c-b
    elif m:=P_TWO.fullmatch(formula):
        a,k,c=map(int,m.groups());target=c+k
    elif m:=P_THREE.fullmatch(formula):
        a,b,c=map(int,m.groups());target=c-a*b
    else:
        raise ValueError('Unrecognized equation family; no rewrite permitted')
    if not a or not re.fullmatch(r'-?\d+',expected):
        raise ValueError('Only nonzero-coefficient verified integer equations')
    x=int(expected)
    if Fraction(target,a)!=Fraction(x) or a*x!=target or solve(formula)['result']!='x = '+expected:
        raise ValueError('Equation does not prove its own expected solution')

    new_prompt = (
        f'Loese die Gleichung {formula}. Gib genau zwei Zeilen aus: '
        'zuerst den Umformungsschritt "ax = Zahl" mit dem echten Koeffizienten, '
        'danach "x = Zahl". Keine weiteren Woerter.'
        if lang=='de' else
        f'Solve the equation {formula}. Output exactly two lines: '
        'first the algebra step "ax = number" using the actual coefficient, '
        'then "x = number". No other text.'
    )
    two_line=f'{a}x = {target}\nx = {x}'
    if '\n' not in two_line or two_line.count('\n')!=1:
        raise ValueError('Two-line algebra contract broken')
    return prompt,completion,new_prompt,two_line


def adapt_row(row:dict,source_line:int):
    if not isinstance(row,dict) or set(row)!={'messages'}:
        raise ValueError('Unexpected QLoRA input')
    msgs=row['messages']
    if not isinstance(msgs,list) or len(msgs)!=3 or [z.get('role') for z in msgs]!=['system','user','assistant']:
        raise ValueError('Invalid QLoRA role sequence')
    if source_section(source_line)=='other':return copy.deepcopy(row),False
    old_prompt,old_completion,new_prompt,new_completion=replacement_for_source_line(source_line)
    if (msgs[1]['content']!=old_prompt or msgs[2]['content']!=old_completion):
        raise ValueError(f'Original algebra source line {source_line} differs from pinned curriculum')
    changed=copy.deepcopy(row)
    changed['messages'][1]['content']=new_prompt
    changed['messages'][2]['content']=new_completion
    return changed,True


def read_split(folder:Path,kind:str):
    expected=SOURCE_PINNED[kind]
    p=folder/(kind+'.jsonl')
    r=folder/(kind+'.review.json')
    for path in (p,r):
        if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:
            raise ValueError('Untrusted or missing local split')
    data=p.read_bytes();review_raw=r.read_bytes()
    if sha(data)!=expected['sha256'] or sha(review_raw)!=expected['review_sha256']:
        raise ValueError('Original v3 split hash differs')
    lines=data.decode('utf8').splitlines()
    manifest=json.loads(review_raw)
    if (len(lines)!=expected['count'] or len(manifest.get('rows',[]))!=len(lines) or
            manifest.get('dataset_sha256')!=expected['sha256']):
        raise ValueError('Split size or review coverage changed')
    if set(z.get('line') for z in manifest['rows'])!=set(range(1,len(lines)+1)):
        raise ValueError('Duplicate or missing review lines')
    by_row={z['line']:z for z in manifest['rows']}
    for n,line in enumerate(lines,1):
        review=by_row[n]
        if (review.get('row_sha256')!=sha(line.encode('utf8')) or
                not review.get('source_line') or not review.get('evidence') or not review.get('finding')):
            raise ValueError('Unverified source line or provenance')
    return lines,manifest


def prepare(source_folder:Path,out_folder:Path,source_code:Path):
    if (out_folder.exists() or out_folder.is_symlink() or
            not out_folder.parent.is_dir() or out_folder.parent.is_symlink() or
            source_code.is_symlink() or not source_code.is_file()):
        raise ValueError('Source or fresh destination not approved')
    originals={k:read_split(source_folder,k) for k in SOURCE_PINNED}
    from scripts.neral_v3_split import SECTION_SPECS
    complete=set()
    report={'schema':1,'status':'NERAL_V3_TWO_LINE_ALGEBRA_REVIEW_ONLY',
            'original_split_run':42283867443,
            'training_performed':False,'weights_promoted':False,
            'independent_holdout':False,'splits':{}}
    # Strictly create separate output from the original 320/64 split.
    out_folder.mkdir(exist_ok=False)
    key='scripts/neral_v3_two_line_equations.py'
    for kind,(lines,manifest) in originals.items():
        changed_lines=[];new_reviews=[];changed_count=0
        sources=dict(manifest['sources'])
        if key in sources:raise ValueError('Conversion already in source provenance')
        sources[key]=sha(source_code.read_bytes())
        for n,line in enumerate(lines,1):
            prev=manifest['rows'][n-1]
            source_line=prev['source_line']
            if source_line in complete:raise ValueError('Train-validation source overlap')
            complete.add(source_line)
            row=json.loads(line)
            rewritten,is_equation=adapt_row(row,source_line)
            changed_count+=int(is_equation)
            encoded=json.dumps(rewritten,ensure_ascii=False,separators=(',',':'))
            changed_lines.append(encoded)
            entry=copy.deepcopy(prev)
            entry['row_sha256']=sha(encoded.encode('utf8'))
            if is_equation:
                entry['evidence']+=['exact-algebra-source-reconstruction',
                                     'bounded-solve-verification',
                                     'verified-two-line-completion']
                entry['finding']='Two-line algebra and final x value checked using exact rational arithmetic; not a human review'
            new_reviews.append(entry)
        expected_equations=(EXPECTED_TRAIN_EQUATIONS if kind=='train' else EXPECTED_VALIDATION_EQUATIONS)
        if changed_count!=expected_equations:
            raise ValueError('Expected equation inventory differs; abort')
        out_data=('\n'.join(changed_lines)+'\n').encode('utf8')
        out_review={'schema':1,'dataset_sha256':sha(out_data),'sources':sources,
                    'rows':new_reviews,'reviewer':'Programmatically verified exact algebra, unchanged other row reviews; no training'}
        review_text=json.dumps(out_review,ensure_ascii=False,indent=2)+'\n'
        if len(out_data)>1024*1024 or len(review_text.encode('utf8'))>1024*1024:
            raise ValueError('Dataset exceeds pinned trainer limit')
        (out_folder/(kind+'.jsonl')).write_bytes(out_data)
        (out_folder/(kind+'.review.json')).write_text(review_text,encoding='utf8',newline='\n')
        report['splits'][kind]={'count':len(changed_lines),'converted_equations':changed_count,
                                'unchanged_other_examples':len(changed_lines)-changed_count,
                                'data_sha256':sha(out_data),
                                'review_sha256':sha(review_text.encode('utf8'))}
    if complete!=set(range(1,385)):
        raise ValueError('Train+validation do not cover exact original 384 source lines')
    (out_folder/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-in-trainer',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.split,a.out,a.source_in_trainer),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
