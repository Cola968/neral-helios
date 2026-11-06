#!/usr/bin/env python3
"""Fail-closed arithmetic review of the sealed 430-row NeralTalk legacy candidate.

Only exactly regenerated math examples with independently checked numerical
arithmetic are admitted. No human-dialogue labels or model-generated code.
Does not train, download, or change model checkpoints. Data stays local.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

SOURCE_GIT_BLOB = 'b293c06e06634a816a75495cf8d18ed7af493437'
EXPECTED_CANDIDATE_SHA256 = '50c3c200913f6e40ed4b3975de39ecba795c1839b9983fdcd3029095b716fa9a'
EXPECTED_INTEGRATED_SHA256 = EXPECTED_CANDIDATE_SHA256
MAX_ROWS = 100


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def git_blob_sha(raw:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()


def message_text(row):
    if not isinstance(row,dict) or set(row)!={'messages'}:raise ValueError('Not a QLoRA row')
    ms=row['messages']
    if not isinstance(ms,list) or len(ms)!=3 or [x.get('role') for x in ms]!=['system','user','assistant']:
        raise ValueError('Expected system-user-assistant row')
    if any(not isinstance(x.get('content'),str) or not x['content'].strip() or set(x)!={'role','content'} for x in ms):
        raise ValueError('Invalid message content')
    return ms[1]['content'],ms[2]['content']


def check_math(kind:int, row:dict)->bool:
    """Independently compute the factual answer from the eight generator formats."""
    try:
        p,a=message_text(row)
        if kind == 0:
            m=re.search(r'(?:Berechne|Compute) (\d+) \+ (\d+)',p)
            if not m:return False
            x,y=map(int,m.groups());return f'{x} + {y} = {x+y}.' in a and a.rstrip().endswith(f'{x+y}.')
        if kind == 1:
            m=re.search(r'(?:Berechne|Compute) (\d+) - (\d+)',p)
            if not m:return False
            x,y=map(int,m.groups());return a.strip()==f'{x} - {y} = {x-y}.'
        if kind == 2:
            m=re.search(r'(?:Wie viel ist|What is) (\d+) (?:mal|times) (\d+)',p)
            if not m:return False
            x,y=map(int,m.groups());return a.strip() in (f'{x} · {y} = {x*y}.',f'{x} × {y} = {x*y}.')
        if kind == 3:
            m=re.search(r'(\d+)x \+ \((-?\d+)\) = (-?\d+)',p)
            if not m:return False
            x,y,z=map(int,m.groups())
            if x == 0 or (z-y)%x:return False
            sol=(z-y)//x
            return bool(re.search(rf'\bx\s*=\s*{sol}\.$',a.strip())) and f'{x}x = {z-y}.' in a
        if kind == 4:
            m=re.search(r'(\d+)% (?:von|of) (\d+)',p)
            if not m:return False
            x,y=map(int,m.groups());v=Fraction(x*y,100)
            if v.denominator!=1:return False
            return a.strip().endswith(f'= {v.numerator}.')
        if kind == 5:
            m=re.search(r'(\d+)/(\d+) [·×] (\d+)',p)
            if not m:return False
            num,den,mul=map(int,m.groups())
            if den==0:return False
            res=Fraction(num*mul,den)
            expected=str(res.numerator) if res.denominator==1 else f'{res.numerator}/{res.denominator}'
            return a.strip().endswith(f'= {expected}.') and f'{num*mul}/{den}' in a
        if kind == 6:
            m=re.search(r'm = (\d+) kg (?:und|and) a = (\d+) m/s²',p)
            if not m:return False
            x,y=map(int,m.groups());return a.strip().endswith(f'= {x*y} N.')
        if kind == 7:
            m=re.search(r'(-?\d+), (-?\d+), (-?\d+), (-?\d+), \.\.\.',p)
            if not m:return False
            vals=list(map(int,m.groups()));step=vals[1]-vals[0]
            if step <= 0 or vals[2]-vals[1]!=step or vals[3]-vals[2]!=step:return False
            return str(step) in a and bool(re.search(rf'(?:Zahl ist|number is) {vals[3]+step}\.$',a.strip()))
    except (TypeError,ValueError,ZeroDivisionError):
        return False
    return False


def choose_verified_math(examples, expected:dict[str,tuple[int,str]], *, max_rows=100):
    """Only exact prompt/answer pairs from pinned deterministic source survive."""
    if not 1<=max_rows<=100:raise ValueError('Unreviewed training size')
    stats=Counter();out=[];seen=set()
    for row in examples:
        p,a=message_text(row)
        if p in seen:
            stats['duplicate_prompt']+=1;continue
        seen.add(p)
        e=expected.get(p)
        if not e or e[1]!=a:
            stats['excluded_no_exact_generator_match']+=1;continue
        if not check_math(e[0],row):
            stats['excluded_math_disagrees']+=1;continue
        if len(out)<max_rows:
            out.append(row)
            stats['verified_math']+=1
        else:
            stats['excluded_quota']+=1
    stats['input']=len(examples)
    return out,dict(stats)


def build_review(dataset:Path, sources:dict[str,str], *, source_sha:str)->dict:
    lines=dataset.read_text(encoding='utf8').splitlines()
    if not lines or len(lines)>100:raise ValueError('No bounded reviewed examples')
    return {'schema':1,'reviewer':'Deterministic exact-source matching plus arithmetic verifier; NO human semantic approval',
            'human_semantic_review':False, 'dataset_sha256':file_sha(dataset),
            'sources':sources,
            'rows':[{'line':i,'row_sha256':hashlib.sha256(line.encode('utf8')).hexdigest(),
                     'kind':'arithmetic-independently-computed',
                     'evidence':['pinned-generator-original-template','independent-fraction-and-integer-check'],
                     'finding':'Computed arithmetic agrees with the exact source-template answer; no other domain reviewed.'}
                    for i,line in enumerate(lines,1)]}


def prepare(*,candidate:Path, trainer_root:Path,out:Path,max_rows:int=100)->dict:
    candidate=Path(candidate);trainer_root=Path(trainer_root);out=Path(out)
    if candidate.is_symlink() or out.exists() or out.is_symlink() or not out.parent.is_dir() or out.parent.is_symlink():
        raise ValueError('Input/output path restriction')
    if file_sha(candidate)!=EXPECTED_CANDIDATE_SHA256:raise ValueError('430 source candidate SHA mismatch')
    script=trainer_root/'build_targeted_sft50_sources.py'
    if script.is_symlink() or git_blob_sha(script.read_bytes())!=SOURCE_GIT_BLOB:
        raise ValueError('Unpinned generator')
    # Load only trusted exact git-blob-pinned generator; no calls to external services.
    sys.path.insert(0,str(trainer_root))
    import build_targeted_sft50_sources as gen
    expected={}
    for lang,seed in [('de',968611),('en',968612)]:
        for i,rec in enumerate(gen.generate_math(lang,6000,seed)):
            p=rec['prompt']; answer=rec['answer']
            if p in expected and expected[p]!=(i%8,answer):
                raise ValueError('Generator duplicate contradictory prompt')
            expected[p]=(i%8,answer)
    with candidate.open('r',encoding='utf8') as f:
        examples=[json.loads(line) for line in f if line.strip()]
    if len(examples)!=430:raise ValueError('Wrong number of legacy candidate rows')
    selected,stats=choose_verified_math(examples,expected,max_rows=max_rows)
    # Require at least 80 distinct independently verified examples.
    if len(selected)<80:raise ValueError('Not enough independently verified math examples')
    out.mkdir(exist_ok=False)
    dataset=out/'math_train.jsonl'
    with dataset.open('x',encoding='utf8',newline='\n') as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    sources={'build_targeted_sft50_sources.py':file_sha(script)}
    review=build_review(dataset,sources,source_sha=EXPECTED_CANDIDATE_SHA256)
    (out/'math_train.review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report={'status':'MATH_MACHINE_REVIEW_READY','training_performed':False,
            'reviewed_rows':len(selected),'candidate_sha256':EXPECTED_CANDIDATE_SHA256,
            'dataset_sha256':file_sha(dataset),'review_sha256':file_sha(out/'math_train.review.json'),
            'generator_git_blob':SOURCE_GIT_BLOB,'source_license':'Project-generated deterministic data',
            'human_semantic_review':False,'model_improvement_measured':False,
            'stats':stats}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--trainer-root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(argv)
    print(json.dumps(prepare(candidate=args.candidate,trainer_root=args.trainer_root,out=args.out),ensure_ascii=False))

if __name__=='__main__':main()

