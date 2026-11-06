#!/usr/bin/env python3
"""Split an already SHA-reviewed NERAL math-v3 corpus into frozen local 320/64.

No training, network, model loads, or promotions. The validation set is a
within-generator development check, NOT an independent generalization holdout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

EXPECTED_V3_DATA_SHA256='4b0e54712ea889cadd98eb668d315f3eb8647a7b9ac2209a1ed9942f7bbd3941'
EXPECTED_V3_REVIEW_SHA256='db1a460da4a19057877a32aa4bccc5814dd819442c4b678e7437ca9dca30771d'
SECTION_SPECS=(
    ('core',86,72),
    ('legacy',40,34),
    ('equation_old',70,58),
    ('equation_new',100,84),
    ('other_math',88,72),
)
SALT=b'NERAL-V3-STRATIFIED-SPLIT-20261106-v1'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_original_rows(lines: list[str], review: dict) -> None:
    if (len(lines)!=384 or not isinstance(review,dict) or review.get('schema')!=1 or
            review.get('dataset_sha256')!=sha(('\n'.join(lines)+'\n').encode('utf8'))):
        raise ValueError('Reviewed 384-row source inventory mismatch')
    if not isinstance(review.get('sources'),dict) or not review['sources']:
        raise ValueError('Source provenance missing')
    rows=review.get('rows')
    if not isinstance(rows,list) or len(rows)!=384:
        raise ValueError('Review must cover all 384 source rows')
    if {r.get('line') for r in rows if isinstance(r,dict)} != set(range(1,385)):
        raise ValueError('Row review line coverage incomplete')
    for i,(line,r) in enumerate(zip(lines,rows),1):
        if (r['line']!=i or r.get('row_sha256')!=sha(line.encode('utf8')) or
                not r.get('evidence') or not r.get('finding')):
            raise ValueError(f'Source line {i}: missing or wrong provenance')
        obj=json.loads(line)
        if (not isinstance(obj,dict) or set(obj)!={'messages'} or
                [x.get('role') for x in obj['messages']] != ['system','user','assistant']):
            raise ValueError('Unexpected QLoRA message structure')


def split_indices(lines: list[str]) -> tuple[list[int],list[int]]:
    if len(lines)!=384 or len(SECTION_SPECS)!=5:
        raise ValueError('Source length differs from frozen balanced pack')
    train=[];val=[]
    offset=0
    for _,count,take in SECTION_SPECS:
        positions=list(range(offset,offset+count))
        positions.sort(key=lambda i:sha(SALT+b'|'+lines[i].encode('utf8')))
        train.extend(positions[:take])
        val.extend(positions[take:])
        offset+=count
    if offset!=384 or len(train)!=320 or len(val)!=64 or set(train)&set(val):
        raise ValueError('Train/validation split size or disjointness failed')
    return sorted(train),sorted(val)


def slice_review(lines: list[str],source_review:dict, selected: list[int]):
    if not selected or len(set(selected))!=len(selected) or not all(0<=i<len(lines) for i in selected):
        raise ValueError('Invalid/duplicated selected indexes')
    records=[lines[i] for i in selected]
    raw=('\n'.join(records)+'\n').encode('utf8')
    rows=[]
    for n,i in enumerate(selected,1):
        entry=copy.deepcopy(source_review['rows'][i])
        entry['line']=n
        entry['source_line']=i+1
        entry['source_dataset']='MATH_V3_DRAFT_REVIEW_ONLY_SHA256_PINNED'
        rows.append(entry)
    result={'schema':1,'dataset_sha256':sha(raw),
            'sources':copy.deepcopy(source_review['sources']),
            'rows':rows,
            'reviewer':'Inherited exact original source reviews; deterministic local split, NOT independent holdout'}
    return records,result


def prepare(data:Path,review:Path,out:Path):
    data,review,out=Path(data),Path(review),Path(out)
    if (out.exists() or out.is_symlink() or not out.parent.is_dir() or
            out.parent.is_symlink() or data.is_symlink() or review.is_symlink()):
        raise ValueError('Source/output directory restrictions')
    if (not data.is_file() or not review.is_file() or
            data.stat().st_size>1024*1024 or review.stat().st_size>1024*1024):
        raise ValueError('Missing or oversized local source')
    if sha(data.read_bytes())!=EXPECTED_V3_DATA_SHA256 or sha(review.read_bytes())!=EXPECTED_V3_REVIEW_SHA256:
        raise ValueError('V3 source SHA-256 is not the immutable audited candidate')
    raw=data.read_bytes()
    if not raw.endswith(b'\n') or b'\r' in raw:
        raise ValueError('Non-canonical source line endings')
    lines=raw.decode('utf8').splitlines()
    original=json.loads(review.read_text(encoding='utf8'))
    verify_original_rows(lines,original)
    train,val=split_indices(lines)
    report={'schema':1,'status':'NERAL_V3_FROZEN_SPLIT_PREPARED',
            'source_sha256':EXPECTED_V3_DATA_SHA256,'source_review_sha256':EXPECTED_V3_REVIEW_SHA256,
            'total':384,'train':320,'validation':64,'sections':{},
            'data_shared_with_external_service':False,
            'training_performed':False,'release_validated':False,'independent_holdout':False}
    for name, count, take in SECTION_SPECS:
        report['sections'][name]={'total':count,'train':take,'validation':count-take}
    with TemporaryDirectory(prefix='.neral-v3-split-',dir=out.parent) as temp:
        tmp=Path(temp)
        for part,select in (('train',train),('validation',val)):
            chosen,manifest=slice_review(lines,original,select)
            dataset=('\n'.join(chosen)+'\n').encode('utf8')
            manifest_text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
            if len(dataset)>1024*1024 or len(manifest_text.encode('utf8'))>1024*1024:
                raise ValueError('Original trainer dataset/review size limit exceeded')
            (tmp/(part+'.jsonl')).write_bytes(dataset)
            (tmp/(part+'.review.json')).write_text(manifest_text,encoding='utf8',newline='\n')
            report[part+'_sha256']=sha(dataset)
            report[part+'_review_sha256']=sha(manifest_text.encode('utf8'))
        (tmp/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        if out.exists():
            raise ValueError('Destination appeared concurrently; refusing overwrite')
        tmp.rename(out)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--review',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(argv)
    print(json.dumps(prepare(a.data,a.review,a.out),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
