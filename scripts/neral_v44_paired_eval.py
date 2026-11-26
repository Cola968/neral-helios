"""Offline aggregation of independently graded paired development-holdout receipts.

Does not generate model answers, grade them, open a model, start training or
promote an adapter. Correctness and provenance of grading must be established
externally; this module verifies pairing, identity and aggregate arithmetic.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import re
from statistics import median
from pathlib import Path
from scripts.neral_v43_rft_cpu import read_pinned, rows, protected_hashes, canonical
from scripts.neral_v43_rft_pilot import holdout_hashes, prompt_hash

CATEGORIES=('everyday','knowledge','math','code')
STATUSES=('graded','timeout','unsupported','invalid','error')
COMMON=('holdout_sha256','checkpoint_sha256','conditions_sha256','grader_sha256')

def sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('required SHA256')
    return value

def index_receipt(receipt,cases,holdout_sha,adapter):
    for key in COMMON:sha(receipt[key])
    if receipt['holdout_sha256']!=holdout_sha:raise ValueError('holdout identity mismatch')
    if adapter:
        sha(receipt['adapter_sha256'])
    elif receipt['adapter_sha256'] is not None:
        raise ValueError('baseline must have no adapter')
    indexed={}
    for row in receipt['rows']:
        key=row['id']
        if key not in cases or key in indexed:raise ValueError('unknown or duplicate result')
        if row['prompt_sha256']!=prompt_hash(cases[key]['prompt']):
            raise ValueError('prompt identity mismatch')
        if (row['status'] not in STATUSES or type(row['correct']) is not bool
                or (row['status']!='graded' and row['correct'])):
            raise ValueError('inconsistent grade')
        latency=row['latency_ms']
        if type(latency) not in (int,float) or not math.isfinite(latency) or latency<0:
            raise ValueError('invalid latency')
        indexed[key]=row
    if set(indexed)!=set(cases):raise ValueError('incomplete paired evaluation')
    return indexed

def stats(keys,baseline,adapter):
    bc=sum(baseline[k]['correct'] for k in keys)
    ac=sum(adapter[k]['correct'] for k in keys)
    gains=sum(not baseline[k]['correct'] and adapter[k]['correct'] for k in keys)
    regressions=sum(baseline[k]['correct'] and not adapter[k]['correct'] for k in keys)
    return dict(total=len(keys),baseline_correct=bc,adapter_correct=ac,gains=gains,
                regressions=regressions,both_correct=bc-regressions,
                both_incorrect=len(keys)-bc-gains,
                delta_percentage_points=100*(ac-bc)/len(keys))

def compare(holdout,baseline,adapter,*,holdout_sha,protected=frozenset()):
    sha(holdout_sha)
    holdout_hashes(holdout,protected)
    cases={}
    for case in holdout:
        key=case['id']
        if (not isinstance(key,str) or not key.strip() or key in cases
                or case.get('reviewed') is not True
                or case.get('source')!='independent_holdout'
                or not isinstance(case.get('family'),str) or not case['family'].strip()):
            raise ValueError('unreviewed or duplicate holdout identity')
        cases[key]=case
    b=index_receipt(baseline,cases,holdout_sha,False)
    a=index_receipt(adapter,cases,holdout_sha,True)
    if any(baseline[k]!=adapter[k] for k in COMMON):
        raise ValueError('evaluation conditions differ')
    result=stats(list(cases),b,a)
    n=result['gains']+result['regressions']
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(result['gains'],result['regressions'])+1))/2**n)
    result.update(exact_mcnemar_p=p,
        by_category={cat:stats([k for k,c in cases.items() if c['category']==cat],b,a) for cat in CATEGORIES},
        baseline_status_counts=dict(Counter(r['status'] for r in b.values())),
        adapter_status_counts=dict(Counter(r['status'] for r in a.values())),
        baseline_median_latency_ms=median(r['latency_ms'] for r in b.values()),
        adapter_median_latency_ms=median(r['latency_ms'] for r in a.values()),
        holdout_sha256=holdout_sha,checkpoint_sha256=baseline['checkpoint_sha256'],
        conditions_sha256=baseline['conditions_sha256'],grader_sha256=baseline['grader_sha256'],
        adapter_sha256=adapter['adapter_sha256'],
        scope='development_holdout_only',grading_independently_verified_by_this_tool=False,
        promotion_authorized=False)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('holdout','baseline','adapter','protected'):
        p.add_argument('--'+key,type=Path,required=True)
        p.add_argument('--'+key+'-sha',required=True)
    args=p.parse_args()
    protected=protected_hashes(json.loads(read_pinned(args.protected,args.protected_sha)))
    holdout=rows(read_pinned(args.holdout,args.holdout_sha))
    baseline=json.loads(read_pinned(args.baseline,args.baseline_sha))
    adapter=json.loads(read_pinned(args.adapter,args.adapter_sha))
    report=compare(holdout,baseline,adapter,holdout_sha=args.holdout_sha,protected=protected)
    if protected_hashes(json.loads(read_pinned(args.protected,args.protected_sha)))!=protected:
        raise ValueError('protected data changed')
    print(canonical(report))

if __name__=='__main__':
    try:main()
    except Exception:raise SystemExit('V44_PAIRED_EVAL_FAILED (private details withheld)')
