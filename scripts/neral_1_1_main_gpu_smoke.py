"""Small, known dev24 integration smoke on RTX 4070; never a new blind benchmark."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.neral_1_1_postverify_dev24_cases import build, manifest
from scripts.neral_1_1_prospective_blind100 import require_gpu, run_case
from neral_nl_local_ollama import _localhost_transport

# One known mathematics, one known Z3 graph, one known original Python example.
INDICES=(0,6,12)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    actual=subprocess.run(['git','rev-parse','HEAD'],text=True,
      capture_output=True,check=True,timeout=12).stdout.strip()
    if actual!=os.environ.get('NERAL_INTEGRATION_SOURCE_SHA'):
        raise SystemExit('INTEGRATION_SHA_MISMATCH')
    if args.out.exists():
        raise SystemExit('OUTPUT_ALREADY_EXISTS')
    hardware=require_gpu()
    cases=build()
    assert len(cases)==24 and manifest()['cases']==24
    selected=[cases[i] for i in INDICES]
    assert [c['category'] for c in selected]==[
        'mathematics','logic','programming']
    summaries=[]
    for case in selected:
        # No prompts, code, model envelopes or raw traces persisted.
        r=run_case(case,_localhost_transport)
        if r['id']!=case['id'] or not isinstance(r['base_seconds'],(int,float)):
            raise RuntimeError('INCOMPLETE_GPU_SMOKE_PAIR')
        summaries.append({
          'case_id':r['id'],'category':r['category'],
          'base_correct':bool(r['base_strict_correct']),
          'neral_correct':bool(r['neral_strict_correct']),
          'verified':bool(r['independent_verified']),
          'proof_backend':r['proof_backend'],
          'baseline_s':r['base_seconds'],'neral_s':r['neral_seconds'],
          'proposal_calls':r['proposal_calls'],
        })
        print('GPU_SMOKE_PROGRESS '+str(len(summaries))+'/3',flush=True)
    regressions=sum(x['base_correct'] and not x['neral_correct'] for x in summaries)
    n_verified=sum(x['verified'] for x in summaries)
    success=(regressions==0 and n_verified>=1 and
      all(x['neral_correct'] for x in summaries))
    summary={
      'source_sha':actual,'development_not_blind':True,
      'no_training':True,'no_main_write':True,'count':len(summaries),
      'regressions':regressions,'independent_verified':n_verified,
      'status':'PASS' if success else 'BLOCKED',
      'cases':summaries,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print('INTEGRATION_GPU_SMOKE '+summary['status']+' '+
          json.dumps({'source_sha':actual,'count':len(summaries),
          'regressions':regressions,'independent_verified':n_verified},sort_keys=True),flush=True)
    if not success: raise SystemExit(2)

if __name__=='__main__': main()
