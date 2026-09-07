"""Build the minimal frozen-harness report consumed by promotion_gate50.py."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def bundle(a,p):return metrics(load(getattr(a,f'{p}_free')),load(getattr(a,f'{p}_v3')),load(getattr(a,f'{p}_v4')),load(getattr(a,f'{p}_rawmath')))

def main():
    ap=argparse.ArgumentParser()
    for p in ('baseline20','candidate'):
        for k in ('free','v3','v4','rawmath'):ap.add_argument(f'--{p}-{k}',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args();b=bundle(a,'baseline20');c=bundle(a,'candidate')
    fb=load(a.baseline20_free);fc=load(a.candidate_free)
    if not fb.get('suite_sha256') or fb['suite_sha256']!=fc.get('suite_sha256'):raise ValueError('20M/50M frozen free-suite mismatch')
    report={'definition':'Direct identical frozen-harness 20M Preview02 vs final 50M candidate. Tools excluded.','stages':{'baseline20':b,'instruct':c},'nis':{'baseline20':nis_v1(b),'instruct':nis_v1(c)},'free_suite_sha256':fb['suite_sha256']}
    Path(a.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps({'nis20':report['nis']['baseline20']['score'],'nis50':report['nis']['instruct']['score']},ensure_ascii=False))
if __name__=='__main__':main()
