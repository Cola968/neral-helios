from __future__ import annotations

"""Frozen release gate for Neral 350M Frontier.

Input is a JSON scorecard produced by the existing frozen benchmark stack. Training
must never inspect or select on benchmark examples; this gate is post-training only.
"""

import argparse, json
from pathlib import Path

MINIMUMS = {
    'nis2': 82.0,
    'rawmath_exact': 42.0,
    'targeted_math': 80.0,
    'code': 78.0,
    'de': 96.0,
    'reasoning': 92.0,
    'context': 100.0,
    'free_eos': 85.0,
    'utf8': 99.9,
}
MAXIMUMS = {
    'repetition': 2.5,
    'bpb4095': 1.32,
}
STRETCH = {
    'nis2': 86.0,
    'rawmath_exact': 50.0,
    'targeted_math': 88.0,
    'code': 84.0,
    'de': 98.0,
    'reasoning': 95.0,
    'free_eos': 92.0,
    'repetition': 1.5,
    'bpb4095': 1.25,
}


def assess(m: dict) -> dict:
    missing=[k for k in [*MINIMUMS,*MAXIMUMS] if k not in m]
    if missing:
        return {'pass':False,'failures':[f'missing:{x}' for x in missing],'stretch_hits':{}}
    failures=[]
    for k,v in MINIMUMS.items():
        if float(m[k]) < v: failures.append(f'{k}<{v}')
    for k,v in MAXIMUMS.items():
        if float(m[k]) > v: failures.append(f'{k}>{v}')
    stretch_hits={}
    for k,v in STRETCH.items():
        if k in ('repetition','bpb4095'): stretch_hits[k]=float(m.get(k,1e9)) <= v
        else: stretch_hits[k]=float(m.get(k,-1e9)) >= v
    return {
        'pass': not failures,
        'release_candidate': not failures,
        'failures': failures,
        'minimums': MINIMUMS,
        'maximums': MAXIMUMS,
        'stretch': STRETCH,
        'stretch_hits': stretch_hits,
        'metrics': m,
        'policy': 'No Play Store model promotion unless every hard release metric passes. Stretch metrics are aspirational and never fabricated.'
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scorecard',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    m=json.loads(Path(a.scorecard).read_text(encoding='utf8'))
    result=assess(m)
    Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['pass'] else 2)

if __name__=='__main__': main()
