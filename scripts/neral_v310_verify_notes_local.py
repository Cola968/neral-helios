"""CPU-only verified note check against actual installed modules; no app access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--installed-root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args(argv)
    sys.path.insert(0,str(args.installed_root))
    from neral_verified_notes import exact_note_lookup
    rows=[json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)<97: raise ValueError('Expected at least 97 local training examples')
    math_example=next(x for x in rows if x['meta'].get('source')=='v39_benchmark' and x['meta'].get('category')=='mathematics')
    note=exact_note_lookup(math_example['instruction'],args.data)
    if not note or note['text']!=math_example['response'] or note['source']!='v39_benchmark' or note['verifier_ok']:
        raise ValueError('Benchmark memory improperly promoted or lookup failed')
    web_example=next(x for x in rows if x['meta'].get('source')=='v39_benchmark' and x['meta'].get('category')=='web')
    if exact_note_lookup(web_example['instruction'],args.data) is not None:
        raise ValueError('Stale web notes must be rejected')
    if exact_note_lookup(math_example['instruction']+' extra',args.data) is not None:
        raise ValueError('Approximate note retrieval prohibited')
    print(json.dumps({'status':'installed_notes_exact_match_cpu_verified',
                      'records':len(rows),'no_model_retrained':True,
                      'benchmark_examples_not_runtime_verifier_certified':True}))
    return 0

if __name__=='__main__':raise SystemExit(main())
