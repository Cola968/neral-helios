"""Summarize B1 -> B2 with the frozen Neral benchmark harness.

Direct metrics remain canonical. NIS 1.0 is included only as a secondary versioned
Neral-only composite when every required component is present.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from summarize_strong50 import metrics
from neral_intelligence_score import nis_v1


def load(path):
    if not path:return None
    p=Path(path);return json.loads(p.read_text(encoding='utf8')) if p.is_file() else None


def stage(args,prefix):
    return metrics(load(getattr(args,f'{prefix}_free')),load(getattr(args,f'{prefix}_v3')),load(getattr(args,f'{prefix}_v4')),load(getattr(args,f'{prefix}_rawmath')))


def score_or_incomplete(m):
    try:return nis_v1(m)
    except ValueError as e:return {'name':'Neral Intelligence Score','version':'1.0','status':'incomplete','score':None,'reason':str(e)}


def main():
    ap=argparse.ArgumentParser()
    for prefix in ('baseline20','b1','b2'):
        for kind in ('free','v3','v4','rawmath'):ap.add_argument(f'--{prefix}-{kind}')
    ap.add_argument('--training-summary',required=True);ap.add_argument('--out-json',required=True);ap.add_argument('--out-md',required=True);a=ap.parse_args()
    sets={p:stage(a,p) for p in ('baseline20','b1','b2')}
    nis={p:score_or_incomplete(sets[p]) for p in ('baseline20','b1','b2')}
    free_reports={p:load(getattr(a,f'{p}_free')) for p in ('baseline20','b1','b2')}
    suite_hashes={p:r.get('suite_sha256') for p,r in free_reports.items() if r}
    nonnull={v for v in suite_hashes.values() if v}
    if len(nonnull)>1:raise ValueError(f'Frozen free-generation suite mismatch: {suite_hashes}')
    delta={k:sets['b2'][k]-sets['b1'][k] for k in set(sets['b1'])&set(sets['b2']) if sets['b1'][k] is not None and sets['b2'][k] is not None}
    report={'definition':'Direct frozen-suite metrics are canonical. NIS 1.0 is a secondary versioned Neral-only composite.','suite_hashes':suite_hashes,'stages':sets,'nis':nis,'delta_b2_minus_b1':delta,'training_summary':load(a.training_summary)}
    preferred=['utf8_valid_pct','eos_pct','repeated_trigram_pct','non_repetition_trigram_pct','raw_math_exact_pct','rawmath_v1_exact_pct','rawmath_v1_eos_pct','v3_pair_overall_pct','v4_pair_overall_pct','context_recall_overall_pct','decode_tokens_per_second']
    for k in ['true_byte_bpb_macro_1024','true_byte_bpb_de_1024','true_byte_bpb_en_1024']:
        if any(k in s for s in sets.values()):preferred.insert(5,k)
    lines=['# Neral-50M B2 benchmark','', 'Same frozen harness before and after broader continued pretraining. SOLVE and Helios tools are excluded.','', '| Metric | 20M Preview 02 | 50M B1 | 50M B2 | Δ B2-B1 |','|---|---:|---:|---:|---:|']
    for k in preferred:
        vals=[sets[p].get(k) for p in ('baseline20','b1','b2')]
        if all(v is None for v in vals):continue
        d=delta.get(k);plain=k=='decode_tokens_per_second' or k.startswith('true_byte_bpb_');suffix='' if plain else '%'
        f=lambda v:'—' if v is None else f'{v:.3f}{suffix}'
        lines.append(f'| {k.replace("_"," ")} | {f(vals[0])} | {f(vals[1])} | {f(vals[2])} | {f(d)} |')
    lines.extend(['','## Neral Intelligence Score 1.0','', 'Secondary summary only; promotion still uses the direct regression gates.','', '| Model | NIS 1.0 | Status |','|---|---:|---|'])
    labels={'baseline20':'20M Preview 02','b1':'50M B1','b2':'50M B2'}
    for p in ('baseline20','b1','b2'):
        r=nis[p];score='—' if r.get('score') is None else f"{r['score']:.2f}"
        lines.append(f"| {labels[p]} | {score} | {r['status']} |")
    lines.extend(['','B2 is not promoted from NIS or this report alone. Instruction tuning and the RC promotion gate remain separate stages.'])
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    Path(a.out_md).write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'saved':a.out_md,'metric_count_b2':len(sets['b2']),'suite_hashes':suite_hashes,'nis':{p:r.get('score') for p,r in nis.items()}},ensure_ascii=False))


if __name__=='__main__':main()
