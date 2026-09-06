"""Summarize B1 -> B2 with the frozen Neral benchmark harness.

No composite intelligence score is created. RawMath is model-only; SOLVE/Helios results are
not accepted as inputs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from summarize_strong50 import metrics


def load(path):
    if not path:return None
    p=Path(path);return json.loads(p.read_text(encoding='utf8')) if p.is_file() else None


def stage(args,prefix):
    return metrics(load(getattr(args,f'{prefix}_free')),load(getattr(args,f'{prefix}_v3')),load(getattr(args,f'{prefix}_v4')),load(getattr(args,f'{prefix}_rawmath')))


def main():
    ap=argparse.ArgumentParser()
    for prefix in ('baseline20','b1','b2'):
        for kind in ('free','v3','v4','rawmath'):ap.add_argument(f'--{prefix}-{kind}')
    ap.add_argument('--training-summary',required=True);ap.add_argument('--out-json',required=True);ap.add_argument('--out-md',required=True);a=ap.parse_args()
    sets={p:stage(a,p) for p in ('baseline20','b1','b2')}
    free_reports={p:load(getattr(a,f'{p}_free')) for p in ('baseline20','b1','b2')}
    suite_hashes={p:r.get('suite_sha256') for p,r in free_reports.items() if r}
    nonnull={v for v in suite_hashes.values() if v}
    if len(nonnull)>1:raise ValueError(f'Frozen free-generation suite mismatch: {suite_hashes}')
    delta={k:sets['b2'][k]-sets['b1'][k] for k in set(sets['b1'])&set(sets['b2']) if sets['b1'][k] is not None and sets['b2'][k] is not None}
    report={'definition':'Direct frozen-suite metrics only; no synthetic intelligence composite.','suite_hashes':suite_hashes,'stages':sets,'delta_b2_minus_b1':delta,'training_summary':load(a.training_summary)}
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    preferred=['utf8_valid_pct','eos_pct','repeated_trigram_pct','non_repetition_trigram_pct','raw_math_exact_pct','rawmath_v1_exact_pct','rawmath_v1_eos_pct','v3_pair_overall_pct','v4_pair_overall_pct','context_recall_overall_pct','decode_tokens_per_second']
    # If the shared summarizer exposes true-byte BPB, include it automatically.
    for k in ['true_byte_bpb_macro_1024','true_byte_bpb_de_1024','true_byte_bpb_en_1024']:
        if any(k in s for s in sets.values()):preferred.insert(5,k)
    lines=['# Neral-50M B2 benchmark','', 'Same frozen harness before and after broader continued pretraining. SOLVE and Helios tools are excluded.','', '| Metric | 20M Preview 02 | 50M B1 | 50M B2 | Δ B2-B1 |','|---|---:|---:|---:|---:|']
    for k in preferred:
        vals=[sets[p].get(k) for p in ('baseline20','b1','b2')]
        if all(v is None for v in vals):continue
        d=delta.get(k);plain=k=='decode_tokens_per_second' or k.startswith('true_byte_bpb_');suffix='' if plain else '%'
        f=lambda v:'—' if v is None else f'{v:.3f}{suffix}'
        lines.append(f'| {k.replace("_"," ")} | {f(vals[0])} | {f(vals[1])} | {f(vals[2])} | {f(d)} |')
    lines.extend(['','B2 is not promoted from this report alone. Instruction tuning and the RC promotion gate remain separate stages.'])
    Path(a.out_md).write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'saved':a.out_md,'metric_count_b2':len(sets['b2']),'suite_hashes':suite_hashes},ensure_ascii=False))


if __name__=='__main__':main()
