"""Compare a B3 phase checkpoint against its immediately preceding frozen-harness baseline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))

def bundle(a,prefix):return metrics(load(getattr(a,f'{prefix}_free')),load(getattr(a,f'{prefix}_v3')),load(getattr(a,f'{prefix}_v4')),load(getattr(a,f'{prefix}_rawmath')))

def main():
    ap=argparse.ArgumentParser()
    for prefix in ('before','after'):
        for kind in ('free','v3','v4','rawmath'):ap.add_argument(f'--{prefix}-{kind}',required=True)
    ap.add_argument('--phase',choices=('a','b'),required=True);ap.add_argument('--training-summary',required=True);ap.add_argument('--out-json',required=True);ap.add_argument('--out-md',required=True);a=ap.parse_args()
    before=bundle(a,'before');after=bundle(a,'after');fb=load(a.before_free);fa=load(a.after_free)
    if not fb.get('suite_sha256') or fb.get('suite_sha256')!=fa.get('suite_sha256'):raise ValueError('frozen free-generation suite mismatch')
    nb,na=nis_v1(before),nis_v1(after);common={k for k in before if before[k] is not None and after.get(k) is not None};delta={k:after[k]-before[k] for k in common}
    report={'phase':a.phase,'definition':'Same frozen raw-model harness before/after B3; tools excluded.','before':before,'after':after,'delta':delta,'nis_before':nb,'nis_after':na,'training':load(a.training_summary),'free_suite_sha256':fb['suite_sha256']}
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    keys=('eos_pct','repeated_trigram_pct','rawmath_v1_exact_pct','v3_pair_overall_pct','v4_pair_overall_pct','v4_math_pct','v4_code_pct','context_recall_overall_pct','true_byte_bpb_macro_1024','decode_tokens_per_second')
    lines=[f'# Neral-50M B3{a.phase} benchmark','',f'NIS 1.0: **{nb["score"]:.3f} → {na["score"]:.3f} ({na["score"]-nb["score"]:+.3f})**.','', '| Metric | Before | After | Delta |','|---|---:|---:|---:|']
    for k in keys:
        if k not in common:continue
        lines.append(f'| {k} | {before[k]:.4f} | {after[k]:.4f} | {delta[k]:+.4f} |')
    lines+=['','This report is evidence only. The next training phase must not start automatically.']
    Path(a.out_md).write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps({'phase':a.phase,'nis_before':nb['score'],'nis_after':na['score']},ensure_ascii=False))

if __name__=='__main__':main()
