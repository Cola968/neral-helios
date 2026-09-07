"""Generic before/after frozen-harness summary for one Neral-50M weight-changing stage."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def bundle(a,p):return metrics(load(getattr(a,f'{p}_free')),load(getattr(a,f'{p}_v3')),load(getattr(a,f'{p}_v4')),load(getattr(a,f'{p}_rawmath')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True)
    for p in ('before','after'):
        for k in ('free','v3','v4','rawmath'):ap.add_argument(f'--{p}-{k}',required=True)
    ap.add_argument('--training-summary',required=True);ap.add_argument('--out-json',required=True);ap.add_argument('--out-md',required=True);a=ap.parse_args()
    b=bundle(a,'before');c=bundle(a,'after');fb=load(a.before_free);fa=load(a.after_free)
    if fb.get('suite_sha256')!=fa.get('suite_sha256') or not fb.get('suite_sha256'):raise ValueError('free-suite mismatch')
    nb,na=nis_v1(b),nis_v1(c);common={k for k in b if b[k] is not None and c.get(k) is not None};delta={k:c[k]-b[k] for k in common}
    rm_before=load(a.before_rawmath).get('by_distribution',{});rm_after=load(a.after_rawmath).get('by_distribution',{})
    report={'stage':a.stage,'definition':'Same frozen raw-model harness before/after; SOLVE/VERIFY/Helios excluded.','before':b,'after':c,'delta':delta,'nis_before':nb,'nis_after':na,'rawmath_distribution_before':rm_before,'rawmath_distribution_after':rm_after,'training':load(a.training_summary),'free_suite_sha256':fb['suite_sha256']}
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    keys=('eos_pct','repeated_trigram_pct','rawmath_v1_exact_pct','rawmath_distribution_id_pct','rawmath_distribution_ood_pct','rawmath_distribution_transfer_pct','v3_pair_overall_pct','v4_pair_overall_pct','v4_math_pct','v4_code_pct','context_recall_overall_pct','true_byte_bpb_macro_1024','decode_tokens_per_second')
    lines=[f'# {a.stage} frozen benchmark','',f'NIS 1.0: **{nb["score"]:.3f} → {na["score"]:.3f} ({na["score"]-nb["score"]:+.3f})**.','','| Metric | Before | After | Delta |','|---|---:|---:|---:|']
    for k in keys:
        if k in common:lines.append(f'| {k} | {b[k]:.4f} | {c[k]:.4f} | {delta[k]:+.4f} |')
    lines+=['','The next weight-changing stage requires a separate manual workflow start.']
    Path(a.out_md).write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps({'stage':a.stage,'nis_before':nb['score'],'nis_after':na['score']},ensure_ascii=False))
if __name__=='__main__':main()
