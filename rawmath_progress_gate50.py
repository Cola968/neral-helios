"""Fail-closed review gate for RawMath Stage C before instruction tuning."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def decide(report):
    b=report.get('before',{});a=report.get('after',{})
    need=('rawmath_v1_exact_pct','rawmath_distribution_id_pct','rawmath_distribution_ood_pct','rawmath_distribution_transfer_pct','true_byte_bpb_macro_1024','repeated_trigram_pct','v3_pair_overall_pct','v4_pair_overall_pct')
    missing=[k for k in need if b.get(k) is None or a.get(k) is None]
    if missing:return {'proceed':False,'reason':'missing_metrics','missing':missing,'severe':[]}
    severe=[]
    if a['true_byte_bpb_macro_1024']>b['true_byte_bpb_macro_1024']*1.03:severe.append('bpb_regressed_gt_3pct')
    if a['repeated_trigram_pct']>b['repeated_trigram_pct']+5:severe.append('repetition_regressed_gt_5pp')
    if a['v3_pair_overall_pct']<b['v3_pair_overall_pct']-5:severe.append('v3_regressed_gt_5pp')
    if a['v4_pair_overall_pct']<b['v4_pair_overall_pct']-5:severe.append('v4_regressed_gt_5pp')
    math_wins=sum(a[k]>b[k] for k in ('rawmath_v1_exact_pct','rawmath_distribution_id_pct','rawmath_distribution_ood_pct','rawmath_distribution_transfer_pct'))
    proceed=not severe and a['rawmath_v1_exact_pct']>b['rawmath_v1_exact_pct'] and math_wins>=2
    return {'proceed':proceed,'reason':'pass' if proceed else 'math_not_broadly_improved_or_regression','math_slice_wins':math_wins,'required_math_slice_wins':2,'severe':severe,'note':'Instruction tuning remains a separate manual workflow start.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('comparison');p.add_argument('--out',required=True);a=p.parse_args();r=decide(json.loads(Path(a.comparison).read_text(encoding='utf8')));Path(a.out).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r));raise SystemExit(0 if r['proceed'] else 2)
if __name__=='__main__':main()
