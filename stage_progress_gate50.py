"""Fail-closed progress review between Neral-50M weight-changing stages.

This is intentionally stricter than "loss went down" but weaker than the final 20M promotion
gate. It detects the exact trade-off the training policy forbids: e.g. lower repetition bought
with materially worse byte-level language modelling. It never starts another stage itself.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HIGHER=('eos_pct','rawmath_v1_exact_pct','v3_pair_overall_pct','v4_pair_overall_pct','context_recall_overall_pct')
LOWER=('repeated_trigram_pct','true_byte_bpb_macro_1024')


def decide(report):
    before=report.get('before',{});after=report.get('after',{});missing=[k for k in HIGHER+LOWER if before.get(k) is None or after.get(k) is None]
    if missing:return {'proceed':False,'reason':'missing_metrics','missing':missing,'wins':0,'severe':[]}
    wins=sum(after[k]>before[k] for k in HIGHER)+sum(after[k]<before[k] for k in LOWER);severe=[]
    if after['repeated_trigram_pct']>before['repeated_trigram_pct']+5:severe.append('repetition_regressed_gt_5pp')
    if after['true_byte_bpb_macro_1024']>before['true_byte_bpb_macro_1024']*1.03:severe.append('bpb_regressed_gt_3pct')
    for k in ('v3_pair_overall_pct','v4_pair_overall_pct'):
        if after[k]<before[k]-5:severe.append(f'{k}_regressed_gt_5pp')
    if after['context_recall_overall_pct']<before['context_recall_overall_pct']-10:severe.append('context_regressed_gt_10pp')
    # Explicitly reject repetition cosmetics that harm the language model.
    if after['repeated_trigram_pct']<before['repeated_trigram_pct'] and after['true_byte_bpb_macro_1024']>before['true_byte_bpb_macro_1024']*1.01:
        severe.append('repetition_improved_but_bpb_worsened_gt_1pct')
    return {'proceed':not severe and wins>=3,'reason':'pass' if (not severe and wins>=3) else 'insufficient_or_regressive_progress','wins':wins,'required_wins':3,'severe':severe,
            'note':'Decision support only. The next weight-changing stage still requires a separate manual workflow start.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('comparison');ap.add_argument('--out',required=True);a=ap.parse_args();report=json.loads(Path(a.comparison).read_text(encoding='utf8'));d=decide(report);Path(a.out).write_text(json.dumps(d,indent=2),encoding='utf8');print(json.dumps(d));
    if not d['proceed']:raise SystemExit(2)

if __name__=='__main__':main()
