"""Explicit I1 acceptance targets; consumes frozen model-only reports unchanged."""
import argparse
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate(comparison, before_v5, after_v5, promotion):
    before=comparison.get('stages',{}).get('baseline20',{})
    after=comparison.get('stages',{}).get('instruct',{})
    checks={}
    def check(name,b,a,relation):
        checks[name]={'baseline_or_target':b,'candidate':a,
                      'pass':bool(finite(b) and finite(a) and relation(a,b))}
    nis=comparison.get('nis',{})
    score=nis.get('instruct',{}).get('score')
    check('nis_at_least_55',55,score,lambda a,b:a>=b)
    check('nis_beats_preview02',nis.get('baseline20',{}).get('score'),score,lambda a,b:a>b)
    check('rawmath_at_least_60_pct',60,after.get('rawmath_v1_exact_pct'),lambda a,b:a>=b)
    check('free_eos_at_least_60_pct',60,after.get('eos_pct'),lambda a,b:a>=b)
    check('free_repetition_below_5_pct',5,after.get('repeated_trigram_pct'),lambda a,b:a<b)
    for key in ('v3_code_pct','v4_code_pct','context_recall_overall_pct'):
        check(key+'_held',before.get(key),after.get(key),lambda a,b:a>=b)
    for key in ('single_needle','multi_needle'):
        check(key+'_held',before_v5.get(key,{}).get('accuracy'),after_v5.get(key,{}).get('accuracy'),lambda a,b:a>=b)
    for seq in ('2048','3072','4095'):
        key='document_bounded_true_byte_bpb'
        b=before_v5.get(key,{}).get('macro_bits_per_byte',{}).get(seq)
        a=after_v5.get(key,{}).get('macro_bits_per_byte',{}).get(seq)
        check('bpb_'+seq+'_held',b,a,lambda a,b:0<a<=b)
    checks['v5_same_frozen_suite']={'pass':bool(before_v5.get('suite_sha256') and before_v5.get('suite_sha256')==after_v5.get('suite_sha256'))}
    checks['existing_promotion_gate']={'pass':promotion.get('ready_for_rc_review') is True}
    return {'campaign':'Neral-50M I1','targets_met':all(v['pass'] for v in checks.values()),
            'checks':checks,'policy':'Final frozen comparison only. No SOLVE/Helios credit. No weights deleted by this gate.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reports',required=True);a=ap.parse_args()
    root=Path(a.reports)
    load=lambda p:json.loads((root/p).read_text(encoding='utf8'))
    result=evaluate(load('I1_20M_VS_50M.json'),load('baseline20/v5.json'),load('candidate50/v5.json'),load('I1_PROMOTION_GATE.json'))
    (root/'I1_TARGET_GATE.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
