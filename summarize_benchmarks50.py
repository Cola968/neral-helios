"""Create a concise, auditable percentage summary from Neral benchmark reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    if not path:
        return None
    p=Path(path)
    return json.loads(p.read_text(encoding='utf8')) if p.is_file() else None


def pct(x):
    return None if x is None else 100.0*float(x)


def free_metrics(r):
    if not r:return {}
    s=r.get('summary',{})
    total=s.get('numeric_total',0) or 0
    return {
        'utf8_valid_pct':pct(s.get('valid_utf8_fraction')),
        'eos_pct':pct(s.get('eos_fraction')),
        'non_repetition_trigram_pct':None if s.get('mean_repeated_trigram_fraction') is None else 100.0*(1.0-s['mean_repeated_trigram_fraction']),
        'repeated_trigram_pct':pct(s.get('mean_repeated_trigram_fraction')),
        'raw_math_exact_pct':100.0*s.get('numeric_exact',0)/total if total else None,
        'decode_tokens_per_second':s.get('mean_tokens_per_second'),
    }


def v3_metrics(r):
    if not r:return {}
    p=r.get('pair_preference',{});g=r.get('free_generation',{})
    out={'v3_pair_overall_pct':pct(p.get('accuracy'))}
    for d,v in p.get('domains',{}).items():out[f'v3_{d}_pct']=pct(v.get('accuracy'))
    for group,key in [('chat_instruction','v3_chat'),('base_continuation','v3_base')]:
        s=g.get(group,{}).get('summary',{})
        if s:
            out[f'{key}_eos_pct']=pct(s.get('eos_fraction'))
            out[f'{key}_unique_pct']=pct(s.get('unique_output_fraction'))
            out[f'{key}_repeated_trigram_pct']=pct(s.get('mean_repeated_trigram_fraction'))
    return out


def v4_metrics(r):
    if not r:return {}
    p=r.get('pair_suite',{});c=r.get('context_recall',{});g=r.get('generation_grid',{})
    out={'v4_pair_overall_pct':pct(p.get('accuracy')),'context_recall_overall_pct':pct(c.get('accuracy'))}
    for d,v in p.get('domains',{}).items():out[f'v4_{d}_pct']=pct(v.get('accuracy'))
    for d,v in c.get('by_distance',{}).items():
        total=v.get('total',0) or 0
        out[f'context_recall_{d}B_pct']=100.0*v.get('correct',0)/total if total else None
    for mode,s in g.get('summary',{}).items():
        out[f'v4_{mode}_eos_pct']=pct(s.get('eos_fraction'))
        out[f'v4_{mode}_unique_pct']=pct(s.get('unique_fraction'))
        out[f'v4_{mode}_repeated_trigram_pct']=pct(s.get('mean_repeated_trigram_fraction'))
    e=r.get('next_token_entropy',{})
    out['next_token_entropy_bits']=e.get('mean_entropy_bits')
    out['next_token_mean_top1_pct']=pct(e.get('mean_top1_probability'))
    return out


def flatten(free,v3,v4):
    return free_metrics(free)|v3_metrics(v3)|v4_metrics(v4)


def delta(before,after):
    keys=sorted(set(before)&set(after));out={}
    for k in keys:
        if before[k] is not None and after[k] is not None:out[k]=after[k]-before[k]
    return out


def fmt(v):
    if v is None:return '—'
    return f'{v:.2f}'


def main():
    ap=argparse.ArgumentParser()
    for prefix in ['before','after','baseline20']:
        ap.add_argument(f'--{prefix}-free')
        ap.add_argument(f'--{prefix}-v3')
        ap.add_argument(f'--{prefix}-v4')
    ap.add_argument('--training-summary')
    ap.add_argument('--out-json',required=True)
    ap.add_argument('--out-md',required=True)
    a=ap.parse_args()

    sets={}
    for prefix in ['baseline20','before','after']:
        sets[prefix]=flatten(load(getattr(a,f'{prefix}_free')),load(getattr(a,f'{prefix}_v3')),load(getattr(a,f'{prefix}_v4')))
    report={
        'definition':'Percentages are direct scores from frozen Neral tests. No synthetic composite intelligence percentage is invented.',
        'baseline20_preview02':sets['baseline20'],
        'neral50m_before_full_pretrain':sets['before'],
        'neral50m_after_full_pretrain':sets['after'],
        'delta_after_minus_before':delta(sets['before'],sets['after']),
        'training_summary':load(a.training_summary),
    }
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')

    preferred=[
        'utf8_valid_pct','eos_pct','repeated_trigram_pct','non_repetition_trigram_pct','raw_math_exact_pct',
        'v3_pair_overall_pct','v3_de_grammar_pct','v3_en_grammar_pct','v3_facts_pct','v3_commonsense_pct','v3_code_pct','v3_math_pct',
        'v4_pair_overall_pct','v4_de_grammar_pct','v4_en_grammar_pct','v4_physics_pct','v4_astronomy_pct','v4_aviation_pct','v4_transport_pct','v4_math_pct','v4_logic_pct','v4_commonsense_pct','v4_code_pct','v4_translation_pct','v4_induction_pct',
        'context_recall_overall_pct','context_recall_128B_pct','context_recall_512B_pct','context_recall_1024B_pct','context_recall_2048B_pct','context_recall_3072B_pct',
        'v3_base_eos_pct','v3_base_unique_pct','v3_base_repeated_trigram_pct','v4_greedy_eos_pct','v4_greedy_unique_pct','v4_greedy_repeated_trigram_pct',
        'next_token_entropy_bits','next_token_mean_top1_pct','decode_tokens_per_second'
    ]
    labels={k:k.replace('_pct','').replace('_',' ') for k in preferred}
    lines=['# Neral benchmark comparison','',
           'No composite intelligence score is used. Percentages are shown only where the underlying benchmark naturally produces a percentage.','',
           '| Metric | 20M Preview 02 | 50M before | 50M after | Δ after-before |','|---|---:|---:|---:|---:|']
    for k in preferred:
        vals=[sets['baseline20'].get(k),sets['before'].get(k),sets['after'].get(k)]
        if all(v is None for v in vals):continue
        d=report['delta_after_minus_before'].get(k)
        suffix='' if k in {'next_token_entropy_bits','decode_tokens_per_second'} else '%'
        lines.append(f"| {labels[k]} | {fmt(vals[0])}{suffix if vals[0] is not None else ''} | {fmt(vals[1])}{suffix if vals[1] is not None else ''} | {fmt(vals[2])}{suffix if vals[2] is not None else ''} | {fmt(d)}{suffix if d is not None else ''} |")
    Path(a.out_md).write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'saved_json':a.out_json,'saved_markdown':a.out_md,'metrics_after':len(sets['after'])},ensure_ascii=False))


if __name__=='__main__':main()
