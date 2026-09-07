"""Summarize only complete, hash-verified local raw-model benchmark bundles."""
import json
from pathlib import Path
from run_frozen_benchmarks50 import sha,BASELINE_SHA
from summarize_strong50 import metrics
from neral_intelligence_score import nis_v1
from campaign50_decision import decide


def main():
    root=Path('reports/roadmap_20260907');stages={};raw={};identities={}
    for stage in ('baseline20','b1','termination_probe','b2','rawmath','instruct'):
        folder=root/stage
        if not all((folder/(k+'.audit.json')).exists() for k in ('free','v3','v4','rawmath','v5')):continue
        reports={};audits=[]
        for kind in ('free','v3','v4','rawmath','v5'):
            p=folder/(kind+'.json');audit=json.loads((folder/(kind+'.audit.json')).read_text())
            if audit['report_sha256']!=sha(p) or audit['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Report identity changed')
            if kind=='v5' and audit['v5_protocol_sha256']!=sha(root/'protocol_v5.json'):raise ValueError('V5 identity changed')
            reports[kind]=json.loads(p.read_text(encoding='utf8'));audits.append(audit)
        hashes={a['checkpoint_sha256'] for a in audits}
        if len(hashes)!=1:raise ValueError('Mixed checkpoint bundle')
        identities[stage]=hashes.pop()
        if stage=='baseline20' and identities[stage]!=BASELINE_SHA:raise ValueError('Wrong 20M baseline')
        stages[stage]=metrics(*[reports[k] for k in ('free','v3','v4','rawmath')]);raw[stage]=reports
    if 'baseline20' not in stages:raise ValueError('Complete exact baseline required')
    nis={s:nis_v1(v) for s,v in stages.items()};report={'definition':'Raw own Neral weights only; no tools or decoder penalties','stages':stages,'nis':nis,'checkpoint_sha256':identities,'protocol_sha256':sha(root/'protocol.json'),'decisions':{}}
    for s in stages:
        if s!='baseline20':report['decisions'][s]=decide(report,raw['baseline20']['v5'],raw[s]['v5'],candidate_name=s)
    (root/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    keys=['eos_pct','repeated_trigram_pct','utf8_valid_pct','rawmath_v1_exact_pct','v3_pair_overall_pct','v4_pair_overall_pct','true_byte_bpb_macro_1024']
    lines=['# Frozen local Neral comparison','','Model-only performance. EOS recognition with supplied answers is not free generation.','', '| Stage | EOS % | Repetition % | UTF8 % | RawMath % | v3 % | v4 % | BPB ↓ | NIS | Gate |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for s,m in stages.items():
        vals=[f'{m[k]:.3f}' for k in keys];g=report['decisions'].get(s)
        gate='baseline' if g is None else str(g['promotion']['strict_wins'])+'/9 wins; '+str(g['surpassed_20m'])
        lines.append('| '+s+' | '+' | '.join(vals)+f" | {nis[s]['score']:.3f} | {gate} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines))


if __name__=='__main__':main()
