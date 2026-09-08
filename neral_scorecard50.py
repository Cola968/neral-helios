"""Normalize a complete frozen Neral benchmark bundle into a compact cross-run scorecard."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from neral_intelligence_score import nis_v1

REFERENCE={
'I1':{'nis':65.24583333333334,'math':8.0,'targeted_math':40.0,'code':45.0,'de':80.0,'reasoning':70.0,'context':100.0,'eos':11.0,'repetition':7.408910533910534,'bpb4095':1.71070806794},
'M1':{'nis':63.77333333333333,'math':12.2,'targeted_math':45.0,'code':35.0,'de':80.0,'reasoning':65.0,'context':80.0,'eos':26.0,'repetition':7.6118006993,'bpb4095':1.754103694},
'M2':{'nis':66.18333333333334,'math':8.0,'targeted_math':40.0,'code':45.0,'de':75.0,'reasoning':70.0,'context':100.0,'eos':11.0,'repetition':9.46,'bpb4095':1.710567},
'M2.1':{'nis':65.24583333333334,'math':8.0,'targeted_math':40.0,'code':45.0,'de':80.0,'reasoning':70.0,'context':100.0,'eos':11.0,'repetition':7.408910533910534,'bpb4095':1.71070806794}}

def pct(v):return 100.*float(v)
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def bundle_metrics(root:Path):
 v3=load(root/'v3.json');v4=load(root/'v4.json');raw=load(root/'rawmath.json');tar=load(root/'targeted.json');free=load(root/'free.json');v5=load(root/'v5.json')
 d3=v3['pair_preference']['domains'];d4=v4['pair_suite']['domains'];m={}
 for n,k in [('de_grammar','v3_de_grammar_pct'),('en_grammar','v3_en_grammar_pct'),('facts','v3_facts_pct'),('commonsense','v3_commonsense_pct'),('code','v3_code_pct')]:m[k]=pct(d3[n]['accuracy'])
 for n,k in [('code','v4_code_pct'),('physics','v4_physics_pct'),('astronomy','v4_astronomy_pct'),('aviation','v4_aviation_pct'),('transport','v4_transport_pct'),('logic','v4_logic_pct'),('commonsense','v4_commonsense_pct'),('induction','v4_induction_pct'),('translation','v4_translation_pct')]:m[k]=pct(d4[n]['accuracy'])
 m['rawmath_v1_exact_pct']=float(raw['overall']['exact_pct']);m['context_recall_overall_pct']=pct(v4['context_recall']['accuracy'])
 score=nis_v1(m)['score'];td=tar['domains'];fs=free['summary'];bpb=float(v5['document_bounded_true_byte_bpb']['macro_bits_per_byte']['4095'])
 return {'nis':score,'math':float(raw['overall']['exact_pct']),'targeted_math':pct(td['math']['accuracy']),'code':pct(td['code']['accuracy']),'de':pct(td['de_grammar']['accuracy']),'reasoning':pct(td['reasoning']['accuracy']),'context':pct(v4['context_recall']['accuracy']),'eos':pct(fs['eos_fraction']),'repetition':pct(fs['mean_repeated_trigram_fraction']),'bpb4095':bpb}
def deltas(c,r):return {k:c[k]-r[k] for k in c if k in r}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--name',required=True);ap.add_argument('--out',default='scorecard');a=ap.parse_args();c=bundle_metrics(Path(a.bundle));refs=dict(REFERENCE);obj={'name':a.name,'metrics':c,'references':refs,'deltas':{n:deltas(c,r) for n,r in refs.items()},'semantics':{'math':'Frozen RawMath exact %','targeted_math':'supplemental targeted math %','repetition':'lower is better','bpb4095':'lower is better'}}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'NERAL_SCORECARD.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8');cols=['nis','math','code','de','reasoning','context','eos','repetition','bpb4095'];lines=[f'# Neral scorecard — {a.name}','','| Metric | Value | Δ I1 |','|---|---:|---:|']+[f'| {k} | {c[k]:.4f} | {c[k]-refs["I1"][k]:+.4f} |' for k in cols]+['','Math = frozen RawMath exact. Repetition and BPB are lower-is-better.'];(out/'NERAL_SCORECARD.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(obj,ensure_ascii=False))
if __name__=='__main__':main()
