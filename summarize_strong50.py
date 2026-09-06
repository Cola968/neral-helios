"""Summarize 20M, 50M Pilot, Stage B and RawMath Stage C.

Direct frozen metrics remain canonical. A versioned NIS 1.0 secondary composite is added
when every required component is present; missing components are reported, never reweighted.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from summarize_benchmarks50 import flatten
from neral_intelligence_score import nis_v1

STAGES=("baseline20","pilot50","stage_b","stage_c")

def load(path):
    p=Path(path) if path else None
    return json.loads(p.read_text(encoding="utf8")) if p and p.is_file() else None

def rawmath_metrics(r):
    if not r:return {}
    out={"rawmath_v1_exact_pct":r.get("overall",{}).get("exact_pct"),"rawmath_v1_eos_pct":r.get("overall",{}).get("eos_pct"),"rawmath_v1_answer_nll":r.get("overall",{}).get("answer_nll")}
    for k,v in r.get("by_category",{}).items():out[f"rawmath_{k}_pct"]=v.get("exact_pct")
    for k,v in r.get("by_distribution",{}).items():out[f"rawmath_distribution_{k}_pct"]=v.get("exact_pct")
    return out

def metrics(free,v3,v4,rawmath):return flatten(free,v3,v4)|rawmath_metrics(rawmath)

def score_or_incomplete(m):
    try:return nis_v1(m)
    except ValueError as e:return {"name":"Neral Intelligence Score","version":"1.0","status":"incomplete","score":None,"reason":str(e)}

def main():
    ap=argparse.ArgumentParser()
    for s in STAGES:
        for kind in ("free","v3","v4","rawmath"):ap.add_argument(f"--{s}-{kind}")
    ap.add_argument("--stage-b-summary");ap.add_argument("--stage-c-summary");ap.add_argument("--out-json",required=True);ap.add_argument("--out-md",required=True);a=ap.parse_args()
    sets={s:metrics(load(getattr(a,f"{s}_free")),load(getattr(a,f"{s}_v3")),load(getattr(a,f"{s}_v4")),load(getattr(a,f"{s}_rawmath"))) for s in STAGES}
    nis={s:score_or_incomplete(sets[s]) for s in STAGES}
    report={"definition":"Direct frozen benchmark metrics are canonical. NIS 1.0 is a secondary versioned Neral-only composite.","stages":sets,"nis":nis,"stage_b_training":load(a.stage_b_summary),"stage_c_training":load(a.stage_c_summary)}
    preferred=["utf8_valid_pct","eos_pct","repeated_trigram_pct","non_repetition_trigram_pct","raw_math_exact_pct","rawmath_v1_exact_pct","rawmath_v1_eos_pct","rawmath_add_sub_pct","rawmath_mul_div_pct","rawmath_negatives_pct","rawmath_fractions_pct","rawmath_percent_pct","rawmath_powers_pct","rawmath_linear_pct","rawmath_sequence_pct","rawmath_word_pct","rawmath_transfer_ood_pct","rawmath_distribution_id_pct","rawmath_distribution_ood_pct","rawmath_distribution_transfer_pct","v3_pair_overall_pct","v4_pair_overall_pct","context_recall_overall_pct","decode_tokens_per_second","rawmath_v1_answer_nll"]
    names={"baseline20":"20M Preview 02","pilot50":"50M Pilot A","stage_b":"50M + full language","stage_c":"50M + RawMath"}
    lines=["# Neral strong-training benchmark","","Direct scores below come from the frozen suites. SOLVE/Helios results are excluded from raw Neral metrics.","","| Metric | 20M Preview 02 | 50M Pilot A | 50M + language | 50M + RawMath |","|---|---:|---:|---:|---:|"]
    for k in preferred:
        vals=[sets[s].get(k) for s in STAGES]
        if all(v is None for v in vals):continue
        suffix="" if k in {"decode_tokens_per_second","rawmath_v1_answer_nll"} else "%"
        def f(v):return "—" if v is None else f"{v:.2f}{suffix}"
        lines.append("| "+k.replace("_"," ")+" | "+" | ".join(f(v) for v in vals)+" |")
    lines.extend(["","## Neral Intelligence Score 1.0","","NIS is secondary and Neral-only; it never replaces the direct promotion gates.","","| Model | NIS 1.0 | Status |","|---|---:|---|"])
    for s in STAGES:
        r=nis[s];score="—" if r.get("score") is None else f"{r['score']:.2f}"
        lines.append(f"| {names[s]} | {score} | {r['status']} |")
    raw_keys=[k for k in preferred if k.startswith("rawmath_") and k.endswith("_pct")]
    wins={s:0 for s in STAGES}
    for k in raw_keys:
        present={s:sets[s].get(k) for s in STAGES if sets[s].get(k) is not None}
        if present:
            m=max(present.values())
            for s,v in present.items():
                if v==m:wins[s]+=1
    report["rawmath_category_or_slice_wins_including_ties"]=wins
    lines.extend(["","RawMath category/slice wins (ties count for each tied stage): "+", ".join(f"{names[s]}={wins[s]}" for s in STAGES)+"."])
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    Path(a.out_md).write_text("\n".join(lines)+"\n",encoding="utf8");print(json.dumps({"out":a.out_md,"stage_metric_counts":{s:len(v) for s,v in sets.items()},"nis":{s:r.get('score') for s,r in nis.items()}},ensure_ascii=False))
if __name__=="__main__":main()
