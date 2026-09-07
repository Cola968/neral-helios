"""Build the frozen 20M -> B2 -> RawMath -> Instruct comparison for Neral-50M RC work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics

STAGES=("baseline20","b2","rawmath","instruct")


def load(path):
    p=Path(path) if path else None
    return json.loads(p.read_text(encoding="utf8")) if p and p.is_file() else None


def stage(args,prefix):
    return metrics(load(getattr(args,f"{prefix}_free")),load(getattr(args,f"{prefix}_v3")),load(getattr(args,f"{prefix}_v4")),load(getattr(args,f"{prefix}_rawmath")))


def main():
    ap=argparse.ArgumentParser()
    for prefix in STAGES:
        for kind in ("free","v3","v4","rawmath"):
            ap.add_argument(f"--{prefix}-{kind}",required=True)
    ap.add_argument("--rawmath-summary");ap.add_argument("--instruct-summary")
    ap.add_argument("--out-json",required=True);ap.add_argument("--out-md",required=True);a=ap.parse_args()
    sets={s:stage(a,s) for s in STAGES}
    free={s:load(getattr(a,f"{s}_free")) for s in STAGES}
    hashes={s:r.get("suite_sha256") for s,r in free.items() if r}
    nonnull={v for v in hashes.values() if v}
    if len(hashes)!=len(STAGES) or any(not v for v in hashes.values()) or len(nonnull)!=1:raise ValueError(f"Frozen free-generation suite mismatch or missing hash: {hashes}")
    settings=[r.get('generation') for r in free.values()]
    if any(s is None for s in settings) or any(s!=settings[0] for s in settings):
        raise ValueError('Frozen free-generation settings mismatch or missing settings')
    nis={s:nis_v1(sets[s]) for s in STAGES}
    delta={k:sets["instruct"][k]-sets["baseline20"][k] for k in set(sets["instruct"])&set(sets["baseline20"]) if sets["instruct"][k] is not None and sets["baseline20"][k] is not None}
    report={
        "definition":"Same frozen raw-model harness at every checkpoint; SOLVE/VERIFY/Helios excluded.",
        "suite_hashes":hashes,"stages":sets,"nis":nis,"delta_instruct_minus_20m":delta,
        "rawmath_training":load(a.rawmath_summary),"instruct_training":load(a.instruct_summary),
    }
    Path(a.out_json).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    preferred=[
        "utf8_valid_pct","eos_pct","repeated_trigram_pct","non_repetition_trigram_pct","rawmath_v1_exact_pct",
        "v3_pair_overall_pct","v4_pair_overall_pct","v4_math_pct","v4_code_pct","context_recall_overall_pct",
        "true_byte_bpb_macro_1024","decode_tokens_per_second",
    ]
    names={"baseline20":"20M Preview02","b2":"50M B2","rawmath":"50M + RawMath","instruct":"50M + Instruct"}
    lines=["# Neral-50M RC benchmark","","Same frozen harness. Tools are excluded from model capability.","",
           "| Metric | 20M Preview02 | 50M B2 | 50M + RawMath | 50M + Instruct | Δ Instruct-20M |","|---|---:|---:|---:|---:|---:|"]
    for key in preferred:
        vals=[sets[s].get(key) for s in STAGES]
        if all(v is None for v in vals):continue
        plain=key=="decode_tokens_per_second" or key.startswith("true_byte_bpb_");suffix="" if plain else "%"
        f=lambda v:"—" if v is None else f"{v:.3f}{suffix}"
        d=delta.get(key)
        lines.append(f"| {key.replace('_',' ')} | {f(vals[0])} | {f(vals[1])} | {f(vals[2])} | {f(vals[3])} | {f(d)} |")
    lines.extend(["","## Neral Intelligence Score 1.0","","| Stage | NIS 1.0 |","|---|---:|"])
    for s in STAGES:lines.append(f"| {names[s]} | **{nis[s]['score']:.3f}** |")
    lines.extend(["",f"NIS delta Instruct vs 20M: **{nis['instruct']['score']-nis['baseline20']['score']:+.3f}**.",
                  "", "A higher NIS is secondary evidence only. promotion_gate50.py remains the hard RC decision rule."])
    Path(a.out_md).write_text("\n".join(lines)+"\n",encoding="utf8")
    print(json.dumps({"saved":a.out_md,"nis":{s:nis[s]["score"] for s in STAGES},"suite_hashes":hashes},ensure_ascii=False))


if __name__=="__main__":main()
