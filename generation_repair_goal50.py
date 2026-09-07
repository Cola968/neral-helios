"""Hard acceptance gate for the Neral-50M G1 generation-repair stage.

The gate consumes only frozen raw-model benchmark reports. It does not alter decoding or model
weights. Thresholds mirror GitHub issue #36 and intentionally protect B2's context/code/language
modeling gains while requiring a real termination/repetition improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation_health_gate50 import extract_metrics
from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics as direct_metrics

TARGETS={
    "utf8_valid_pct_min":99.0,
    "free_eos_pct_min":10.0,
    "free_repeated_trigram_pct_max":8.0,
    "v3_chat_eos_pct_min":10.0,
    "v3_chat_repeated_trigram_pct_max":20.0,
    "v4_greedy_eos_pct_min":10.0,
    "v4_greedy_repeated_trigram_pct_max":20.0,
    "nis_min":48.0,
    "v4_pair_overall_pct_min":62.0,
    "context_recall_overall_pct_min":70.0,
    "v4_code_pct_min":60.0,
    "v4_math_pct_min":33.3,
    "true_byte_bpb_macro_1024_max":2.50,
}


def load(path):return json.loads(Path(path).read_text(encoding="utf8"))


def assess(before_free,before_v3,before_v4,before_rawmath,after_free,after_v3,after_v4,after_rawmath):
    before=direct_metrics(before_free,before_v3,before_v4,before_rawmath)
    after=direct_metrics(after_free,after_v3,after_v4,after_rawmath)
    gen=extract_metrics(after_free,after_v3,after_v4)
    missing=[]
    required=("v4_pair_overall_pct","context_recall_overall_pct","v4_code_pct","v4_math_pct","true_byte_bpb_macro_1024","rawmath_v1_exact_pct")
    for key in required:
        if after.get(key) is None:missing.append(key)
    for key,value in gen.items():
        if value is None:missing.append(key)
    try:nis=nis_v1(after)
    except ValueError as exc:
        nis={"score":None,"status":"incomplete","reason":str(exc)};missing.append("nis_1_0")
    if missing:
        return {"proceed":False,"reason":"missing_metrics","missing":sorted(set(missing)),"targets":TARGETS,"candidate":after,"generation":gen,"nis":nis}

    failures=[]
    checks=(
        (gen["utf8_valid_pct"]>=TARGETS["utf8_valid_pct_min"],"utf8_below_99pct"),
        (gen["eos_pct"]>=TARGETS["free_eos_pct_min"],"free_eos_below_10pct"),
        (gen["repeated_trigram_pct"]<=TARGETS["free_repeated_trigram_pct_max"],"free_repetition_above_8pct"),
        (gen["v3_chat_eos_pct"]>=TARGETS["v3_chat_eos_pct_min"],"v3_chat_eos_below_10pct"),
        (gen["v3_chat_repeated_trigram_pct"]<=TARGETS["v3_chat_repeated_trigram_pct_max"],"v3_chat_repetition_above_20pct"),
        (gen["v4_greedy_eos_pct"]>=TARGETS["v4_greedy_eos_pct_min"],"v4_greedy_eos_below_10pct"),
        (gen["v4_greedy_repeated_trigram_pct"]<=TARGETS["v4_greedy_repeated_trigram_pct_max"],"v4_greedy_repetition_above_20pct"),
        (nis["score"]>=TARGETS["nis_min"],"nis_below_48"),
        (after["v4_pair_overall_pct"]>=TARGETS["v4_pair_overall_pct_min"],"v4_overall_below_62pct"),
        (after["context_recall_overall_pct"]>=TARGETS["context_recall_overall_pct_min"],"context_below_70pct"),
        (after["v4_code_pct"]>=TARGETS["v4_code_pct_min"],"code_below_60pct"),
        (after["v4_math_pct"]>=TARGETS["v4_math_pct_min"],"math_below_33_3pct"),
        (after["true_byte_bpb_macro_1024"]<=TARGETS["true_byte_bpb_macro_1024_max"],"bpb1024_above_2_50"),
        (after["rawmath_v1_exact_pct"]>=before.get("rawmath_v1_exact_pct",0),"rawmath_exact_regressed"),
    )
    failures=[name for ok,name in checks if not ok]
    deltas={k:after[k]-before[k] for k in set(before)&set(after) if isinstance(before.get(k),(int,float)) and isinstance(after.get(k),(int,float))}
    stretch={"nis_ge_50":nis["score"]>=50.0,"v4_math_ge_50pct":after["v4_math_pct"]>=50.0}
    return {
        "proceed":not failures,"reason":"pass" if not failures else "g1_goal_not_met","goal_issue":36,
        "targets":TARGETS,"failures":failures,"candidate":after,"generation":gen,"nis":nis,"delta_after_minus_before":deltas,"stretch":stretch,
        "next_action":"review_for_promotion_or_next_benchmark_gated_stage" if not failures else "do_not_compound_checkpoint; diagnose G1 result",
        "note":"Frozen raw-model acceptance only; no SOLVE/VERIFY/Helios capability and no decoding hacks counted.",
    }


def main():
    ap=argparse.ArgumentParser()
    for prefix in ("before","after"):
        for kind in ("free","v3","v4","rawmath"):ap.add_argument(f"--{prefix}-{kind}",required=True)
    ap.add_argument("--out",required=True);a=ap.parse_args()
    result=assess(load(a.before_free),load(a.before_v3),load(a.before_v4),load(a.before_rawmath),load(a.after_free),load(a.after_v3),load(a.after_v4),load(a.after_rawmath))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result["proceed"]:raise SystemExit(2)


if __name__=="__main__":main()
