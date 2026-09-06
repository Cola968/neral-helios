"""Fail-closed promotion gate for Neral-50M release candidates.

Consumes ``strong_comparison.json`` generated from frozen benchmark reports. It does not
invent an intelligence percentage and never promotes a checkpoint automatically. It only
states whether the candidate has demonstrated enough direct benchmark wins to justify RC
review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIRECTIONAL = {
    "eos_pct": "higher",
    "repeated_trigram_pct": "lower",
    "rawmath_v1_exact_pct": "higher",
    "v3_pair_overall_pct": "higher",
    "v4_pair_overall_pct": "higher",
    "context_recall_overall_pct": "higher",
    "true_byte_bpb_macro_1024": "lower",
    "v4_math_pct": "higher",
    "v4_code_pct": "higher",
}
MIN_STRICT_WINS = 8


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _metric(stage, key):
    value=stage.get(key)
    return None if value is None else float(value)


def evaluate(report, candidate_name="stage_c"):
    stages=report.get("stages",{})
    baseline=stages.get("baseline20",{})
    candidate=stages.get(candidate_name,{})
    missing=[]
    required=["utf8_valid_pct",*DIRECTIONAL]
    for key in required:
        if _metric(baseline,key) is None or _metric(candidate,key) is None:
            missing.append(key)

    rows=[];strict_wins=0
    for key,direction in DIRECTIONAL.items():
        b=_metric(baseline,key);c=_metric(candidate,key)
        if b is None or c is None:
            rows.append({"metric":key,"direction":direction,"baseline":b,"candidate":c,"strict_win":False,"missing":True})
            continue
        win=c>b if direction=="higher" else c<b
        strict_wins+=int(win)
        rows.append({"metric":key,"direction":direction,"baseline":b,"candidate":c,"delta":c-b,"strict_win":win,"missing":False})

    utf8_b=_metric(baseline,"utf8_valid_pct");utf8_c=_metric(candidate,"utf8_valid_pct")
    utf8_nonregressed=utf8_b is not None and utf8_c is not None and utf8_c>=utf8_b-0.1

    # Severe-regression guards are deliberately independent of the win count.
    severe=[]
    def add_if(condition,metric,detail):
        if condition:severe.append({"metric":metric,"detail":detail})
    if utf8_b is not None and utf8_c is not None:
        add_if(utf8_c<utf8_b-1.0,"utf8_valid_pct","more than 1 percentage point below 20M")
    rep_b=_metric(baseline,"repeated_trigram_pct");rep_c=_metric(candidate,"repeated_trigram_pct")
    if rep_b is not None and rep_c is not None:
        add_if(rep_c>rep_b+5.0,"repeated_trigram_pct","more than 5 percentage points worse than 20M")
    bpb_b=_metric(baseline,"true_byte_bpb_macro_1024");bpb_c=_metric(candidate,"true_byte_bpb_macro_1024")
    if bpb_b is not None and bpb_c is not None and bpb_b>0:
        add_if(bpb_c>bpb_b*1.05,"true_byte_bpb_macro_1024","more than 5% worse than 20M")
    for key in ("v3_pair_overall_pct","v4_pair_overall_pct","context_recall_overall_pct"):
        b=_metric(baseline,key);c=_metric(candidate,key)
        if b is not None and c is not None:
            add_if(c<b-10.0,key,"more than 10 percentage points below 20M")

    ready=not missing and utf8_nonregressed and strict_wins>=MIN_STRICT_WINS and not severe
    return {
        "candidate":candidate_name,
        "rule":"At least 8 strict wins across 9 directional core metrics, UTF-8 non-regression, all metrics present, and no severe regression.",
        "release_ready":False,
        "ready_for_rc_review":ready,
        "strict_wins":strict_wins,
        "strict_wins_required":MIN_STRICT_WINS,
        "utf8_nonregressed":utf8_nonregressed,
        "missing_metrics":missing,
        "severe_regressions":severe,
        "scorecard":rows,
        "note":"This gate is a benchmark decision rule, not a composite intelligence score. Human quality, licensing, security and release review remain separate.",
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument("comparison");ap.add_argument("--candidate",default="stage_c");ap.add_argument("--out",required=True);args=ap.parse_args()
    result=evaluate(_load(args.comparison),args.candidate)
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
