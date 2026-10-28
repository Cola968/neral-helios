from __future__ import annotations
import argparse,json
from pathlib import Path
from chat350_generation_metrics import compare_health


def comparison_decision(baseline:dict,candidate:dict,training:dict|None=None)->dict:
    """No positive DEV review status without any observed exact instruction win.

    This is a minimum evidence floor, not a quality/release certification.
    A lower repetition score or higher EOS alone cannot redeem zero correct
    exact-instruction examples. The sealed independent capability suite is
    still required before any model is promotable.
    """
    gate=compare_health(baseline,candidate)
    trained_candidate=training is None or bool(training.get("best_checkpoint_exists"))
    raw=candidate.get("exact_instruction_pct")
    absolute_exact_success_observed=(type(raw) in (int,float) and raw>0)
    if not trained_candidate:
        decision="reject_no_sft_improvement"
    elif not gate["health_no_regression"]:
        decision="reject_generation_regression"
    elif not absolute_exact_success_observed:
        decision="reject_zero_exact_instruction"
    else:
        decision="manual_review_required"
    return {
        "baseline":baseline,"candidate":candidate,
        "gate":gate,"training":training,
        "absolute_exact_success_observed":absolute_exact_success_observed,
        "automatic_promotion":False,"decision":decision,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--training-summary",default="")
    a=ap.parse_args()
    b=json.loads(Path(a.baseline).read_text(encoding="utf8"))
    c=json.loads(Path(a.candidate).read_text(encoding="utf8"))
    for k in ("tokenizer_sha256","case_sha256","evaluation_contract"):
        if b.get(k)!=c.get(k):
            raise ValueError(f"report identity mismatch: {k}")
    training=None
    if a.training_summary:
        training=json.loads(Path(a.training_summary).read_text(encoding="utf8"))
    result=comparison_decision(b["summary"],c["summary"],training)
    Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
