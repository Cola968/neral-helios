from __future__ import annotations
import argparse,json
from pathlib import Path
from chat350_generation_metrics import compare_health

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--baseline",required=True); ap.add_argument("--candidate",required=True); ap.add_argument("--out",required=True); ap.add_argument("--training-summary",default="")
    a=ap.parse_args(); b=json.loads(Path(a.baseline).read_text(encoding="utf8")); c=json.loads(Path(a.candidate).read_text(encoding="utf8"))
    for k in ("tokenizer_sha256","case_sha256","evaluation_contract"):
        if b.get(k)!=c.get(k): raise ValueError(f"report identity mismatch: {k}")
    gate=compare_health(b["summary"],c["summary"])
    training=None
    if a.training_summary: training=json.loads(Path(a.training_summary).read_text(encoding="utf8"))
    trained_candidate=training is None or bool(training.get("best_checkpoint_exists"))
    decision="manual_review_required" if gate["health_no_regression"] and trained_candidate else ("reject_no_sft_improvement" if not trained_candidate else "reject_generation_regression")
    result={"baseline":b["summary"],"candidate":c["summary"],"gate":gate,"training":training,"automatic_promotion":False,"decision":decision}
    Path(a.out).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8"); print(json.dumps(result),flush=True)

if __name__=="__main__": main()
