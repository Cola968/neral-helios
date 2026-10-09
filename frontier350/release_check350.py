from __future__ import annotations
import argparse,json
from pathlib import Path

def load(path):return json.loads(Path(path).read_text(encoding="utf8"))
def check(data_quality,sft_final,sft_capability,sft_slice,dpo_final=None,dpo_capability=None):
    checks={
      "data_quality":bool(data_quality.get("gates",{}).get("pass")),
      "sft_manual_review":sft_final.get("decision")=="manual_review_required",
      "sft_capability":bool(sft_capability.get("comparison",{}).get("capability_no_regression")),
      "sft_slice":bool(sft_slice.get("slice_no_regression")),
    }
    stage="sft_candidate"
    if dpo_final is not None:
        checks["dpo_manual_review"]=dpo_final.get("decision")=="manual_review_required"
        checks["dpo_replay"]=bool(dpo_final.get("replay_guard"))
        checks["dpo_generation"]=bool(dpo_final.get("generation_guard"))
        checks["dpo_capability"]=bool(dpo_final.get("capability_guard"))
        if dpo_capability is not None:checks["dpo_capability_report"]=bool(dpo_capability.get("comparison",{}).get("capability_no_regression"))
        stage="dpo_candidate"
    return {"schema":1,"stage":stage,"checks":checks,"machine_gates_pass":all(checks.values()),"automatic_release":False,
            "decision":"manual_release_review_required" if all(checks.values()) else "reject",
            "note":"Passing machine gates is necessary but not sufficient; manual qualitative review and external benchmark evaluation remain required."}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-quality",required=True);ap.add_argument("--sft-final",required=True);ap.add_argument("--sft-capability",required=True);ap.add_argument("--sft-slice",required=True);ap.add_argument("--dpo-final");ap.add_argument("--dpo-capability");ap.add_argument("--out",required=True);a=ap.parse_args()
    r=check(load(a.data_quality),load(a.sft_final),load(a.sft_capability),load(a.sft_slice),load(a.dpo_final) if a.dpo_final else None,load(a.dpo_capability) if a.dpo_capability else None)
    Path(a.out).write_text(json.dumps(r,indent=2)+"\n",encoding="utf8");print(json.dumps(r))
if __name__=="__main__":main()
