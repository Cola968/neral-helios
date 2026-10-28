from __future__ import annotations
import argparse,json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def check(data_quality,sft_final,sft_capability,sft_slice,dpo_final=None,dpo_capability=None,external_decision=None,require_external=False):
    # Strict booleans matter: bool("false") and bool(1) are True in Python,
    # which could let malformed JSON approve a weaker Arctus checkpoint.
    cap = sft_capability.get("comparison", {})
    checks={
        "data_quality":data_quality.get("gates",{}).get("pass") is True,
        "sft_manual_review":sft_final.get("decision")=="manual_review_required",
        "sft_capability":(
            sft_capability.get("cases") == 30
            and type(sft_capability.get("cases")) is int
            and cap.get("capability_no_regression") is True
            and cap.get("category_no_regression") is True
        ),
        "sft_slice":sft_slice.get("slice_no_regression") is True,
    }
    stage="sft_candidate"
    if dpo_final is not None:
        checks["dpo_manual_review"]=dpo_final.get("decision")=="manual_review_required"
        checks["dpo_replay"]=dpo_final.get("replay_guard") is True
        checks["dpo_generation"]=dpo_final.get("generation_guard") is True
        checks["dpo_capability"]=dpo_final.get("capability_guard") is True
        if dpo_capability is not None:
            checks["dpo_capability_report"]=(dpo_capability.get("cases") == 30 and type(dpo_capability.get("cases")) is int and dpo_capability.get("comparison",{}).get("capability_no_regression") is True and dpo_capability.get("comparison",{}).get("category_no_regression") is True)
        stage="dpo_candidate"
    external_present=external_decision is not None
    if external_present:
        checks["external_holdouts_no_regression"]=external_decision.get("pass") is True
    elif require_external:
        checks["external_holdouts_present"]=False
    passed=all(checks.values())
    return {
        "schema":1,
        "stage":stage,
        "checks":checks,
        "machine_gates_pass":passed,
        "external_evidence_present":external_present,
        "external_evidence_required":bool(require_external),
        "external_evidence_scope":"supplementary; historical pretraining contamination absence is not certified",
        "automatic_release":False,
        "decision":"manual_release_review_required" if passed else "reject",
        "note":"Passing machine gates is necessary but not sufficient; manual qualitative review remains required."
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-quality",required=True)
    ap.add_argument("--sft-final",required=True)
    ap.add_argument("--sft-capability",required=True)
    ap.add_argument("--sft-slice",required=True)
    ap.add_argument("--dpo-final")
    ap.add_argument("--dpo-capability")
    ap.add_argument("--external-decision")
    ap.add_argument("--require-external",action="store_true")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    r=check(
        load(a.data_quality),
        load(a.sft_final),
        load(a.sft_capability),
        load(a.sft_slice),
        load(a.dpo_final) if a.dpo_final else None,
        load(a.dpo_capability) if a.dpo_capability else None,
        load(a.external_decision) if a.external_decision else None,
        a.require_external,
    )
    Path(a.out).write_text(json.dumps(r,indent=2)+"\n",encoding="utf8")
    print(json.dumps(r))
if __name__=="__main__":
    main()
