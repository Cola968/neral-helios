from __future__ import annotations
import argparse,json
from pathlib import Path

def compare(gb,gc,mb,mc):
    if int(gb["cases"])!=int(gc["cases"]) or int(mb["cases"])!=int(mc["cases"]):
        raise ValueError("baseline/candidate case-count mismatch")
    checks={
        "gsm8k_accuracy_no_regression":float(gc["accuracy"])>=float(gb["accuracy"]),
        "gsm8k_parse_no_material_regression":float(gc["parse_fraction"])+.02>=float(gb["parse_fraction"]),
        "mbpp_pass_at_1_no_regression":float(mc["pass_at_1"])>=float(mb["pass_at_1"]),
    }
    return {
        "checks":checks,"pass":all(checks.values()),"automatic_promotion":False,
        "gsm8k":{"baseline_accuracy":gb["accuracy"],"candidate_accuracy":gc["accuracy"],"delta":float(gc["accuracy"])-float(gb["accuracy"])},
        "mbpp":{"baseline_pass_at_1":mb["pass_at_1"],"candidate_pass_at_1":mc["pass_at_1"],"delta":float(mc["pass_at_1"])-float(mb["pass_at_1"])}
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gsm-baseline",required=True);ap.add_argument("--gsm-candidate",required=True)
    ap.add_argument("--mbpp-baseline",required=True);ap.add_argument("--mbpp-candidate",required=True)
    ap.add_argument("--out",required=True);a=ap.parse_args()
    L=lambda p:json.loads(Path(p).read_text(encoding="utf8"))
    r=compare(L(a.gsm_baseline),L(a.gsm_candidate),L(a.mbpp_baseline),L(a.mbpp_candidate))
    Path(a.out).write_text(json.dumps(r,indent=2)+"\n",encoding="utf8");print(json.dumps(r))
    if not r["pass"]:raise SystemExit(2)
if __name__=="__main__":main()
