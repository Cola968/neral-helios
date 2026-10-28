from __future__ import annotations
import argparse,json
from pathlib import Path

def compare(base:dict,cand:dict,*,max_slice_regression=.10,min_tokens=128,min_rows=4):
    for k in ("tokenizer_sha256","dataset_manifest_sha256","split","max_seq"):
        if base.get(k)!=cand.get(k):raise ValueError(f"slice report identity mismatch: {k}")
    bs=base["slices"];cs=cand["slices"]
    if "overall" not in bs or "overall" not in cs:raise ValueError("missing overall slice")
    b0=float(bs["overall"]["answer_nll"]);c0=float(cs["overall"]["answer_nll"])
    eligible={};hard=[]
    for name in sorted(set(bs)&set(cs)):
        b=bs[name];c=cs[name]
        if min(int(b.get("answer_tokens",0)),int(c.get("answer_tokens",0)))<min_tokens or min(int(b.get("rows",0)),int(c.get("rows",0)))<min_rows:continue
        bn=float(b["answer_nll"]);cn=float(c["answer_nll"]);reg=cn/max(bn,1e-12)-1.0
        eligible[name]={"baseline_nll":bn,"candidate_nll":cn,"relative_change":reg,"rows":c["rows"],"answer_tokens":c["answer_tokens"]}
        if name!="overall" and reg>max_slice_regression:hard.append(name)
    # An improved overall NLL is not proof that language/domain/sourcing
    # did not regress if every actual subgroup is filtered out.
    eligible_subgroups=sorted(name for name in eligible if name!="overall")
    skipped=int(bs["overall"].get("skipped_too_long",0))+int(cs["overall"].get("skipped_too_long",0))
    result={"overall_baseline_nll":b0,"overall_candidate_nll":c0,"overall_relative_change":c0/max(b0,1e-12)-1.0,
            "overall_improved":c0<b0,"eligible_slices":eligible,"hard_regressions":hard,
            "eligible_subgroups":eligible_subgroups,"skipped_too_long_total":skipped,
            "slice_no_regression":c0<b0 and bool(eligible_subgroups) and skipped==0 and not hard,"thresholds":{"max_slice_regression":max_slice_regression,"min_tokens":min_tokens,"min_rows":min_rows}}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--baseline",required=True);ap.add_argument("--candidate",required=True);ap.add_argument("--out",required=True);ap.add_argument("--max-slice-regression",type=float,default=.10);a=ap.parse_args()
    b=json.loads(Path(a.baseline).read_text(encoding="utf8"));c=json.loads(Path(a.candidate).read_text(encoding="utf8"));r=compare(b,c,max_slice_regression=a.max_slice_regression)
    Path(a.out).write_text(json.dumps(r,indent=2)+"\n",encoding="utf8");print(json.dumps(r))
if __name__=="__main__":main()
