from __future__ import annotations
import argparse,collections,json,re
from pathlib import Path

def norm(text:str)->str:
    return " ".join(str(text).casefold().strip().split())

def score_one(text:str,spec:dict)->bool:
    t=norm(text);typ=spec.get("type")
    if typ=="exact":
        return str(text).strip()==str(spec["value"]).strip()
    if typ=="integer":
        # Match signed integers even when followed by normal punctuation ("95.", "-7,").
        # Thousands/decimal separators are intentionally unsupported for this frozen benchmark.
        found=re.findall(r"(?<!\d)[+-]?\d+(?!\d)",str(text))
        return bool(found) and int(found[-1])==int(spec["value"])
    if typ=="contains_all":
        return all(norm(x) in t for x in spec.get("values",[]))
    if typ=="contains_any":
        ok=any(norm(x) in t for x in spec.get("values",[]))
        return ok and all(norm(x) in t for x in spec.get("and_values",[]))
    raise ValueError(f"unknown scoring type: {typ}")

def score(report:dict,cases:list[dict])->dict:
    cmap={c["id"]:c for c in cases}
    rows=report.get("rows")
    if not isinstance(rows,list):raise ValueError("report missing rows")
    got={r.get("id"):r for r in rows}
    if set(got)!=set(cmap):raise ValueError(f"case/report id mismatch: missing={sorted(set(cmap)-set(got))} extra={sorted(set(got)-set(cmap))}")
    cats=collections.defaultdict(list);detail=[]
    for cid,c in cmap.items():
        text=str(got[cid].get("text",""))
        ok=score_one(text,c["scoring"]);cats[c["category"]].append(ok)
        detail.append({"id":cid,"category":c["category"],"pass":ok,"text":text})
    by={k:{"passed":sum(v),"cases":len(v),"pct":100*sum(v)/len(v)} for k,v in sorted(cats.items())}
    total=sum(x["pass"] for x in detail)
    return {"cases":len(detail),"passed":total,"overall_pct":100*total/len(detail),"by_category":by,"details":detail}

def compare(base:dict,cand:dict)->dict:
    cats=sorted(set(base["by_category"])|set(cand["by_category"]))
    deltas={k:cand["by_category"].get(k,{"pct":0})["pct"]-base["by_category"].get(k,{"pct":0})["pct"] for k in cats}
    hard_reg=[k for k,d in deltas.items() if d < -20.0]
    return {"overall_delta_pp":cand["overall_pct"]-base["overall_pct"],"category_delta_pp":deltas,"hard_regressions":hard_reg,
            "no_hard_category_regression":not hard_reg,"nonnegative_overall":cand["overall_pct"]>=base["overall_pct"],
            "capability_no_regression":not hard_reg and cand["overall_pct"]>=base["overall_pct"]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--report",required=True);ap.add_argument("--cases",required=True);ap.add_argument("--out",required=True);ap.add_argument("--baseline-score");a=ap.parse_args()
    report=json.loads(Path(a.report).read_text(encoding="utf8"));cases=json.loads(Path(a.cases).read_text(encoding="utf8"));res=score(report,cases)
    if a.baseline_score:
        base=json.loads(Path(a.baseline_score).read_text(encoding="utf8"));res["comparison"]=compare(base,res)
    Path(a.out).write_text(json.dumps(res,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({k:v for k,v in res.items() if k!="details"},ensure_ascii=False))
if __name__=="__main__":main()
