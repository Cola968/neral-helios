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
        found=re.findall(r"(?<![\w.,])[+-]?\d+(?![\w]|[.,]\d)",str(text))
        return bool(found) and int(found[-1])==int(spec["value"])
    if typ=="contains_all":
        return all(norm(x) in t for x in spec.get("values",[]))
    if typ=="contains_any":
        ok=any(norm(x) in t for x in spec.get("values",[]))
        return ok and all(norm(x) in t for x in spec.get("and_values",[]))
    raise ValueError(f"unknown scoring type: {typ}")

def score(report:dict,cases:list[dict])->dict:
    """Score exactly one output per immutable benchmark case, fail closed."""
    if not isinstance(cases,list) or not cases or not all(
        isinstance(c,dict) and isinstance(c.get("id"),str) and c["id"]
        and isinstance(c.get("category"),str)
        and isinstance(c.get("scoring"),dict) for c in cases
    ):
        raise ValueError("invalid frozen case specification")
    cmap={c["id"]:c for c in cases}
    if len(cmap)!=len(cases):raise ValueError("duplicate frozen case ids")
    rows=report.get("rows") if isinstance(report,dict) else None
    if not isinstance(rows,list) or len(rows)!=len(cases):
        raise ValueError("report missing/extra generation rows")
    if not all(isinstance(r,dict) and isinstance(r.get("id"),str)
               and isinstance(r.get("text"),str) for r in rows):
        raise ValueError("report rows require case ids and generated text")
    got={r["id"]:r for r in rows}
    if len(got)!=len(rows):raise ValueError("duplicate generated case id")
    if set(got)!=set(cmap):
        raise ValueError(f"case/report id mismatch: missing={sorted(set(cmap)-set(got))} extra={sorted(set(got)-set(cmap))}")
    # Fingerprint the entire prompt, scoring rule and generation budget:
    # matching IDs/categories alone would allow a silent benchmark rewrite.
    contract=json.dumps(sorted(cases,key=lambda r:r["id"]),ensure_ascii=False,
                        sort_keys=True,separators=(",",":")).encode("utf8")
    import hashlib
    case_sha256=hashlib.sha256(contract).hexdigest()
    cats=collections.defaultdict(list);detail=[]
    for cid,c in cmap.items():
        text=got[cid]["text"]
        ok=score_one(text,c["scoring"]);cats[c["category"]].append(ok)
        detail.append({"id":cid,"category":c["category"],"pass":ok,"text":text})
    by={k:{"passed":sum(v),"cases":len(v),"pct":100*sum(v)/len(v)} for k,v in sorted(cats.items())}
    total=sum(x["pass"] for x in detail)
    return {"cases":len(detail),"passed":total,"overall_pct":100*total/len(detail),
            "case_contract_sha256":case_sha256,
            "by_category":by,"details":detail}

def compare(base:dict,cand:dict)->dict:
    if not base.get("case_contract_sha256") or base["case_contract_sha256"]!=cand.get("case_contract_sha256"):
        raise ValueError("frozen benchmark cases, prompts or answer keys differ")
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
