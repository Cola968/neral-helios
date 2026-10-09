from __future__ import annotations
import argparse,collections,json,statistics,unicodedata
from pathlib import Path
SPLITS=("train","val","test")

def norm(x):return " ".join(unicodedata.normalize("NFKC",str(x)).casefold().split())
def read(p):
    rows=[]
    for n,line in enumerate(Path(p).read_text(encoding="utf8").splitlines(),1):
        if not line.strip():continue
        r=json.loads(line)
        if not isinstance(r,dict):raise ValueError(f"{p}:{n}: row must object")
        rows.append(r)
    if not rows:raise ValueError(f"empty split: {p}")
    return rows

def audit(root:Path):
    root=Path(root);out={"schema":1,"splits":{}}
    seen=set()
    for split in SPLITS:
        rs=read(root/f"{split}.jsonl");langs=collections.Counter();depths=collections.Counter();gaps=[];ratios=[];chosen_l=[];rejected_l=[]
        for r in rs:
            pid=r.get("pair_sha256")
            if not pid or pid in seen:raise ValueError("duplicate/empty preference pair")
            seen.add(pid)
            c=norm(r.get("chosen",""));j=norm(r.get("rejected",""))
            if not c or not j or c==j:raise ValueError("invalid identical/empty preference answers")
            cr=int(r["chosen_rank"]);rr=int(r["rejected_rank"])
            if cr>=rr:raise ValueError("preference ranks not strictly ordered")
            gaps.append(rr-cr);cb=len(c.encode("utf8"));rb=len(j.encode("utf8"));chosen_l.append(cb);rejected_l.append(rb);ratios.append(cb/max(1,rb))
            langs[str(r.get("language","")).lower()]+=1;depths[len(r.get("context") or [])]+=1
        extreme=sum(x<.25 or x>4.0 for x in ratios)
        out["splits"][split]={
          "pairs":len(rs),"languages":dict(sorted(langs.items())),"context_turn_counts":{str(k):v for k,v in sorted(depths.items())},
          "mean_rank_gap":statistics.fmean(gaps),"mean_chosen_bytes":statistics.fmean(chosen_l),"mean_rejected_bytes":statistics.fmean(rejected_l),
          "mean_length_ratio":statistics.fmean(ratios),"extreme_length_ratio_pairs":extreme,"extreme_length_ratio_fraction":extreme/len(rs)
        }
    tr=out["splits"]["train"]
    checks={"enough_train_pairs":tr["pairs"]>=100,"rank_signal":tr["mean_rank_gap"]>=1.0,"extreme_length_bias":tr["extreme_length_ratio_fraction"]<=.15,
            "both_languages":tr["languages"].get("de",0)>0 and tr["languages"].get("en",0)>0}
    out["gates"]={"checks":checks,"pass":all(checks.values())}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--out",required=True);ap.add_argument("--strict",action="store_true");a=ap.parse_args()
    r=audit(Path(a.root));Path(a.out).write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps(r["gates"]))
    if a.strict and not r["gates"]["pass"]:raise SystemExit(2)
if __name__=="__main__":main()
