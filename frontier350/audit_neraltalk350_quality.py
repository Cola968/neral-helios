from __future__ import annotations
import argparse,collections,hashlib,json,math,re,statistics,unicodedata
from pathlib import Path

SPLITS=("train","val","test")

def norm(text:str)->str:
    return " ".join(unicodedata.normalize("NFKC",str(text)).casefold().split())

def row_messages(r):
    if isinstance(r.get("messages"),list): return r["messages"]
    return [{"role":"user","content":r.get("prompt","")},{"role":"assistant","content":r.get("answer","")}]

def read(path):
    out=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf8").splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line)
        if not isinstance(r,dict): raise ValueError(f"{path}:{n}: row must be object")
        out.append(r)
    if not out: raise ValueError(f"empty split: {path}")
    return out

def pct(v,n): return 100.0*v/max(1,n)

def quantiles(xs):
    if not xs:return {}
    ys=sorted(xs)
    def q(p): return ys[min(len(ys)-1,max(0,round((len(ys)-1)*p)))]
    return {"min":ys[0],"p50":q(.5),"p90":q(.9),"p99":q(.99),"max":ys[-1],"mean":statistics.fmean(ys)}

def audit(root:Path)->dict:
    root=Path(root); result={"schema":1,"splits":{}}
    global_ids=set();global_conv=set()
    for split in SPLITS:
        rows=read(root/f"{split}.jsonl");lang=collections.Counter();dom=collections.Counter();src=collections.Counter();turns=collections.Counter()
        lengths=[];answer_lengths=[];prompt_hashes=collections.Counter();assistant_hashes=collections.Counter()
        human=0;multi=0
        for i,r in enumerate(rows):
            rid=str(r.get("id",""))
            if not rid or rid in global_ids: raise ValueError(f"duplicate/empty id: {rid}")
            global_ids.add(rid)
            msgs=row_messages(r)
            payload=json.dumps(msgs,ensure_ascii=False,sort_keys=True,separators=(",",":"))
            dig=hashlib.sha256(payload.encode("utf8")).hexdigest()
            if dig in global_conv: raise ValueError("exact conversation duplicated across dataset")
            global_conv.add(dig)
            language=str(r.get("language","")).lower();domain=str(r.get("domain","")).lower();source=str(r.get("source",""))
            lang[language]+=1;dom[domain]+=1;src[source]+=1
            turns[len(msgs)]+=1;multi+=int(len(msgs)>=4)
            is_human=("oasst" in source.casefold() and "human" in source.casefold())
            human+=int(is_human)
            texts=[str(m.get("content","")).strip() for m in msgs]
            lengths.append(sum(len(x.encode("utf8")) for x in texts))
            ans=[str(m.get("content","")).strip() for m in msgs if m.get("role")=="assistant"]
            answer_lengths.extend(len(x.encode("utf8")) for x in ans)
            users=[str(m.get("content","")).strip() for m in msgs if m.get("role")=="user"]
            if users: prompt_hashes[hashlib.sha256(norm(users[0]).encode()).hexdigest()]+=1
            for x in ans:
                if norm(x): assistant_hashes[hashlib.sha256(norm(x).encode()).hexdigest()]+=1
        result["splits"][split]={
          "rows":len(rows),"languages":dict(sorted(lang.items())),"domains":dict(sorted(dom.items())),"sources":dict(sorted(src.items())),
          "turn_counts":{str(k):v for k,v in sorted(turns.items())},
          "human_rows":human,"human_fraction":human/len(rows),"multiturn_rows":multi,"multiturn_fraction":multi/len(rows),
          "conversation_bytes":quantiles(lengths),"assistant_bytes":quantiles(answer_lengths),
          "duplicate_root_prompt_groups":sum(1 for v in prompt_hashes.values() if v>1),
          "duplicate_assistant_text_groups":sum(1 for v in assistant_hashes.values() if v>1),
        }
    tr=result["splits"]["train"]
    result["train_summary"]={
      "human_fraction":tr["human_fraction"],"multiturn_fraction":tr["multiturn_fraction"],
      "de_fraction":tr["languages"].get("de",0)/max(1,tr["rows"]),"en_fraction":tr["languages"].get("en",0)/max(1,tr["rows"]),
      "domain_count":len(tr["domains"]),"source_count":len(tr["sources"])
    }
    return result

def gates(report,min_human=.20,min_de=.15,min_domains=5,max_duplicate_answer_fraction=.15):
    tr=report["splits"]["train"];summary=report["train_summary"]
    dup=tr["duplicate_assistant_text_groups"]/max(1,tr["rows"])
    checks={
      "human_fraction":summary["human_fraction"]>=min_human,
      "german_fraction":summary["de_fraction"]>=min_de,
      "domain_diversity":summary["domain_count"]>=min_domains,
      "duplicate_answer_groups":dup<=max_duplicate_answer_fraction,
      "has_multiturn":tr["multiturn_rows"]>0,
    }
    return {"checks":checks,"pass":all(checks.values()),"thresholds":{"min_human":min_human,"min_de":min_de,"min_domains":min_domains,"max_duplicate_answer_fraction":max_duplicate_answer_fraction},"observed":{"duplicate_answer_group_fraction":dup}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--out",required=True);ap.add_argument("--strict",action="store_true");a=ap.parse_args()
    r=audit(Path(a.root));r["gates"]=gates(r);Path(a.out).write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({"train_summary":r["train_summary"],"gates":r["gates"]},ensure_ascii=False))
    if a.strict and not r["gates"]["pass"]: raise SystemExit(2)
if __name__=="__main__":main()
