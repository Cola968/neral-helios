from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
try:
 from .chat_cli350 import load_runtime,generate
 from .external_bench350_scoring import gsm_target as target,gsm_response as response
except ImportError:
 from chat_cli350 import load_runtime,generate
 from external_bench350_scoring import gsm_target as target,gsm_response as response
def read(path):
 rs=[json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]
 if not rs:raise ValueError("empty GSM8K JSONL")
 return rs
def subset(rows,limit):
 rows=sorted(rows,key=lambda r:hashlib.sha256(r["question"].encode("utf8")).hexdigest());return rows if limit<=0 or limit>=len(rows) else rows[:limit]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--limit",type=int,default=200);ap.add_argument("--max-new",type=int,default=160);a=ap.parse_args()
 m,tok,dev=load_runtime(Path(a.checkpoint),Path(a.tokenizer));rs=subset(read(a.data),a.limit);details=[];ok=0;parsed=0
 for i,r in enumerate(rs,1):
  q="Solve this math problem. Reason briefly, then give the final numeric answer as #### <number>.\n\n"+r["question"]
  text,_=generate(m,tok,[("user",q)],max_new=a.max_new,max_context=1024,temperature=0,top_p=1,repetition_penalty=1.0,device=dev)
  y=target(r["answer"]);p=response(text);hit=(y is not None and p==y);ok+=int(hit);parsed+=int(p is not None)
  details.append({"id":r["id"],"correct":hit,"target":None if y is None else str(y),"predicted":None if p is None else str(p),"text":text})
  if i%25==0:print(json.dumps({"done":i,"correct":ok}),flush=True)
 out={"schema":1,"benchmark":"GSM8K pinned test","cases":len(rs),"correct":ok,"accuracy":ok/len(rs),"parse_fraction":parsed/len(rs),"details":details}
 Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({k:v for k,v in out.items() if k!="details"}))
if __name__=="__main__":main()
