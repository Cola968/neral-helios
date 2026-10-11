from __future__ import annotations
import argparse,json
from pathlib import Path
try:
 from .chat_cli350 import load_runtime,generate
 from .mbpp_safe_runner350 import run
 from .external_bench350_scoring import code_only
except ImportError:
 from chat_cli350 import load_runtime,generate
 from mbpp_safe_runner350 import run
 from external_bench350_scoring import code_only
def read(path):
 rs=[json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]
 if not rs:raise ValueError("empty MBPP JSONL")
 return rs
def subset(rows,limit):
 rows=sorted(rows,key=lambda r:int(r["task_id"]));return rows if limit<=0 or limit>=len(rows) else rows[:limit]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--tokenizer",required=True);ap.add_argument("--data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--limit",type=int,default=100);ap.add_argument("--max-new",type=int,default=220);a=ap.parse_args()
 m,tok,dev=load_runtime(Path(a.checkpoint),Path(a.tokenizer));rs=subset(read(a.data),a.limit);details=[];ok=0
 for i,r in enumerate(rs,1):
  q="Write Python code that solves this task. Return only the code, with no explanation.\n\n"+r["prompt"]
  text,_=generate(m,tok,[("user",q)],max_new=a.max_new,max_context=1024,temperature=0,top_p=1,repetition_penalty=1.0,device=dev)
  code=code_only(text);result=run(code,r.get("test_imports",[]),r["test_list"]);hit=bool(result.get("pass"));ok+=int(hit)
  details.append({"id":r["id"],"task_id":r["task_id"],"pass":hit,"runner_error":result.get("error"),"code":code})
  if i%20==0:print(json.dumps({"done":i,"passed":ok}),flush=True)
 out={"schema":1,"benchmark":"MBPP sanitized pinned test","cases":len(rs),"passed":ok,"pass_at_1":ok/len(rs),"details":details}
 Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8");print(json.dumps({k:v for k,v in out.items() if k!="details"}))
if __name__=="__main__":main()
