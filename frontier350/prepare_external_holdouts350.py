from __future__ import annotations
import argparse,hashlib,json,urllib.request
from pathlib import Path
SOURCES={
 "gsm8k":{"repo":"openai/gsm8k","revision":"740312add88f781978c0658806c59bc2815b9866","license":"MIT","path":"main/test-00000-of-00001.parquet","sha256":"ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"},
 "mbpp":{"repo":"google-research-datasets/mbpp","revision":"4bb6404fdc6cacfda99d4ac4205087b89d32030c","license":"CC-BY-4.0","path":"sanitized/test-00000-of-00001.parquet","sha256":"e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850"}
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
def download(spec,path):
 url=f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
 req=urllib.request.Request(url,headers={"User-Agent":"NeralExternalEval/1.0"})
 with urllib.request.urlopen(req,timeout=180) as r,Path(path).open("wb") as o:
  for c in iter(lambda:r.read(1024*1024),b""):o.write(c)
 return url
def write_jsonl(path,rows):
 Path(path).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8");return sha(path)
def prepare(out_root:Path):
 try:import pyarrow.parquet as pq
 except ImportError as e:raise RuntimeError("pyarrow is required only for external benchmark preparation") from e
 out_root=Path(out_root)
 if out_root.exists() and any(out_root.iterdir()):raise FileExistsError("external benchmark output must be fresh")
 raw=out_root/"raw";raw.mkdir(parents=True)
 manifest={"schema":1,"purpose":"evaluation-only external holdouts; forbidden from Neral training","sources":{}}
 for name,spec in SOURCES.items():
  p=raw/f"{name}.parquet";url=download(spec,p);actual=sha(p)
  if actual!=spec["sha256"]:raise ValueError(f"{name} pinned parquet SHA mismatch: {actual}")
  table=pq.read_table(p);rows=table.to_pylist()
  if name=="gsm8k":
   if set(table.column_names)!={"question","answer"} or len(rows)!=1319:raise ValueError("unexpected GSM8K schema/count")
   clean=[{"id":f"gsm8k-test-{i}","question":r["question"],"answer":r["answer"]} for i,r in enumerate(rows)]
  else:
   expected={"source_file","task_id","prompt","code","test_imports","test_list"}
   if set(table.column_names)!=expected or len(rows)!=257:raise ValueError("unexpected MBPP schema/count")
   clean=[{"id":f"mbpp-test-{r['task_id']}","task_id":int(r["task_id"]),"prompt":r["prompt"],"test_imports":r["test_imports"],"test_list":r["test_list"]} for r in rows]
  jp=out_root/f"{name}_test.jsonl";outsha=write_jsonl(jp,clean)
  manifest["sources"][name]={**spec,"url":url,"rows":len(clean),"jsonl_sha256":outsha}
 rawj=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode();manifest["manifest_sha256"]=hashlib.sha256(rawj).hexdigest()
 (out_root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8");return manifest
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args();print(json.dumps(prepare(Path(a.out)),ensure_ascii=False))
if __name__=="__main__":main()
