"""Fetch and convert the Apache-2.0 OpenAssistant/oasst1 ready-message export.

This utility is optional and only prepares local Stage-D source files. It downloads no model
weights and calls no external model/API. Only reviewed, non-deleted, non-synthetic human
prompter -> assistant pairs in German/English are retained. The best ranked assistant per
prompt is selected deterministically.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path

UPSTREAM_URL="https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz"
DATASET_URL="https://huggingface.co/datasets/OpenAssistant/oasst1"
LICENSE_URL="https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE"
USER_AGENT="NeralDataBuilder/1.0 (+local reproducible dataset preparation)"


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def download(url: str,path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT})
    with urllib.request.urlopen(req,timeout=120) as r,path.open("wb") as f:
        while True:
            chunk=r.read(1024*1024)
            if not chunk:break
            f.write(chunk)


def reviewed_human(row):
    return row.get("deleted") is not True and row.get("synthetic") is not True and row.get("review_result") is not False


def infer_domain(prompt: str,answer: str):
    text=(prompt+" "+answer[:500]).casefold()
    if re.search(r"\b(python|javascript|java|kotlin|function|code|programm|debug|sql|html|css)\b",text):return "code"
    if re.search(r"\b(calculate|equation|fraction|percent|algebra|geometry|mathe|berechne|gleichung|bruch|prozent)\b",text):return "math"
    if re.search(r"\b(physics|astronom|planet|orbit|gravity|relativity|quantum|physik|chemie|biolog|science)\b",text):return "science"
    if re.search(r"\b(aircraft|airline|airport|aviation|train|railway|transport|flugzeug|flughafen|luftfahrt|bahn)\b",text):return "focus"
    if re.search(r"\b(translate|translation|übersetz|uebersetz)\b",text):return "translation"
    if re.search(r"\b(summarize|summary|zusammenfass)\b",text):return "summary"
    if re.search(r"\b(reason|logic|deduc|infer|warum|begründe|begrunde|logik)\b",text):return "reasoning"
    if re.search(r"\b(explain|describe|what is|how does|erklär|erklaer|beschreib|was ist|wie funktioniert)\b",text):return "explanation"
    return "chat"


def choose_pairs(rows,max_en,max_de):
    messages={r.get("message_id"):r for r in rows if isinstance(r.get("message_id"),str)}
    grouped={}
    for row in rows:
        if row.get("role")!="assistant" or row.get("lang") not in {"de","en"} or not reviewed_human(row):continue
        parent=messages.get(row.get("parent_id"))
        if not parent or parent.get("role")!="prompter" or parent.get("lang")!=row.get("lang") or not reviewed_human(parent):continue
        prompt=parent.get("text");answer=row.get("text")
        if not isinstance(prompt,str) or not isinstance(answer,str):continue
        rank=row.get("rank");rank=10**9 if rank is None else int(rank)
        key=(row["lang"],parent["message_id"]);candidate=(rank,row["message_id"],prompt,answer)
        if key not in grouped or candidate[:2]<grouped[key][:2]:grouped[key]=candidate
    out={"de":[],"en":[]}
    for (lang,parent_id),(rank,message_id,prompt,answer) in sorted(grouped.items()):
        out[lang].append({"id":message_id,"prompt":prompt,"answer":answer,"domain":infer_domain(prompt,answer),"upstream_parent_id":parent_id,"upstream_rank":None if rank==10**9 else rank})
    out["en"]=out["en"][:max_en];out["de"]=out["de"][:max_de]
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="sft50_sources");ap.add_argument("--max-en",type=int,default=14000);ap.add_argument("--max-de",type=int,default=5000);ap.add_argument("--force-download",action="store_true");args=ap.parse_args()
    if min(args.max_en,args.max_de)<=0:raise ValueError("positive language caps required")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);raw=out/"oasst1_ready.messages.jsonl.gz"
    if args.force_download or not raw.is_file():download(UPSTREAM_URL,raw)
    upstream_sha=sha256_file(raw);rows=[]
    with gzip.open(raw,"rt",encoding="utf8") as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    pairs=choose_pairs(rows,args.max_en,args.max_de)
    counts={lang:len(v) for lang,v in pairs.items()}
    if sum(counts.values())<3000:raise ValueError(f"Too few usable OASST1 pairs: {counts}")
    sources=[];safe_train={lang:max(1,int(n*.75)) for lang,n in counts.items() if n}
    safe_total=sum(safe_train.values())
    for lang in ("de","en"):
        if not pairs[lang]:continue
        path=out/f"oasst1_{lang}.jsonl";path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in pairs[lang]),encoding="utf8")
        sources.append({
            "id":f"oasst1-human-{lang}","file":path.name,"sha256":sha256_file(path),"upstream_sha256":upstream_sha,
            "license":"Apache-2.0","license_url":LICENSE_URL,"provenance_url":DATASET_URL,
            "language":lang,"domain":"chat","allow_row_domains":True,"target_fraction":safe_train[lang]/safe_total,
        })
    manifest={"version":1,"recommended_train_examples":safe_total,"upstream_download_url":UPSTREAM_URL,"upstream_sha256":upstream_sha,"filters":"reviewed non-deleted non-synthetic human prompter->assistant; deterministic best rank per prompt; de/en only","sources":sources}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({"download_sha256":upstream_sha,"pairs":counts,"recommended_train_examples":safe_total,"domains":{lang:{d:sum(r['domain']==d for r in pairs[lang]) for d in sorted({x['domain'] for x in pairs[lang]})} for lang in pairs}},indent=2,ensure_ascii=False))


if __name__=="__main__":main()
