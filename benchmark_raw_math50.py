"""Evaluate Neral weights on frozen RawMath v1 with no SOLVE or tools."""
from __future__ import annotations
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from experiment import load_model
from raw_math50_data import load_benchmark, validate_frozen_suite
from tokenizer import ByteTokenizer

TOK=ByteTokenizer()

def prompt_ids(row):
    return TOK.encode("Question: "+row["prompt"].strip()+"\nAnswer:",bos=True)

def answer_nll(model,row,device):
    prefix=prompt_ids(row);answer=TOK.encode(" "+row["answer"].strip()+"\n",eos=True);ids=prefix+answer
    x=torch.tensor(ids[:-1],device=device).unsqueeze(0);targets=torch.tensor(([-100]*len(prefix)+answer)[1:],device=device).unsqueeze(0)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16): logits,_=model(x)
    loss=F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100,reduction="sum")
    n=int((targets!=-100).sum());return float(loss),n

def normalize_output(s):
    # Strict exact match permits surrounding whitespace only; explanations are failures.
    return s.strip()

def aggregate(rows):
    n=len(rows);correct=sum(r["exact"] for r in rows);eos=sum(r["eos"] for r in rows);utf8=sum(r["valid_utf8"] for r in rows)
    nll=sum(r["answer_nll_sum"] for r in rows);tok=sum(r["answer_tokens"] for r in rows)
    return {"tasks":n,"exact":correct,"exact_pct":100*correct/n if n else None,"eos_pct":100*eos/n if n else None,"valid_utf8_pct":100*utf8/n if n else None,"answer_nll":nll/tok if tok else None,"answer_bits_per_token":nll/tok/math.log(2) if tok else None,"mean_seconds":sum(r["seconds"] for r in rows)/n if n else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("checkpoint");ap.add_argument("--out",required=True);ap.add_argument("--max-new-tokens",type=int,default=32);args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for comparable RawMath benchmark")
    device=torch.device("cuda");model,_=load_model(args.checkpoint,device);model.eval();suite=validate_frozen_suite();rows=[]
    for item in load_benchmark():
        ids=prompt_ids(item);x=torch.tensor(ids,device=device).unsqueeze(0);torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode(): out=model.generate(x,max_new_tokens=args.max_new_tokens,temperature=0,eos_id=TOK.eos_id)
        torch.cuda.synchronize();dt=time.perf_counter()-t;new=out[0,len(ids):].tolist();eos=TOK.eos_id in new
        if eos:new=new[:new.index(TOK.eos_id)]
        raw=TOK.decode(new,skip_special=True);valid="\ufffd" not in raw;nll,ntok=answer_nll(model,item,device)
        rows.append({"id":item["id"],"category":item["category"],"language":item["language"],"distribution":item["distribution"],"prompt":item["prompt"],"expected":item["answer"],"output":raw,"normalized_output":normalize_output(raw),"exact":normalize_output(raw)==item["answer"],"eos":eos,"valid_utf8":valid,"seconds":dt,"answer_nll_sum":nll,"answer_tokens":ntok})
    by_category={};by_language={};by_distribution={}
    for key in sorted({r["category"] for r in rows}):by_category[key]=aggregate([r for r in rows if r["category"]==key])
    for key in ("de","en"):by_language[key]=aggregate([r for r in rows if r["language"]==key])
    for key in sorted({r["distribution"] for r in rows}):by_distribution[key]=aggregate([r for r in rows if r["distribution"]==key])
    report={"definition":"Frozen RawMath v1. Greedy strict exact-match after surrounding-whitespace trim only. No SOLVE, no tools, no answer extraction from prose.","suite":suite,"checkpoint":str(Path(args.checkpoint).resolve()),"parameters":model.parameter_report()["total"],"generation":{"temperature":0,"max_new_tokens":args.max_new_tokens},"solve_used":False,"tool_use":False,"overall":aggregate(rows),"by_category":by_category,"by_language":by_language,"by_distribution":by_distribution,"results":rows}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8");print(json.dumps({"out":args.out,"overall":report["overall"],"by_category":by_category},ensure_ascii=False))

if __name__=="__main__":main()
