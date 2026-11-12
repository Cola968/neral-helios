"""GPU smoke on SIX ALREADY KNOWN OLD TASKS, never new 3feca... holdout.

Compare full original Dual-Routing with explicitly requested independent
source-first fast fallback, both using the same evaluation format and models.
No training, no production-mode auto-routing and no blind score promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import time

from scripts.neral_blind_100_cases import build
from scripts.neral_blind_100_gpu import make_api
from scripts.neral_formatted_100_gpu import verify_original_100
from scripts.neral_generalization_100_run import require_exact_models,require_exact_guard
from neral_answer_contract import parse_terminal_final_integer

AUTH="NERAL_VERIFIED_FAST_OLD6_GPU_SMOKE_ONLY"
SELECTED_DEV_IDS=(
    "blind100-mathematics-01",
    "blind100-logic-01",
    "blind100-logic-05",
    "blind100-programming-01",
    "blind100-programming-03",
    "blind100-mixed-01",
)

def select_unchanged_cases():
    cases=build()
    verify_original_100(cases)
    mapped={case["id"]:case for case in cases}
    if len(mapped)!=100 or len(set(SELECTED_DEV_IDS))!=len(SELECTED_DEV_IDS):
        raise ValueError("Old development corpus invalid")
    return [mapped[i] for i in SELECTED_DEV_IDS]

def compare_known_cases(observations):
    if (len(observations)!=len(SELECTED_DEV_IDS)
            or [r.get("id") for r in observations]!=list(SELECTED_DEV_IDS)):
        raise ValueError("Smoke results must cover exact six previously known tasks")
    for r in observations:
        for value in ("full_latency_ms","fast_latency_ms"):
            if type(r.get(value)) not in (float,int) or r[value]<0:
                raise ValueError("Incomplete measured latency")
        if any(type(r.get(q)) is not bool for q in
               ("full_correct","fast_correct","full_verified","fast_verified")):
            raise ValueError("Missing paired known-task result")
    return {
        "count":len(observations),
        "blind_test":False,
        "promotion_gate_eligible":False,
        "full_p50_ms":round(statistics.median(x["full_latency_ms"] for x in observations),2),
        "fast_p50_ms":round(statistics.median(x["fast_latency_ms"] for x in observations),2),
        "median_latency_savings_ms":round(statistics.median(x["full_latency_ms"] for x in observations)-
            statistics.median(x["fast_latency_ms"] for x in observations),2),
        "full_correct":sum(x["full_correct"] for x in observations),
        "fast_correct":sum(x["fast_correct"] for x in observations),
        "paired_answer_disagreements":sum(x["full_correct"]!=x["fast_correct"] for x in observations),
        "full_verified":sum(x["full_verified"] for x in observations),
        "fast_verified":sum(x["fast_verified"] for x in observations),
        "baseline_model_only_responses":sum(not x["fast_verified"] for x in observations),
        "full_model_load_events":sum(x["full_model_loads"] for x in observations),
        "fast_model_load_events":sum(x["fast_model_loads"] for x in observations),
    }

def _api_one(client,headers,item,fast):
    opts={"prompt":item["question"],"experimental_tir":True,
          "conservative_tir":True,"dual_routing":True,
          "evaluation_final_answer":True}
    if fast:opts["fast_verified_only"]=True
    started=time.monotonic()
    result=client.post("/api/neral/chat/think",json=opts,headers=headers,
                       base_url="http://127.0.0.1:9690")
    elapsed=round((time.monotonic()-started)*1000,2)
    data=result.json if type(result.json) is dict else {}
    if result.status_code!=200:
        raise RuntimeError("Old-six API request did not finish: "+str(result.status_code))
    if type(data.get("text")) is not str:
        raise RuntimeError("Invalid finished old-six answer")
    spans=data.get("inference_spans")
    if type(spans) is not list:
        raise RuntimeError("Missing old-six inference telemetry")
    count=sum(type(s) is dict and type(s.get("load_ms")) in (int,float)
              and s["load_ms"]>1000 for s in spans)
    if fast and (data.get("tool_planning_calls")!=0
                 or data.get("structured_schema_attempts")!=0):
        raise RuntimeError("Fast path illegally used QLoRA tool decision")
    return {"latency_ms":elapsed,
           "verified":data.get("verified") is True,
           "correct":parse_terminal_final_integer(data["text"])==item["gold_integer"],
           "model_loads":count,
           "source":data.get("response_source"),
           "selected":data.get("selection_policy"),
           "both_resident":data.get("ollama_resident_after",{}).get("both_resident"),
           "model":data.get("model"),"model_origin":data.get("model_origin")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.authorize!=AUTH or any(os.environ.get(key)!="1" for key in (
        "NERAL_EXPERIMENTAL_TIR","NERAL_EXPERIMENTAL_DUAL_ROUTE",
        "NERAL_EXPERIMENTAL_FAST_VERIFIED_ONLY","NERAL_FORMATTED_EVAL")):
        raise RuntimeError("Old six-case GPU smoke not manually enabled")
    if args.out.exists():raise FileExistsError("Do not overwrite old smoke")
    models=require_exact_models()
    guard=require_exact_guard()
    inputs=select_unchanged_cases()
    app,headers=make_api()
    results=[]
    for item in inputs:
        full=_api_one(app,headers,item,False)
        fast=_api_one(app,headers,item,True)
        results.append({
            "id":item["id"],"category":item["category"],
            "full_latency_ms":full["latency_ms"],
            "fast_latency_ms":fast["latency_ms"],
            "full_correct":full["correct"],"fast_correct":fast["correct"],
            "full_verified":full["verified"],"fast_verified":fast["verified"],
            "full_model_loads":full["model_loads"],
            "fast_model_loads":fast["model_loads"],
            "full_policy":full["selected"],"fast_policy":fast["selected"],
            "full_both_resident":full["both_resident"],
            "fast_both_resident":fast["both_resident"],
            "fast_model_origin":fast["model_origin"]})
        print("FAST_OLD6_PROGRESS "+str(len(results))+"/6",flush=True)
    summary=compare_known_cases(results)
    summary["full_and_fast_same_decoder_profile"]=True
    summary["no_training"]=True
    summary["models"]=models
    summary["guard_hash"]=guard
    summary["original_case_sha256"]=verify_original_100(build())
    summary["cases"]=results
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf8",newline="\n") as f:
        json.dump(summary,f,ensure_ascii=False,sort_keys=True,indent=2)
        f.write("\n")
    print("FAST_OLD6_SUMMARY "+json.dumps(
        {k:v for k,v in summary.items() if k!="cases"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
