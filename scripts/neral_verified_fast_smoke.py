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
    "blind100-logic-02",
    "blind100-logic-06",
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
        if r.get("full_status") not in ("completed","token_limit","inference_error") or r.get("fast_status") not in ("completed","token_limit","inference_error"):
            raise ValueError("Missing outcome status from actual API")
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
        # A failed request has no complete Ollama load telemetry; never
        # reinterpret an unknown count as zero model reloads.
        "full_model_load_events":sum(x["full_model_loads"] for x in observations
                                     if type(x.get("full_model_loads")) is int),
        "fast_model_load_events":sum(x["fast_model_loads"] for x in observations
                                     if type(x.get("fast_model_loads")) is int),
        "full_model_load_missing_cases":sum(x.get("full_model_loads") is None for x in observations),
        "fast_model_load_missing_cases":sum(x.get("fast_model_loads") is None for x in observations),
        "full_token_limit_cases":sum(x["full_status"]=="token_limit" for x in observations),
        "fast_token_limit_cases":sum(x["fast_status"]=="token_limit" for x in observations),
        "full_failure_phases":dict(Counter(x["full_failure_phase"] for x in observations
                    if x.get("full_failure_phase"))),
        "fast_failure_phases":dict(Counter(x["fast_failure_phase"] for x in observations
                    if x.get("fast_failure_phase"))),
        "logic_full_failure_phases":dict(Counter(x["full_failure_phase"] for x in observations
                    if x.get("category")=="logic" and x.get("full_failure_phase"))),
        "logic_fast_failure_phases":dict(Counter(x["fast_failure_phase"] for x in observations
                    if x.get("category")=="logic" and x.get("fast_failure_phase"))),
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
    if result.status_code==503 and data.get("error_code")=="ollama_transport":
        diag=data.get("diagnostic")
        if type(diag) is not dict:
            raise RuntimeError("Transport error lacked valid bounded diagnostics")
        phase=diag.get("inference_phase")
        if phase not in ("qlora_schema_extraction","qlora_tool_proposal",
                         "base_controller_fallback","base_guard_fallback",
                         "base_fast_fallback"):
            raise RuntimeError("Untrusted or missing inference failure phase")
        reason=diag.get("reason")
        if reason=="token_limit":
            if diag.get("done_reason")!="length" or diag.get("http_status")!=200:
                raise RuntimeError("Inconsistent local token-limit diagnostic")
            status="token_limit"
        elif reason in ("timeout","transport_error","http_5xx","done_false",
                         "empty_content","nonstop_finish","invalid_http_payload"):
            status="inference_error"
        else:
            raise RuntimeError("Unsupported transport diagnostic")
        schema_attempts=diag.get("schema_attempts")
        base_attempts=diag.get("base_fallback_attempts")
        if (type(schema_attempts) is not int or not 0<=schema_attempts<=2 or
            type(base_attempts) is not int or not 0<=base_attempts<=3):
            raise RuntimeError("Unsafe or missing scalar controller telemetry")
        if fast and schema_attempts!=0:
            raise RuntimeError("Fast path unexpectedly extracted model schemas")
        return {"latency_ms":elapsed,"status":status,
                "done_reason":diag.get("done_reason"),
                "http_status":diag.get("http_status"),
                "failure_phase":phase,
                "schema_attempts":schema_attempts,
                "schema_rejection_reason":diag.get("extraction_rejection_reason"),
                "base_fallback_attempts":base_attempts,
                "verified":False,"correct":False,
                "model_loads":None,"source":None,"selected":None,
                "both_resident":None,"model":None,"model_origin":None}
    if result.status_code!=200:
        raise RuntimeError("Old-six API unexpected HTTP status: "+str(result.status_code))
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
           "status":"completed","done_reason":"stop","http_status":200,
           "failure_phase":None,
           "schema_attempts":data.get("structured_schema_attempts"),
           "schema_rejection_reason":data.get("extraction_rejection_reason"),
           "base_fallback_attempts":data.get("base_raw_calls"),
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
    for idx,item in enumerate(inputs):
        # Alternate order to reduce "full warms BASE for fast" bias;
        # 6-case exploratory measurements cannot establish production P50.
        if idx%2==0:
            full=_api_one(app,headers,item,False)
            fast=_api_one(app,headers,item,True)
        else:
            fast=_api_one(app,headers,item,True)
            full=_api_one(app,headers,item,False)
        results.append({
            "id":item["id"],"category":item["category"],
            "full_latency_ms":full["latency_ms"],
            "fast_latency_ms":fast["latency_ms"],
            "full_correct":full["correct"],"fast_correct":fast["correct"],
            "full_verified":full["verified"],"fast_verified":fast["verified"],
            "full_model_loads":full["model_loads"],
            "fast_model_loads":fast["model_loads"],
            "full_policy":full["selected"],"fast_policy":fast["selected"],
            "full_status":full["status"],"fast_status":fast["status"],
            "full_done_reason":full["done_reason"],"fast_done_reason":fast["done_reason"],
            "full_failure_phase":full["failure_phase"],"fast_failure_phase":fast["failure_phase"],
            "full_schema_attempts":full["schema_attempts"],
            "fast_schema_attempts":fast["schema_attempts"],
            "full_schema_rejection_reason":full["schema_rejection_reason"],
            "fast_schema_rejection_reason":fast["schema_rejection_reason"],
            "full_base_fallback_attempts":full["base_fallback_attempts"],
            "fast_base_fallback_attempts":fast["base_fallback_attempts"],
            "full_both_resident":full["both_resident"],
            "fast_both_resident":fast["both_resident"],
            "fast_model_origin":fast["model_origin"]})
        print("FAST_OLD6_PROGRESS "+str(len(results))+"/6",flush=True)
    summary=compare_known_cases(results)
    summary["full_and_fast_same_decoder_profile"]=True
    summary["request_order"]="alternating_full_fast_by_case; six-case exploratory known development tasks"
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
