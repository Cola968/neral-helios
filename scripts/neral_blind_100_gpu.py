"""Real local RTX prospective 100-task Flask-API blind comparison.

Control: exact pinned, untrained Qwen3.5-9B; candidate: productive THINK
dual-routing with genuine PR537 guard, QLoRA tool model, untuned raw fallback.
Persist complete synthetic questions, expected integer, BOTH raw responses, and
only local Ollama telemetry. No user chat prompts or private files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from neral_agent_autonomy import OllamaLocalBackend
from neral_identity import build_identity_prompt
from neral_preview_server import create_preview_app
from neral_dual_routing import _ollama_chat, sample_device_mib
from scripts.neral_generalization_100_cases import build as old100
from scripts.neral_generalization_100_run import (
    BASE_TAG,TUNED_TAG,BASE_DIGEST,TUNED_DIGEST,require_exact_models,
    parse_answer as parse_exact, gpu_offload_mib,
)
from scripts.neral_blind_100_cases import build,exclusion,serialize

AUTH="NERAL_PROD_DUAL_BLIND100_EXPLICIT"
API_ORIGIN="http://127.0.0.1:9690"
EXPECTED_MODEL={"base":BASE_TAG,"tool":TUNED_TAG}
TOTAL=100

def parse_answer(text):
    return parse_exact(text)

def score_pairs(cases,rows):
    if len(cases)!=100 or len(rows)!=100:
        raise ValueError("Exactly 100 complete prospective observations required")
    by_id={}
    for item in rows:
        ident=item.get("id")
        if ident in by_id or type(item.get("base_correct")) is not bool or type(item.get("dual_correct")) is not bool:
            raise ValueError("Missing, duplicate, or invalid outcome")
        by_id[ident]=item
    if set(by_id)!={x["id"] for x in cases}:
        raise ValueError("Case IDs differ from sealed prospective test")
    grouped={};wins=losses=0
    for category in ("mathematics","logic","programming","mixed"):
        selected=[x for x in cases if x["category"]==category]
        if len(selected)!=25:raise ValueError("Expected 25 in category")
        control=sum(by_id[x["id"]]["base_correct"] for x in selected)
        dual=sum(by_id[x["id"]]["dual_correct"] for x in selected)
        gained=sum(not by_id[x["id"]]["base_correct"] and by_id[x["id"]]["dual_correct"] for x in selected)
        lost=sum(by_id[x["id"]]["base_correct"] and not by_id[x["id"]]["dual_correct"] for x in selected)
        grouped[category]={"base":control,"dual":dual,
                           "gained":gained,"lost":lost,"total":25}
        wins+=gained;losses+=lost
    n=wins+losses
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.0
    # The criterion is strict, not a guarantee of 0% future regressions.
    passed=losses==0 and wins>=8
    return {"total":100,
            "base_correct":sum(x["base_correct"] for x in rows),
            "dual_correct":sum(x["dual_correct"] for x in rows),
            "gained":wins,"lost":losses,"net":wins-losses,
            "by_category":grouped,"exact_mcnemar_two_sided_p":p,
            "promotion_gate_passed":passed}

def make_api():
    # Same production code / CSRF / busy lock as the regular NERAL API;
    # test_client prevents binding a second local server while the user's
    # normal preview is already running.
    low=OllamaLocalBackend("qwen3:8b")
    high=OllamaLocalBackend("qwen3.5:9b",
        inference_profile="measured_nonthinking_2048")
    app=create_preview_app(low,experimental_high_backend=high)
    client=app.test_client()
    client.get("/",base_url=API_ORIGIN)
    info=client.get("/api/neral/preview/session",base_url=API_ORIGIN)
    if info.status_code!=200:
        raise RuntimeError("Local NERAL preview session was unavailable")
    csrf=info.json.get("csrf")
    if type(csrf) is not str:
        raise RuntimeError("Local CSRF token missing")
    return client,{"X-Neral-CSRF":csrf}

def pair_case(case,client,headers):
    messages=[
       {"role":"system","content":build_identity_prompt(
           "web","local_foundation_not_neral_adapter")},
       {"role":"user","content":case["question"]}]
    raw0=dual0=None
    before=sample_device_mib()
    start=time.monotonic()
    # Both arms use precisely the same user question, identity system,
    # temperature 0, 2048 tokens, 4096 context, seed 0, non-thinking.
    control,meta=_ollama_chat(BASE_TAG,messages,0)
    control_seconds=round(time.monotonic()-start,3)
    if gpu_offload_mib(BASE_TAG)<4096:
        raise RuntimeError("Control model did not use the RTX GPU")
    start=time.monotonic()
    response=client.post("/api/neral/chat/think",
         json={"prompt":case["question"],"experimental_tir":True,
               "conservative_tir":True,"dual_routing":True},
         headers=headers,base_url=API_ORIGIN)
    dual_seconds=round(time.monotonic()-start,3)
    if response.status_code!=200:
        raise RuntimeError("Production dual-route API HTTP "+str(response.status_code))
    result=response.json
    dual=result.get("text")
    if type(dual) is not str or not dual.strip():
        raise RuntimeError("Product API returned no complete response")
    if result.get("tool_model")!=TUNED_TAG or result.get("fallback_model")!=BASE_TAG:
        raise RuntimeError("Productive API used a different model pairing")
    if result.get("model_digests")!={BASE_TAG:BASE_DIGEST,TUNED_TAG:TUNED_DIGEST}:
        raise RuntimeError("Productive model digest differs from pinned pair")
    expected=case["gold_integer"]
    end=sample_device_mib()
    return {"id":case["id"],"category":case["category"],
            "family":case["family"],"question_sha256":case["question_sha256"],
            "base_correct":parse_exact(control)==expected,
            "dual_correct":parse_exact(dual)==expected,
            "policy":result.get("selection_policy","unknown"),
            "verified":result.get("verified") is True,
            "base_seconds":control_seconds,"dual_seconds":dual_seconds,
            "base_integer_format":parse_exact(control) is not None,
            "dual_integer_format":parse_exact(dual) is not None,
            "both_resident":bool(result.get("ollama_resident_after",{}).get("both_resident")),
            "gpu_peak_mib":result.get("gpu_device_peak_mib"),
            "gpu_snapshot_before_mib":before,"gpu_snapshot_after_mib":end,
            "base_load_ms":meta.get("load_ms"),
            "dual_model_load_ms":[z.get("load_ms") for z in result.get("inference_spans",[])],
            "dual_model_spans":result.get("inference_spans",[]),
            "dual_base_calls":result.get("base_raw_calls"),
            "dual_tool_planning_calls":result.get("tool_planning_calls"),
            "dual_schema_calls":result.get("structured_schema_attempts"),
            "raw":{"question":case["question"],"gold_integer":expected,
                   "base_reply":control,"dual_reply":dual}}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--authorize",required=True)
    parser.add_argument("--train",required=True,type=Path)
    parser.add_argument("--holdout",required=True,type=Path)
    parser.add_argument("--corpus",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--pilot",action="store_true")
    args=parser.parse_args()
    if args.authorize!=AUTH or os.environ.get("NERAL_EXPERIMENTAL_TIR")!="1" or os.environ.get("NERAL_EXPERIMENTAL_DUAL_ROUTE")!="1":
        raise RuntimeError("Product dual blind gate requires manual local authorization")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Never overwrite blind results")
    tags=require_exact_models()
    from scripts.neral_generalization_100_run import require_exact_guard
    guard_hash=require_exact_guard()
    client,headers=make_api()
    if args.pilot:
        # NEVER tune to or score a future blind case. Pilot only on OLD 100.
        sample=old100()[50]
        task={"question":sample["question"],"gold_integer":sample["expected_integer"],
              "category":sample["category"],"id":sample["id"],
              "family":sample["subfamily"],"question_sha256":sample["question_sha256"]}
        tested=pair_case(task,client,headers)
        data={"status":"OLD_100_DEV_API_PILOT_PASS",
              "id":task["id"],"policy":tested["policy"],
              "tool_model":TUNED_TAG,"raw_fallback_model":BASE_TAG,
              "gpu_peak_mib":tested["gpu_peak_mib"],
              "both_resident":tested["both_resident"],
              "no_training":True}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf8",newline="\n") as f:
            f.write(json.dumps(data,sort_keys=True)+"\n")
        print("PRODUCTION_DUAL_API_PILOT "+json.dumps(data,sort_keys=True),flush=True)
        return 0
    if args.corpus is None or args.manifest is None:
        raise RuntimeError("Full blind run requires two frozen evidence files")
    cases=build()
    receipt=exclusion(cases,args.train,args.holdout)
    frozen=args.corpus.read_bytes()
    if frozen!=serialize(cases) or hashlib.sha256(frozen).hexdigest()!=receipt["blind100_sha256"]:
        raise RuntimeError("The unseen prospective corpus was modified")
    if json.loads(args.manifest.read_text(encoding="utf8"))!=receipt:
        raise RuntimeError("The prospective blind receipt was modified")
    progress=args.out.with_suffix(".progress.jsonl")
    raw_file=args.out.with_suffix(".full_raw.jsonl")
    if progress.exists() or raw_file.exists():
        raise FileExistsError("Previous partial blind run exists")
    rows=[]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with progress.open("x",encoding="utf8",newline="\n") as stat,raw_file.open("x",encoding="utf8",newline="\n") as raws:
        for i,task in enumerate(cases):
            observed=pair_case(task,client,headers)
            raw=observed.pop("raw")
            # Full trace ONLY for these publicly generated synthetic cases,
            # never for the user's private conversation or local data.
            raws.write(json.dumps({"id":observed["id"],**raw},
                 ensure_ascii=False,sort_keys=True)+"\n")
            raws.flush()
            stat.write(json.dumps(observed,ensure_ascii=False,sort_keys=True)+"\n")
            stat.flush()
            rows.append(observed)
            print("BLIND100_PROGRESS "+str(i+1)+"/100",flush=True)
    result=score_pairs(cases,rows)
    latencies=sorted(x["dual_seconds"] for x in rows)
    base_latencies=sorted(x["base_seconds"] for x in rows)
    result.update({
       "status":"PROSPECTIVE_100_PRODUCTION_DUAL_REAL_RTX",
       "model_digests":tags,"guard_git_blob":guard_hash,
       "blind100_sha256":receipt["blind100_sha256"],
       "train_sha256":receipt["train_sha256"],
       "holdout24_sha256":receipt["holdout24_sha256"],
       "control":"same checkpoint, same direct system prompt + decoder settings",
       "configuration":"production Flask THINK dual_routing=true, conservative_tir=true",
       "both_models_resident_observed_cases":sum(bool(x["both_resident"]) for x in rows),
       "gpu_peak_observed_mib":max((x["gpu_peak_mib"] or 0 for x in rows),default=0),
       "model_load_events_over_1s":sum(
            load is not None and load>1000 for x in rows
            for load in x["dual_model_load_ms"]),
       "model_swap_latency_ms_per_call":[
           x["dual_model_load_ms"] for x in rows],
       "base_p50_s":statistics.median(base_latencies),
       "base_p95_s":base_latencies[94],
       "dual_p50_s":statistics.median(latencies),
       "dual_p95_s":latencies[94],
       "selection_counts":dict(Counter(x["policy"] for x in rows)),
       "no_training":True,
       "full_raw_saved_local":True,
       "limits":"Public new synthetic corpus; 0 observed losses does not prove zero future regressions; nonthinking integer-only scoring."
    })
    with args.out.open("x",encoding="utf8",newline="\n") as f:
        f.write(json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n")
    print("PROSPECTIVE_BLIND100_FINAL "+json.dumps({k:v for k,v in result.items()
            if k not in ("model_swap_latency_ms_per_call",)},
            ensure_ascii=False,sort_keys=True),flush=True)
    return 0
if __name__=="__main__":
    raise SystemExit(main())
