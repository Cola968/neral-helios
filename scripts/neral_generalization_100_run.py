"""Exact paired 100-case RTX 4070 generalization gate; NEVER trains.

Arm A: matched quantized, untrained Qwen3.5 9B, raw Chat.
Arm B: QLoRA 9B with actual PR537 conservative choice around experimental TIR.
Inputs/answers never leave the local PC through model calls; output only scalar
per-case statuses, model digests, timestamps, token metrics and summary.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from neral_tir_experimental import run_experimental_tir
from neral_tir_regression_guard import choose_guarded_tir_reply
from scripts.neral_generalization_100_cases import build, disjoint
from scripts.neral_qlora_holdout_live import request, query_structured, gpu_offload_mib

SEED = 202611121004
AUTH = "NERAL_GENERALIZATION_100_GPU_EXPLICIT"
BASE_TAG = "qwen35-9b-qlora-control-20261112:latest"
TUNED_TAG = "neral-qwen35-qlora-v1:latest"
BASE_DIGEST = "ed6a74343db547257a9a78150d647f33fbf25db93d87b8e3d80273b116fa7bb9"
TUNED_DIGEST = "9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8"
GUARD_GIT_BLOB = "b87c594ee54a20b2209b6bf176bc0aacfe256ce8"
BASE_OPTIONS = {"temperature":0,"num_predict":2048,"num_ctx":4096}
SYSTEM = "You are a precise local assistant. Follow the question. Return ONLY ONE INTEGER, no explanation."
VALID_POLICIES = frozenset({"independently_verified_tir","explicit_safe_refusal",
    "raw_qwen_already_selected","raw_qwen_due_to_unverified_tir"})

def require_exact_models()->dict:
    entries=request("GET","/api/tags").get("models")
    if type(entries) is not list:
        raise RuntimeError("No local Ollama model tags")
    indexed={x.get("name"):x.get("digest") for x in entries if type(x) is dict}
    for model,digest in ((BASE_TAG,BASE_DIGEST),(TUNED_TAG,TUNED_DIGEST)):
        if indexed.get(model)!=digest:
            raise RuntimeError("Unexpected local Ollama model digest: "+model)
    return {BASE_TAG:BASE_DIGEST,TUNED_TAG:TUNED_DIGEST}

def require_exact_guard()->str:
    root=Path(__file__).resolve().parents[1]
    process=subprocess.run(["git","hash-object","neral_tir_regression_guard.py"],
                           cwd=root,capture_output=True,text=True,timeout=10)
    if process.returncode!=0 or process.stdout.strip()!=GUARD_GIT_BLOB:
        raise RuntimeError("PR537 guard file is modified, do not benchmark")
    return GUARD_GIT_BLOB

def raw_chat(model:str,messages:list[dict],seed:int)->str:
    if model not in (BASE_TAG,TUNED_TAG) or type(seed) is not int:
        raise ValueError("Unexpected model/seed")
    started=time.monotonic()
    packet={"model":model,"messages":messages,"stream":False,
       "think":False,"keep_alive":"5m","options":{**BASE_OPTIONS,"seed":seed}}
    data=request("POST","/api/chat",packet,timeout=180)
    if data.get("done") is not True or data.get("done_reason")!="stop":
        raise RuntimeError("Raw model ended with an incomplete generation")
    reply=data.get("message")
    text=reply.get("content") if type(reply) is dict else None
    if type(text) is not str or not text.strip() or len(text.encode("utf8"))>32_768:
        raise RuntimeError("Invalid raw answer")
    return text

def parse_answer(raw:object)->int|None:
    if type(raw) is not str:
        return None
    # Fail closed: never hunt arbitrary prose for a plausible number.
    match=re.fullmatch(r"\s*(?:(?:FINAL_ANSWER|NERAL_FINAL_TEXT):\s*)?(-?\d{1,10})\s*",raw)
    return int(match.group(1)) if match else None

def control_case(case:dict,seed:int)->dict:
    started=time.monotonic()
    try:
        answer=raw_chat(BASE_TAG,[
           {"role":"system","content":SYSTEM},
           {"role":"user","content":case["question"]}],seed)
        parsed=parse_answer(answer)
        status="completed"
    except (OSError,RuntimeError,ValueError,TimeoutError):
        parsed=None;status="inference_error"
    return {"id":case["id"],"category":case["category"],
       "correct":parsed==case["expected_integer"],
       "integer_output":parsed is not None,
       "status":status,"seconds":round(time.monotonic()-started,3),
       "arm":"raw_control"}

def guarded_case(case:dict,seed:int)->dict:
    started=time.monotonic()
    messages=[{"role":"system","content":SYSTEM},
              {"role":"user","content":case["question"]}]
    extraction_turn=0
    def respond(turns):
        return raw_chat(TUNED_TAG,turns,seed)
    def extract(turns):
        nonlocal extraction_turn
        extraction_turn+=1
        if extraction_turn>2:
            raise RuntimeError("More than one model-authored solver repair")
        return query_structured(TUNED_TAG,turns,seed=seed,
                                repair=(extraction_turn==2))
    status="completed"
    selection="error"
    source="none"
    verified=False
    triggered=0
    try:
        proposal=run_experimental_tir(messages,respond,extract_constraints=extract)
        report=choose_guarded_tir_reply(proposal,messages,respond)
        selection=report.get("selection_policy","error")
        if selection not in VALID_POLICIES:
            raise RuntimeError("Guard bypass or unrecognized selection")
        source=report.get("response_source","none")
        verified=report.get("verified") is True
        triggered=report.get("tool_calls_triggered",0)
        parsed=parse_answer(report.get("text"))
    except (OSError,RuntimeError,ValueError,TimeoutError):
        parsed=None;status="inference_or_guard_error"
    return {"id":case["id"],"category":case["category"],
       "correct":parsed==case["expected_integer"],
       "integer_output":parsed is not None,
       "status":status,"selection_policy":selection,
       "response_source":source if source in {
           "raw_qwen","verified_tir","deterministic_safety_refusal"} else "other",
       "verified":verified,
       "tool_calls_triggered":triggered if type(triggered) is int else None,
       "seconds":round(time.monotonic()-started,3),"arm":"qlora_guarded_pr537"}

def paired_score(cases:list[dict],controls:list[dict],guarded:list[dict])->dict:
    if (len(cases),len(controls),len(guarded))!=(100,100,100):
        raise ValueError("Exactly 100 pairs are required")
    by_control={x["id"]:x for x in controls}
    by_guarded={x["id"]:x for x in guarded}
    identifiers={c["id"] for c in cases}
    if len(by_control)!=100 or len(by_guarded)!=100 or set(by_control)!=identifiers or set(by_guarded)!=identifiers:
        raise ValueError("Missing/duplicate pair IDs")
    breakdown={}; gained=lost=0
    rows=[]
    for category in ("mathematics","logic","programming","mixed"):
        members=[c for c in cases if c["category"]==category]
        if len(members)!=25:raise ValueError("Expected 25 cases in each group")
        cs=sum(by_control[x["id"]]["correct"] is True for x in members)
        ts=sum(by_guarded[x["id"]]["correct"] is True for x in members)
        gw=sum(by_control[x["id"]]["correct"] is False and
               by_guarded[x["id"]]["correct"] is True for x in members)
        ls=sum(by_control[x["id"]]["correct"] is True and
               by_guarded[x["id"]]["correct"] is False for x in members)
        gained+=gw;lost+=ls
        breakdown[category]={"base_correct":cs,"guarded_correct":ts,
                             "gained":gw,"lost":ls,"total":25}
    for case in cases:
        c=by_control[case["id"]];t=by_guarded[case["id"]]
        rows.append({"id":case["id"],"category":case["category"],
                     "base_correct":c["correct"],"guarded_correct":t["correct"],
                     "base_integer":c["integer_output"],"guarded_integer":t["integer_output"],
                     "guard_selection":t["selection_policy"],
                     "guard_verified":t["verified"],
                     "base_status":c["status"],"guard_status":t["status"],
                     "base_seconds":c["seconds"],"guard_seconds":t["seconds"]})
    discordant=gained+lost
    p_two_sided=(min(1,2*sum(math.comb(discordant,k) for k in range(min(gained,lost)+1))/2**discordant)
                 if discordant else 1.0)
    return {"total":100,"base_correct":sum(x["correct"] is True for x in controls),
       "guarded_correct":sum(x["correct"] is True for x in guarded),
       "gained":gained,"lost":lost,"net":gained-lost,
       "exact_mcnemar_two_sided_p":p_two_sided,
       "by_category":breakdown,"cases":rows,
       "guard_selection_counts":dict(Counter(x["selection_policy"] for x in guarded)),
       "base_errors":sum(x["status"]!="completed" for x in controls),
       "guard_errors":sum(x["status"]!="completed" for x in guarded),
       "base_latency_p50_s":round(statistics.median(x["seconds"] for x in controls),3),
       "guard_latency_p50_s":round(statistics.median(x["seconds"] for x in guarded),3),
       "base_latency_p95_s":round(sorted(x["seconds"] for x in controls)[94],3),
       "guard_latency_p95_s":round(sorted(x["seconds"] for x in guarded)[94],3)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.authorize!=AUTH or os.environ.get("NERAL_EXPERIMENTAL_TIR")!="1":
        raise RuntimeError("Explicit local experimental guarded GPU permission required")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Do not overwrite a prior transfer measurement")
    original=build()
    manifest=disjoint(original,args.train,args.holdout)
    corpus=args.corpus.read_bytes()
    if hashlib.sha256(corpus).hexdigest()!=manifest["case_sha256"]:
        raise RuntimeError("Frozen corpus SHA mismatch")
    if json.loads(args.manifest.read_text(encoding="utf8"))!=manifest:
        raise RuntimeError("Frozen manifest mismatch")
    if [json.loads(x) for x in corpus.decode("utf8").splitlines() if x]!=original:
        raise RuntimeError("Frozen corpus cases changed")
    guard_hash=require_exact_guard()
    model_hashes=require_exact_models()
    control=[];guarded=[]
    for i,case in enumerate(original):
        seed=SEED+i
        if i%2==0:
            c=control_case(case,seed)
            t=guarded_case(case,seed)
        else:
            t=guarded_case(case,seed)
            c=control_case(case,seed)
        control.append(c);guarded.append(t)
        # Both have to be GPU-resident when inspected; check last-used arm.
        last_tag=TUNED_TAG if i%2==0 else BASE_TAG
        if gpu_offload_mib(last_tag)<4096:
            raise RuntimeError("Model not RTX-resident: refuse CPU-backed scoring")
        print(f"GENERALIZATION_PROGRESS {i+1}/100",flush=True)
    summary=paired_score(original,control,guarded)
    summary.update({"status":"REAL_LOCAL_GPU_TRANSFER100_GUARDED",
       "seed":SEED,"profile":{"temperature":0,"num_predict":2048,"num_ctx":4096},
       "model_digests":model_hashes,"pr537_guard_git_blob":guard_hash,
       "train_sha256":manifest["train_sha256"],
       "prior_holdout_sha256":manifest["prior_holdout_sha256"],
       "case_sha256":manifest["case_sha256"],
       "data_scope":"new deterministic public development cases, not blinded",
       "measurement_scope":"paired direct control vs QLoRA with PR537 conservative TIR; solver tools in guarded arm only",
       "no_training":True})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf8") as f:
        json.dump(summary,f,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("TRANSFER100_SCALAR_SUMMARY "+json.dumps({
        k:v for k,v in summary.items() if k!="cases"},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
