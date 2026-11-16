"""GPU-only new 24 development pairs: unchanged Qwen3.5:9B vs NERAL1.1.

Not a future blind test. Never reads the burned e80c6017 corpus, trains,
updates main, or rewrites earlier score evidence. Requires strict one-shot
pin, RTX4070, local Ollama and existing Docker. Streams complete raw pairs
to unique files before generating any metric.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from scripts.neral_1_1_postverify_dev24_cases import build,manifest
from scripts.neral_1_1_prospective_blind100 import (
    require_gpu,run_case,write_new,append_case,json_bytes,sha
)

AUTH="NERAL_1_1_POSTVERIFY_DEV24_LOCAL_GPU_ONCE"
CATS=("mathematics","logic","programming","mixed")

def require_pin():
    sha_env=os.environ.get("NERAL_PINNED_SOURCE_SHA")
    if type(sha_env) is not str or len(sha_env)!=40 or any(
            c not in "0123456789abcdef" for c in sha_env):
        raise RuntimeError("Pinned experiment SHA required")
    actual=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,
                          text=True,timeout=12,check=True).stdout.strip()
    if actual!=sha_env:
        raise RuntimeError("Pinned code checkout drift")
    return actual

def gpu_mib()->int|None:
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=memory.used",
                  "--format=csv,noheader,nounits"],
                  capture_output=True,text=True,timeout=8,check=True).stdout.strip()
        return int(result.splitlines()[0].strip())
    except Exception:return None

def offline_metrics(cases,rows):
    if (len(cases)!=24 or len(rows)!=24
            or len({x["id"] for x in rows})!=24
            or [x["id"] for x in cases]!=[x["id"] for x in rows]):
        raise ValueError("Only 24 ordered, complete and distinct pairs may be scored")
    bycat={}
    for cat in CATS:
        selection=[x for x in rows if x["category"]==cat]
        if len(selection)!=6:raise ValueError("Missing category")
        bycat[cat]={
          "base":sum(x["base_strict_correct"] for x in selection),
          "neral":sum(x["neral_strict_correct"] for x in selection),
          "gained":sum(x["neral_strict_correct"] and not x["base_strict_correct"] for x in selection),
          "regressed":sum(x["base_strict_correct"] and not x["neral_strict_correct"] for x in selection)}
    won=sum(x["neral_strict_correct"] and not x["base_strict_correct"] for x in rows)
    lost=sum(x["base_strict_correct"] and not x["neral_strict_correct"] for x in rows)
    b=sum(x["base_strict_correct"] for x in rows)
    n=sum(x["neral_strict_correct"] for x in rows)
    verified=[x for x in rows if x["independent_verified"]]
    z3=[x for x in rows if x["independent_verified"] and x["proof_backend"]=="z3_unsat"]
    py=[x for x in rows if x["independent_verified"] and x["proof_backend"]!="z3_unsat"]
    times_base=sorted(x["base_seconds"] for x in rows)
    times_neral=sorted(x["neral_seconds"] for x in rows)
    classifications=Counter(x["selection_policy"] for x in rows)
    return {
       "development_not_blind":True,"complete_pairs":24,
       "base_correct":b,"neral_correct":n,
       "gained":won,"regressions":lost,"net_gain":n-b,
       "category_breakdown":bycat,
       "z3_verified":len(z3),
       "python_independently_verified":len(py),
       "total_independently_verified":len(verified),
       "z3_verification_rate":len(z3)/24,
       "python_verification_rate":len(py)/24,
       "qlora_proposal_calls":sum(x["proposal_calls"] for x in rows),
       "python_docker_primary_calls":sum(x["docker_calls"] for x in rows),
       "qwen_fallback_calls":sum(x["fallback_calls"] for x in rows),
       "selection_counts":dict(classifications),
       "base_median_s":statistics.median(times_base),
       "neral_median_s":statistics.median(times_neral),
       "base_p95_s":times_base[22],"neral_p95_s":times_neral[22],
       "max_observed_gpu_mib":max((x.get("gpu_mib") or 0 for x in rows),default=0),
       "release_gate_passed":bool(n>b and lost==0 and len(z3)>0 and
                  len(py)>0 and sum(x["proposal_calls"] for x in rows)>0
                  and sum(x["docker_calls"] for x in rows)>0)
    }

def cpu_preflight():
    cases=build(); receipt=manifest()
    assert len(cases)==24 and receipt["cases"]==24
    assert receipt["categories"]=={c:6 for c in CATS}
    assert len({x["question_sha256"] for x in cases})==24
    from neral_python_post_source import parse_post_source,original_python_program
    from neral_nl_verified_route import strict_source_certificate
    for c in cases:
        if c["family"]=="graph_coloring_z3":
            assert strict_source_certificate(c["question"]) is not None
        elif c["category"]=="programming":
            assert original_python_program(c["question"]) is not None
        elif c["family"]!="unsupported_free_logic":
            task=parse_post_source(c["question"])
            assert task is not None and task.answer==c["gold_integer"]
    print("POSTVERIFY_DEV24_CPU_PREFLIGHT_OK cases=24 distinct=24 categories=6_each",flush=True)

def live(out:Path):
    pin=require_pin()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Unique GPU development output path required")
    if out.parent.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError("No development artifact inside source checkout")
    hardware=require_gpu()
    cpu_preflight()
    cases=build()
    provenance={"status":"DEV24_FROZEN_BEFORE_MODEL_CALL",
        "source_sha":pin,"casepack_sha256":manifest()["sha256"],
        "model_digests":hardware["model_digests"],"gpu":hardware["gpu"],
        "no_training":True,"no_main_write":True,
        "no_burned_blind100_access":True,
        "strict_output_format":"FINAL_ANSWER: <integer> final line"}
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/"dev24-manifest.json",json_bytes(provenance))
    write_new(out/"dev24-cases.jsonl",b"".join(json_bytes(c) for c in cases))
    progress=out/"dev24-progress.jsonl"
    rawfile=out/"dev24-full-raw.jsonl"
    write_new(progress,b"");write_new(rawfile,b"")
    from neral_nl_local_ollama import _localhost_transport
    finished=[]
    started=time.monotonic()
    for i,case in enumerate(cases):
        try:
            result=run_case(case,_localhost_transport)
        except Exception as exc:
            write_new(out/"dev24-failure.json",json_bytes({
                "case_index":i,"id":case["id"],
                "error_class":type(exc).__name__,"completed_pairs":len(finished),
                "casepack_sha256":manifest()["sha256"]}))
            raise
        raw=result.pop("raw")
        result["gpu_mib"]=gpu_mib()
        append_case(rawfile,{"id":case["id"],**raw})
        append_case(progress,result)
        finished.append(result)
        print("POSTVERIFY_DEV24_PROGRESS "+str(len(finished))+"/24",flush=True)
    summary=offline_metrics(cases,finished)
    summary.update({
        "status":"DEV24_GPU_COMPLETE_PAIRED_RAW",
        "source_sha":pin,
        "casepack_sha256":manifest()["sha256"],
        "full_raw_sha256":sha(rawfile.read_bytes()),
        "progress_sha256":sha(progress.read_bytes()),
        "total_seconds":round(time.monotonic()-started,3),
        "no_training":True,"no_main_write":True,
        "zero_future_regressions_not_proven":True,
        "next_new_blind100_authorized":False})
    write_new(out/"dev24-summary.json",json_bytes(summary))
    print("POSTVERIFY_DEV24_FINAL "+json.dumps(summary,sort_keys=True),flush=True)
    print("DEV24_RELEASE_GATE "+("PASS" if summary["release_gate_passed"] else "BLOCKED"),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--cpu-preflight",action="store_true")
    p.add_argument("--live",action="store_true")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.authorize!=AUTH or a.cpu_preflight==a.live or (a.live and a.out is None):
        p.error("Explicit local dev24 authorization + exactly one mode required")
    if a.cpu_preflight:cpu_preflight()
    else:live(a.out)
if __name__=="__main__":
    main()
