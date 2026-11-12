"""Opt-in production dual route: QLoRA for proposals, matching untuned Qwen for raw.

Only localhost Ollama; require TWO pinned model digests before ANY inference.
No dynamic keyword router, model retraining, model promotion or prompt logging.
The exact PR537 conservative response guard is the only selection policy.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time

from neral_tir_experimental import run_experimental_tir
from neral_tir_regression_guard import choose_guarded_tir_reply
from neral_turn0_extractor import extraction_response_format
from scripts.neral_generalization_100_run import (
    BASE_TAG, TUNED_TAG, BASE_OPTIONS, require_exact_models,
    require_exact_guard, VALID_POLICIES,
)
from scripts.neral_qlora_holdout_live import request

_LIMITED_CODES=frozenset({
    "accepted","invalid_input","kind_none","invalid_json","solver_type_error",
    "solver_unsat_fallback","solver_rejected","invalid_structure",
    "variable_not_in_source","other","extraction_none","direct_python_source",
    "extracted_formal","unstructured",
})

def sample_residency()->dict:
    """Read observed Ollama residency; not an estimate of total GPU demand."""
    response=request("GET","/api/ps")
    models=response.get("models")
    if type(models) is not list or len(models)>100:
        raise RuntimeError("Invalid local Ollama process inventory")
    names={}
    for row in models:
        if type(row) is not dict:continue
        name=row.get("name")
        if name not in (BASE_TAG,TUNED_TAG):continue
        size=row.get("size_vram",0)
        if type(size) is not int or size<0:continue
        names[name]=size//(1024*1024)
    return {"models_vram_mib":names,
            "both_resident":BASE_TAG in names and TUNED_TAG in names}

def sample_device_mib()->int|None:
    """Sample real nvidia-smi usage without raw process names or memory logs."""
    try:
        r=subprocess.run(
            ["nvidia-smi","--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],capture_output=True,
             text=True,timeout=4,check=False)
        if r.returncode!=0:return None
        first=r.stdout.strip().splitlines()[0].strip()
        return int(first) if first.isdecimal() else None
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
        return None

def _ollama_chat(model:str,messages:list[dict],seed:int,*,repair:bool=False,
                 structured:bool=False)->tuple[str,dict]:
    if model not in (BASE_TAG,TUNED_TAG) or type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("Model or seed differs from pinned local dual route")
    if type(messages) is not list or not messages or type(messages[-1]) is not dict:
        raise ValueError("Invalid local chat turns")
    if structured and model!=TUNED_TAG:
        raise ValueError("Only trained adapter may propose schemas")
    max_predict=(1536 if repair else 384) if structured else BASE_OPTIONS["num_predict"]
    options={**BASE_OPTIONS,"seed":seed,"num_predict":max_predict}
    packet={"model":model,"messages":messages,"think":False,"stream":False,
            "options":options,"keep_alive":"5m"}
    if structured:
        q=messages[-1].get("content")
        if type(q) is not str:raise ValueError("Nontext schema question")
        if repair:
            original=json.loads(q).get("original_question")
            if type(original) is not str:
                raise ValueError("Repair missing source question")
            q=original
        packet["format"]=extraction_response_format(q)
    started=time.monotonic()
    data=request("POST","/api/chat",packet,timeout=180)
    spent=round((time.monotonic()-started)*1000,1)
    if data.get("done") is not True or data.get("done_reason")!="stop":
        raise RuntimeError("Incomplete local Ollama inference")
    msg=data.get("message")
    text=msg.get("content") if type(msg) is dict else None
    if type(text) is not str or not text.strip() or len(text.encode("utf8"))>32768:
        raise RuntimeError("Invalid or oversized local response")
    if structured and len(text.encode("utf8"))>4096:
        raise RuntimeError("Oversized untrusted schema")
    load=data.get("load_duration")
    total=data.get("total_duration")
    return text,{"model":model,"structured":structured,
        "elapsed_ms":spent,
        "load_ms":round(load/1e6,1) if type(load) is int and load>=0 else None,
        "ollama_total_ms":round(total/1e6,1) if type(total) is int and total>=0 else None,
        "model_residency_after":sample_residency()}

def run_dual_route(messages:list[dict],*,seed:int=0,
                   sample_gpu:bool=True)->dict:
    """Exact same controller and guard as GPU ablation; real API is opt-in.

    The question is never included in telemetry. No route by task wording.
    Raw internal Controller fallback and Guard fallback BOTH invoke BASE.
    """
    if (type(messages) is not list or len(messages)<2
            or any(type(x) is not dict or set(x)!={"role","content"}
                   or type(x["content"]) is not str
                   for x in messages)
            or messages[0]["role"]!="system" or messages[-1]["role"]!="user"):
        raise ValueError("Invalid NERAL dialogue")
    model_digests=require_exact_models()
    guard_blob=require_exact_guard()
    original=[dict(x) for x in messages]
    metrics=[]
    samples=[]
    residency_before=sample_residency()
    stop=threading.Event()

    def monitor():
        while not stop.is_set():
            v=sample_device_mib()
            if v is not None:samples.append(v)
            stop.wait(.25)
    monitor_thread=threading.Thread(target=monitor,daemon=True)
    if sample_gpu:
        monitor_thread.start()
    started=time.monotonic()
    extractor_calls=0
    raw_calls=0
    tool_model_calls=0

    def chat(model,turns,structured=False,repair=False):
        output,event=_ollama_chat(model,turns,seed,structured=structured,
                                  repair=repair)
        metrics.append(event)
        return output

    def respond(turns):
        nonlocal raw_calls,tool_model_calls
        if turns==original:
            raw_calls+=1
            return chat(BASE_TAG,turns)
        tool_model_calls+=1
        return chat(TUNED_TAG,turns)

    def extract(turns):
        nonlocal extractor_calls
        extractor_calls+=1
        if extractor_calls>2:
            raise RuntimeError("Too many untrusted schema attempts")
        return chat(TUNED_TAG,turns,structured=True,
                    repair=(extractor_calls==2))

    def raw_fallback(turns):
        nonlocal raw_calls
        if turns!=original:
            raise ValueError("Guard raw fallback is not the original dialogue")
        raw_calls+=1
        return chat(BASE_TAG,turns)

    try:
        proposal=run_experimental_tir(original,respond,extract_constraints=extract)
        selected=choose_guarded_tir_reply(proposal,original,raw_fallback)
        policy=selected.get("selection_policy")
        if policy not in VALID_POLICIES:
            raise RuntimeError("Unexpected conservative guard decision")
        if policy.startswith("raw_qwen") and raw_calls!=1:
            raise RuntimeError("Raw fallback must call untrained Qwen exactly once")
        if policy=="independently_verified_tir" and raw_calls!=0:
            raise RuntimeError("Verified tool may not be replaced by raw model")
        text=selected.get("text")
        if type(text) is not str or not text.strip():
            raise RuntimeError("Empty guarded answer")
        return {**selected,
            "dual_routing":True,
            "experimental_tir":True,"conservative_tir":True,
            "tool_model":TUNED_TAG,"fallback_model":BASE_TAG,
            "model_digests":model_digests,
            "guard_git_blob":guard_blob,
            "structured_schema_attempts":extractor_calls,
            "tool_planning_calls":tool_model_calls,
            "base_raw_calls":raw_calls,
            "latency_ms":round((time.monotonic()-started)*1000,1),
            "gpu_device_peak_mib":max(samples,default=None),
            "gpu_sample_count":len(samples),
            "ollama_resident_before":residency_before,
            "ollama_resident_after":sample_residency(),
            "inference_spans":metrics,
            "response_mode":"local_pr537_dual_route"}
    finally:
        stop.set()
        if sample_gpu:
            monitor_thread.join(timeout=5)
