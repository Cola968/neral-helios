"""Manual paired TIR12 GPU development diagnostic, never release approval.

One RTX 4070; 4 math, 4 reasoning and 4 code-trace cases.
A = direct Qwen3.5, C = NERAL HIGH + experimental two-stage TIR.
Same model/first-pass decoding, but DIFFERENT total calls and compute budget.
Never export model text, generated code, observations, seeds or user prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time

from scripts.neral_high12_manual import (
    select_cases, score_pairs, paired_case_seed, _FAILURE_REASONS, High12GateError, MODEL, PROFILE,
)
from scripts.neral_same_base_ab_score import TOOLS

AUTHORIZATION="NERAL_TIR12_GPU_MANUAL_ONLY"
ARMS=("raw_qwen35","neral_high_router")
_TARGET=re.compile(r"[0-9a-f]{40}\Z")

class Tir12GateError(ValueError):
    """Sanitized error; do not put private generated text in exceptions."""


def score_tir_pairs(cases: list[dict], observations: list[dict], *,
                    model_digest: str) -> dict:
    """Reuse the exact HIGH12 scorer for both arms; only add tool provenance."""
    if type(observations) is not list or len(observations)!=12:
        raise Tir12GateError("Expected 12 paired TIR observations")
    converted=[]
    triggered=succeeded=fallback_after_tool=0
    for row in observations:
        if not isinstance(row,dict) or set(row)!={"id","prompt_sha256","arms"}:
            raise Tir12GateError("Missing paired case provenance")
        arms=row["arms"]
        if type(arms) is not dict or set(arms)!=set(ARMS):
            raise Tir12GateError("Missing raw or two-stage TIR arm")
        normalized={}
        for arm in ARMS:
            obs=arms[arm]
            if (type(obs) is not dict or set(obs)!={
                    "text","complete","failed","failure_reason","seconds",
                    "model","model_digest","profile","tool_used",
                    "tool_calls_triggered","tool_calls_succeeded"}):
                raise Tir12GateError("Malformed observation metadata")
            t,u=obs["tool_calls_triggered"],obs["tool_calls_succeeded"]
            if (type(t) is not int or type(u) is not int
                    or not 0<=u<=t<=1
                    or (arm=="raw_qwen35" and (t or u or obs["tool_used"] is not None))
                    or (obs["tool_used"] is not None and (not t or not u))
                    or obs["tool_used"] not in TOOLS
                    or (obs["failed"] and (t or u or obs["tool_used"] is not None))):
                raise Tir12GateError("Forged or invalid tool evidence")
            if obs["failed"] and obs["failure_reason"] not in _FAILURE_REASONS:
                raise Tir12GateError("Unknown failure category")
            if arm=="neral_high_router":
                triggered+=t;succeeded+=u
                if t and obs["tool_used"] is None:
                    fallback_after_tool+=1
            normalized["neral_high" if arm=="neral_high_router" else arm]={
                key:value for key,value in obs.items()
                if key not in ("tool_calls_triggered","tool_calls_succeeded")}
        converted.append({"id":row["id"],"prompt_sha256":row["prompt_sha256"],
                          "arms":normalized})
    try:
        report=score_pairs(cases,converted,model_digest=model_digest)
    except (High12GateError,TypeError,KeyError) as exc:
        raise Tir12GateError("Invalid paired TIR observations") from None
    def remap(data):
        return {("neral_high_router" if key=="neral_high" else key):value
                for key,value in data.items()}
    for field in ("failed_calls","error_counts","technical_failures",
                  "format_failures","median_seconds"):
        report[field]=remap(report[field])
    report["errors_by_domain"]={
        domain:remap(arms) for domain,arms in report["errors_by_domain"].items()}
    for row in report["by_domain"].values():
        row["tir_correct"]=row.pop("high_correct")
    report["tir_correct"]=report.pop("high_correct")
    report.update(
        status="NERAL_TIR12_PAIRED_DEVELOPMENT_ONLY",
        model_weights_improved=False,release_approval=False,
        automatic_merge_allowed=False,independent_holdout_verified=False,
        experimental_tir=True,same_inference_budget=False,
        first_pass_token_budget=2048,
        tool_calls_triggered=triggered,tool_calls_succeeded=succeeded,
        tool_attempt_fallback_signals=fallback_after_tool,
        limitations="Twelve known-family development tasks, with extra TIR/tool calls; no blind release guarantee.")
    return report


def run_gpu(*, authorize: str, target_sha: str, high_sha: str,
            output: Path) -> bool:
    if (authorize!=AUTHORIZATION or os.environ.get("NERAL_EXPERIMENTAL_TIR")!="1"
            or not isinstance(target_sha,str) or not _TARGET.fullmatch(target_sha)
            or not isinstance(high_sha,str) or not _TARGET.fullmatch(high_sha)):
        raise Tir12GateError("Manual TIR authorization or target SHAs invalid")
    output=Path(output)
    if (output.exists() or output.is_symlink() or output.suffix!=".json"
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise Tir12GateError("Fresh scalar-only output file required")
    from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
    from scripts.neral_same_base_live import _high_client, require_model
    from scripts.neral_tir_hard_live import _run_one
    from scripts.neral_low_high_quality import make_prompt
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    from neral_agent_autonomy import OllamaLocalBackend

    digest=require_model(OllamaLocal(timeout=120).inventory())
    if digest!=ORIGINAL_DIGESTS[MODEL]:
        raise Tir12GateError("Model digest mismatch")
    backend=OllamaLocalBackend(MODEL,timeout=90,inference_profile=PROFILE)
    client,headers=_high_client(backend)
    seed=secrets.randbits(64)
    cases=select_cases(seed)
    rows=[]
    try:
        for i,case in enumerate(cases):
            backend.set_benchmark_seed(paired_case_seed(seed,case["id"]))
            prompt=make_prompt(case)
            obs={"id":case["id"],
                 "prompt_sha256":hashlib.sha256(prompt.encode("utf8")).hexdigest(),
                 "arms":{}}
            order=ARMS if i%2==0 else tuple(reversed(ARMS))
            for arm in order:
                result=_run_one(arm,prompt,backend,client,headers)
                text,complete,failed,seconds,tool,t,u,reason=result
                obs["arms"][arm]={
                    "text":text if complete else "","complete":complete,
                    "failed":failed,"failure_reason":reason if failed else None,
                    "seconds":round(seconds,3),"tool_used":tool,
                    "tool_calls_triggered":t,"tool_calls_succeeded":u,
                    "model":MODEL,"model_digest":digest,"profile":PROFILE}
            rows.append(obs)
            if (i+1)%2==0:
                print("TIR12_PAIR_PROGRESS "+str(i+1)+"/12",flush=True)
    finally:
        backend.set_benchmark_seed(None)
        backend.unload()
    if require_model(OllamaLocal(timeout=120).inventory())!=digest:
        raise Tir12GateError("Model digest changed")
    result=score_tir_pairs(cases,rows,model_digest=digest)
    result["seed_option_present"]=True
    result["target_sha"]=target_sha
    result["high_sha"]=high_sha
    result["seed_commitment_sha256"]=hashlib.sha256(
        str(seed).encode("ascii")).hexdigest()
    result["elapsed_type"]="actual_local_inference_only"
    with output.open("x",encoding="utf8") as handle:
        json.dump(result,handle,ensure_ascii=False,sort_keys=True,allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status":result["status"],"raw_correct":result["raw_correct"],
        "tir_correct":result["tir_correct"],"paired_wins":result["paired_wins"],
        "paired_losses":result["paired_losses"],
        "tool_calls_succeeded":result["tool_calls_succeeded"],
        "development_gate_passed":result["development_gate_passed"],
    },sort_keys=True),flush=True)
    return result["development_gate_passed"]


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--authorize",required=True)
    parser.add_argument("--target-sha",required=True)
    parser.add_argument("--high-sha",required=True)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    try:
        passed=run_gpu(authorize=args.authorize,target_sha=args.target_sha,
                       high_sha=args.high_sha,output=args.out)
    except Exception as exc:
        print("TIR12_ABORT: "+type(exc).__name__,flush=True)
        return 2
    return 0 if passed else 1

if __name__=="__main__":raise SystemExit(main())
