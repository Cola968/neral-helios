"""Fresh local NERAL app A/B: qwen3:8b versus qwen3.5:9b, exact same profile.

Opt-in development pilot, 36 new private-seeded cases in six independently
checked CPU-oracle families. REAL HTTP handlers via Flask test client and the
REAL Ollama backend. No remote model, new weights, training, or GPU downloads.
Run only after the current main release smoke has completed successfully.
Private answers stay on the Windows PC outside the repo, only scalar reports
may be uploaded. NOT an independently blinded model-quality holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import statistics
import time

from scripts.neral_low_high_fresh import generate_fresh_cases
from scripts.neral_low_high_quality import _strict_integer_grade, _mcnemar, make_prompt
from scripts.neral_foundation_2048_retest import (
    SampledVram, _read_gpu_memory, ORIGINAL_DIGESTS
)
from scripts.neral_v7_foundation_tournament import OllamaLocal

MODELS=("qwen3:8b","qwen3.5:9b")
DOMAINS=("math","reasoning","code_trace")
STATUS="NERAL_FRESH_CROSSMODEL_APP_36_DEVELOPMENT_ONLY"
PROFILE="measured_nonthinking_2048"
SAFE_FAILURE_CODES=frozenset(("token_limit","timeout","incomplete",
                              "empty_answer","invalid_response","unclassified"))
BASE_URL="http://127.0.0.1:9690"

class PilotError(ValueError):
    """Aborts rather than accidentally publishing unsupported quality claims."""


def client_config(model:str)->dict:
    if model not in MODELS:
        raise PilotError("Unapproved model")
    return {"model":model, "profile":PROFILE,"think":False,"token_budget":2048}


def validate_releases_ready(receipt:dict)->bool:
    """Pure CPU policy function; caller must supply verified GitHub run receipt."""
    if (type(receipt) is not dict or receipt.get("conclusion")!="success"):
        raise PilotError("Latest release smoke is not green: do not queue GPU")
    return True


def validate_local_models(installed:dict)->dict:
    if type(installed) is not dict or any(
            installed.get(name)!=ORIGINAL_DIGESTS[name] for name in MODELS):
        raise PilotError("Only original installed qwen3:8b and qwen3.5:9b digests")
    return {name:installed[name] for name in MODELS}


def _validate_cases(cases:list[dict])->dict:
    if type(cases) is not list or len(cases)!=36:
        raise PilotError("Exactly 36 newly generated cases required")
    keys=set()
    families=set()
    domains=Counter()
    for case in cases:
        if (type(case) is not dict or
            set(case)!={"id","domain","family","kind","answer","prompt"} or
            type(case["id"]) is not str or
            not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",case["id"]) or
            case["id"] in keys or case["domain"] not in DOMAINS or
            type(case["family"]) is not str or len(case["family"])>80 or
            case["kind"]!="integer" or type(case["answer"]) is not int):
            raise PilotError("Invalid source case")
        make_prompt(case)
        keys.add(case["id"])
        families.add((case["domain"],case["family"]))
        domains[case["domain"]]+=1
    if len(families)!=6 or any(domains[d]!=12 for d in DOMAINS):
        raise PilotError("Incomplete three-domain/six-family coverage")
    return {c["id"]:c for c in cases}


def _percentile95(values:list[float])->float:
    values=sorted(values)
    return round(values[max(0,math.ceil(len(values)*.95)-1)],3)


def summarize_pairs(cases:list[dict],observations:dict,
                    vram:dict,model_digests:dict)->dict:
    source=_validate_cases(cases)
    if (type(observations) is not dict or set(observations)!=set(MODELS)
            or type(vram) is not dict or set(vram)!=set(MODELS)
            or type(model_digests) is not dict or set(model_digests)!=set(MODELS)
            or any(type(x) is not str or
                   re.fullmatch(r"sha256:[0-9a-f]{64}",x) is None
                   for x in model_digests.values())
            or len(set(model_digests.values()))!=2):
        raise PilotError("Missing model arm, pinned digest or GPU telemetry")
    totals={}
    case_correct={}
    for model in MODELS:
        arm=observations[model]
        if type(arm) is not dict or set(arm)!=set(source):
            raise PilotError("Paired app observations incomplete")
        gpu=vram[model]
        if (type(gpu) is not dict or
            set(gpu)!={"peak_used_mib","total_mib","sample_count"} or
            any(type(gpu[k]) is not int for k in gpu) or
            not 0<gpu["peak_used_mib"]<=gpu["total_mib"] or
            not 10000<=gpu["total_mib"]<=17000 or
            gpu["sample_count"]<3):
            raise PilotError("Missing or invalid sampled RTX 4070 VRAM peak")
        by_domain={d:{"count":0,"correct":0,"finished":0,"api_failures":0}
                   for d in DOMAINS}
        elapsed=[]
        ok_set=set()
        failure_codes=Counter()
        complete=errors=0
        for cid,case in source.items():
            row=arm[cid]
            allowed={"text","finished","seconds","http_error","error_code"}
            required=allowed-{"error_code"}
            if (type(row) is not dict or
                not required.issubset(set(row)) or
                not set(row).issubset(allowed) or
                type(row["text"]) is not str or
                len(row["text"].encode("utf8"))>32768 or
                type(row["finished"]) is not bool or
                type(row["http_error"]) is not bool or
                row["finished"] == row["http_error"] or
                (not row["finished"] and row["text"]!="") or
                type(row["seconds"]) not in (int,float) or
                not math.isfinite(row["seconds"]) or
                not 0<=row["seconds"]<=360):
                raise PilotError("Malformed, truncated or untrustworthy app response")
            failure_code=row.get("error_code",
                "unclassified" if row["http_error"] else None)
            if ((row["http_error"] and failure_code not in SAFE_FAILURE_CODES) or
                    (not row["http_error"] and failure_code is not None)):
                raise PilotError("Invalid private or unsupported failure code")
            if failure_code:
                failure_codes[failure_code]+=1
            d=by_domain[case["domain"]]
            d["count"]+=1
            d["finished"]+=int(row["finished"])
            d["api_failures"]+=int(row["http_error"])
            elapsed.append(row["seconds"])
            complete+=int(row["finished"])
            errors+=int(row["http_error"])
            result=_strict_integer_grade(case,{
                "text":row["text"],"complete":row["finished"]
            })
            if result=="correct":
                ok_set.add(cid)
                d["correct"]+=1
        if errors>=18:
            raise PilotError("One model failed HTTP for most tasks; pilot invalid")
        case_correct[model]=ok_set
        totals[model]={
            "correct":len(ok_set),
            "finished":complete,
            "api_failures":errors,
            "api_failure_codes":dict(sorted(failure_codes.items())),
            "by_domain":by_domain,
            "mean_seconds":round(statistics.mean(elapsed),3),
            "median_seconds":round(statistics.median(elapsed),3),
            "p95_seconds":_percentile95(elapsed),
            "sampled_peak_vram_mib":gpu["peak_used_mib"],
            "sampled_peak_vram_gib":round(gpu["peak_used_mib"]/1024,3),
            "gpu_total_vram_mib":gpu["total_mib"],
            "vram_observations":gpu["sample_count"],
            "model_digest":model_digests[model],
        }
    a,b=(case_correct[m] for m in MODELS)
    wins=len(b-a)
    losses=len(a-b)
    return {
        "schema":1,"status":STATUS,"cases_per_model":36,
        "family_count":6,"domains":list(DOMAINS),
        "model_profile":PROFILE,
        "generation_budget":2048,
        "decoder":{"think":False,"temperature":0.7,"top_p":0.8,
                   "top_k":20,"min_p":0.0,"presence_penalty":1.5},
        "models":totals,
        "paired":{"qwen35_only_correct":wins,"qwen3_only_correct":losses,
                  "both_correct":len(a & b),"neither_correct":36-len(a | b),
                  "delta_pp":round(100*(len(b)-len(a))/36,3),
                  "exact_mcnemar_p":round(_mcnemar(wins,losses),8)},
        "memory_telemetry":"sampled_gpu_wide_used_mib_not_instantaneous_process_peak",
        "private_random_seed_uploaded":False,
        "raw_prompts_or_responses_uploaded":False,
        "training_performed":False,"model_promotion_authorized":False,
        "release_validated":False,"neural_improvement_proven":False,
        "independent_holdout_verified":False,
        "caution":"36 private-randomized instances of PUBLIC generators; "
                  "real app LOW mode for both foundations, no independent "
                  "blind holdout, 1 stochastic generation per task. "
                  "No final HIGH model promotion or release authorization.",
    }


def _unload_model(model:str)->None:
    # Inference lock is held at the workflow level and each model is
    # unloaded before the next loads. No delete or download occurs.
    reply=OllamaLocal(timeout=120)._request("/api/chat",{
        "model":model,"messages":[],"keep_alive":0,"stream":False
    })
    if reply.get("done") is not True:
        raise PilotError("Local model unload not acknowledged")


def _run_arm(model:str,cases:list[dict],private_file)->tuple[dict,dict]:
    from neral_agent_autonomy import OllamaLocalBackend
    from neral_preview_server import create_preview_app

    backend=OllamaLocalBackend(model,timeout=90,inference_profile=PROFILE)
    app=create_preview_app(backend)
    client=app.test_client()
    if client.get("/",base_url=BASE_URL).status_code!=200:
        raise PilotError("Actual app route did not initialize")
    session=client.get("/api/neral/preview/session",base_url=BASE_URL)
    if session.status_code!=200 or not isinstance(session.json,dict):
        raise PilotError("Actual app secure local session failed")
    token=session.json.get("csrf")
    if not isinstance(token,str) or len(token)<24:
        raise PilotError("Missing local CSRF token")
    headers={"X-Neral-CSRF":token}
    stats={}
    with SampledVram() as sampler:
        try:
            for i,case in enumerate(cases):
                t=time.monotonic()
                response=client.post("/api/neral/chat/generate",
                    json={"prompt":make_prompt(case)},headers=headers,
                    base_url=BASE_URL)
                seconds=time.monotonic()-t
                packet=response.get_json(silent=True)
                if response.status_code==200:
                    if (type(packet) is not dict or
                        packet.get("model")!=model or
                        packet.get("inference_profile")!=PROFILE or
                        packet.get("model_origin")!="local_foundation_not_neral_adapter" or
                        packet.get("adapter_loaded") is not False or
                        packet.get("mode")!="LOW" or
                        packet.get("model_generated") is not True or
                        packet.get("tool_used") is not None or
                        type(packet.get("text")) is not str or
                        not packet["text"].strip()):
                        raise PilotError("Unexpected tool override, provenance, or app output")
                    raw=packet["text"]
                    item={"text":raw,"finished":True,"error_code":None,
                          "http_error":False,"seconds":round(seconds,3)}
                elif response.status_code==503:
                    # A partial model answer MUST NOT be scored as a final.
                    # Only fixed local backend codes are allowed in scalar
                    # metrics; never include exception strings or raw output.
                    raw=""
                    code=(packet.get("error_code") if type(packet) is dict
                          else None)
                    code=code if code in SAFE_FAILURE_CODES else "unclassified"
                    item={"text":"","finished":False,"error_code":code,
                          "http_error":True,"seconds":round(seconds,3)}
                else:
                    raise PilotError("App security, route, or protocol unexpectedly failed")
                stats[case["id"]]=item
                private_file.write(json.dumps({
                    "case_id":case["id"],"model":model,
                    "question_private":case["prompt"],
                    "reference_private":case["answer"],
                    "response_private":raw,
                    "finished":item["finished"],
                    "latency_seconds":item["seconds"],
                },ensure_ascii=False)+"\n")
                private_file.flush()
                if (i+1)%6==0:
                    print(f"PRIVATE_LOCAL_APP_PROGRESS model={model} completed={i+1}/36",
                          flush=True)
        finally:
            _unload_model(model)
    return stats,sampler.report()


def run_pilot(out_dir:Path)->dict:
    out_dir=Path(out_dir)
    root=Path(__file__).resolve().parents[1]
    if (out_dir.exists() or out_dir.is_symlink() or
        not out_dir.parent.is_dir() or
        out_dir.parent.is_symlink() or
        out_dir.resolve().is_relative_to(root)):
        raise PilotError("Create-only PRIVATE local directory outside checkout")
    model_digests=validate_local_models(OllamaLocal(timeout=120).inventory())
    used,total=_read_gpu_memory()
    if total-used<7000:
        raise PilotError("GPU not free: refuse local concurrent inference")

    # Never publish the generated seed or raw per-case prompts to GitHub.
    cases=generate_fresh_cases(secrets.randbits(64))
    _validate_cases(cases)
    out_dir.mkdir(exist_ok=False)
    obs={}
    vram={}
    with (out_dir/"private-fresh-app-responses.jsonl").open("x",encoding="utf8") as raw:
        for model in MODELS:
            now,available_total=_read_gpu_memory()
            if available_total-now<7000:
                raise PilotError("Other model not unloaded or GPU occupied")
            obs[model],vram[model]=_run_arm(model,cases,raw)
    if validate_local_models(OllamaLocal(timeout=120).inventory())!=model_digests:
        raise PilotError("Installed Ollama model changed during test")
    report=summarize_pairs(cases,obs,vram,model_digests)
    scalar=out_dir/"scalar-summary.json"
    with scalar.open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "status":report["status"],"models":report["models"],
        "paired":report["paired"],"release_validated":False,
        "raw_prompts_or_responses_uploaded":False,
    },sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run_pilot(args.out_dir)


if __name__=="__main__":
    main()
